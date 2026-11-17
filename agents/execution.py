"""
Execution Agent - entry market + TP LIMIT + SL STOP_MARKET
- Isolated margin + leverage
- ML on_open with risk_amount / notional for R-multiple labels
- TP: LIMIT reduceOnly
- SL: STOP_MARKET only (LIMIT SL fills immediately = bug)
- Size clamped to exchange min/max quantity (avoid -4005)
- Cancel old TP/SL before placing one pair on full position size
- Block same symbol twice in one cycle
"""
from typing import Optional, Set, Tuple
from loguru import logger
from core.exchange import ExchangeWrapper
from core.models import Signal, Side, Mode
from agents.risk_gate import RiskGate


class ExecutionAgent:
    def __init__(
        self,
        exchange: ExchangeWrapper,
        risk_gate: RiskGate,
        default_leverage: int = 5,
        max_leverage: int = 10,
        order_logger=None,
        ml_agent=None,
        ohlcv_cache: dict = None,
    ):
        self.exchange = exchange
        self.risk_gate = risk_gate
        self.default_leverage = max(1, int(default_leverage))
        self.max_leverage = max(self.default_leverage, int(max_leverage))
        self.order_logger = order_logger
        self.ml_agent = ml_agent
        self.ohlcv_cache = ohlcv_cache if ohlcv_cache is not None else {}
        self.skip_symbols: Set[str] = set()
        # Symbols already executed this cycle (block duplicate TP/SL stacks)
        self._cycle_symbols: Set[str] = set()

    def _choose_leverage(self, signal: Signal) -> int:
        lev = self.default_leverage
        try:
            entry = float(signal.entry)
            sl = float(signal.sl)
            if entry > 0 and sl > 0:
                stop_pct = abs(entry - sl) / entry * 100
                if stop_pct >= 4.0:
                    lev = max(2, self.default_leverage // 3)
                elif stop_pct >= 2.5:
                    lev = max(3, self.default_leverage // 2)
                elif stop_pct <= 0.6:
                    lev = min(self.max_leverage, self.default_leverage + 2)
        except Exception:
            pass
        return max(1, min(lev, self.max_leverage))

    def begin_cycle(self) -> None:
        """Call at start of each main loop cycle — clear per-cycle symbol lock."""
        self._cycle_symbols.clear()

    def _exchange_has_symbol(self, symbol: str) -> bool:
        """Live check: any open position on this symbol on the exchange."""
        try:
            for p in self.exchange.get_positions() or []:
                if getattr(p, "symbol", None) == symbol and abs(float(getattr(p, "size", 0) or 0)) > 0:
                    return True
        except Exception as e:
            logger.debug(f"live position check failed: {e}")
        return False

    def _position_size_on_exchange(self, symbol: str, side: Side) -> float:
        """Full absolute size for symbol+side on exchange (0 if none)."""
        try:
            for p in self.exchange.get_positions() or []:
                if getattr(p, "symbol", None) != symbol:
                    continue
                pside = getattr(p, "side", None)
                pside_val = pside.value if hasattr(pside, "value") else str(pside or "")
                want = side.value if hasattr(side, "value") else str(side)
                if str(pside_val).upper() != str(want).upper():
                    continue
                return abs(float(getattr(p, "size", 0) or 0))
        except Exception as e:
            logger.debug(f"position size lookup failed: {e}")
        return 0.0


    def _peak_equity(self) -> float:
        """Peak equity để tính drawdown (lưu trên risk_gate)."""
        peak = float(getattr(self.risk_gate, "peak_equity", 0) or 0)
        eq = float(getattr(self.risk_gate, "equity", 0) or 0)
        if eq > peak:
            peak = eq
            try:
                self.risk_gate.peak_equity = peak
            except Exception:
                pass
        if peak <= 0:
            peak = eq if eq > 0 else 1.0
            try:
                self.risk_gate.peak_equity = peak
            except Exception:
                pass
        return peak

    def _dynamic_risk_amount(self, signal: Signal, equity: float) -> tuple:
        """Quản lý vốn động: equity% × DD × streak × ATR × confidence.

        Returns (risk_amount, detail_dict). risk_amount=0 → không vào lệnh.
        """
        cfg = self.risk_gate.cfg or {}
        base_pct = float(cfg.get("risk_per_trade_pct", 0.01) or 0.01)
        dyn = cfg.get("dynamic_risk") or {}
        enabled = bool(dyn.get("enabled", True))

        detail = {
            "enabled": enabled,
            "base_pct": base_pct,
            "mult_dd": 1.0,
            "mult_streak": 1.0,
            "mult_vol": 1.0,
            "mult_conf": 1.0,
            "mult_style": 1.0,
            "mult_exposure": 1.0,
            "dd_pct": 0.0,
            "final_pct": base_pct,
        }

        if not enabled or equity <= 0:
            amt = max(0.0, equity * base_pct)
            detail["risk_amount"] = amt
            return amt, detail

        mult = 1.0

        # --- Drawdown vs peak ---
        peak = self._peak_equity()
        dd_pct = max(0.0, (peak - equity) / peak) if peak > 0 else 0.0
        detail["dd_pct"] = round(dd_pct, 4)
        dd_steps = dyn.get("dd_steps") or [
            {"dd_pct": 0.03, "mult": 0.7},
            {"dd_pct": 0.06, "mult": 0.4},
            {"dd_pct": 0.10, "mult": 0.0},
        ]
        # apply highest matching threshold
        dd_mult = 1.0
        for step in sorted(dd_steps, key=lambda x: float(x.get("dd_pct", 0))):
            try:
                if dd_pct >= float(step.get("dd_pct", 0)):
                    dd_mult = float(step.get("mult", 1.0))
            except (TypeError, ValueError):
                pass
        detail["mult_dd"] = dd_mult
        mult *= dd_mult
        if dd_mult <= 0:
            detail["final_pct"] = 0.0
            detail["risk_amount"] = 0.0
            detail["reason"] = f"DD {dd_pct:.1%} → stop new entries"
            return 0.0, detail

        # --- Consecutive losses ---
        losses = int(getattr(self.risk_gate, "consecutive_losses", 0) or 0)
        streak_map = dyn.get("loss_streak_mult") or {2: 0.75, 3: 0.5, 5: 0.25}
        # keys may be str in yaml
        streak_mult = 1.0
        try:
            items = []
            for k, v in streak_map.items():
                items.append((int(k), float(v)))
            items.sort(key=lambda x: x[0])
            for thr, m in items:
                if losses >= thr:
                    streak_mult = m
        except Exception:
            streak_mult = 1.0
        detail["mult_streak"] = streak_mult
        detail["losses"] = losses
        mult *= streak_mult

        # --- Volatility (ATR% of price) ---
        if dyn.get("atr_vol_scale", True):
            try:
                atr_pct = None
                extra = getattr(signal, "extra", None) or {}
                if isinstance(extra, dict) and extra.get("atr_pct") is not None:
                    atr_pct = float(extra["atr_pct"])
                if atr_pct is None and signal.entry and signal.sl:
                    # proxy: stop distance relative
                    atr_pct = abs(float(signal.entry) - float(signal.sl)) / float(signal.entry)
                if atr_pct is not None:
                    # soft scale: 1% stop → ~1.0; 3% → ~0.55; 5% → ~0.35
                    ref = float(dyn.get("atr_ref_pct", 0.01) or 0.01)
                    vol_mult = min(1.2, max(0.35, ref / max(atr_pct, 1e-6)))
                    # only reduce when more volatile than ref
                    if atr_pct > ref:
                        detail["mult_vol"] = round(vol_mult, 3)
                        mult *= vol_mult
                    detail["atr_pct"] = round(atr_pct, 5)
            except Exception:
                pass

        # --- Confidence / ML proba ---
        if dyn.get("confidence_scale", True):
            try:
                conf = float(getattr(signal, "confidence", 0.6) or 0.6)
                conf = max(0.0, min(1.0, conf))
                # conf 0.5 → 0.7x; 0.8 → 1.0x; 0.9+ → 1.05x (cap)
                conf_mult = 0.4 + conf * 0.7  # 0.5→0.75, 0.8→0.96, 1.0→1.1
                conf_mult = max(0.4, min(1.1, conf_mult))
                detail["mult_conf"] = round(conf_mult, 3)
                detail["confidence"] = conf
                mult *= conf_mult
            except Exception:
                pass

        # --- Style multiplier ---
        style_mults = dyn.get("style_mult") or {"scalp": 0.8, "day": 1.0, "swing": 1.0}
        try:
            style = ""
            if hasattr(self.risk_gate, "resolve_trade_style"):
                style = self.risk_gate.resolve_trade_style(
                    getattr(signal, "strategy", "") or "",
                    getattr(signal, "timeframe", "") or "",
                )
            sm = float(style_mults.get(style, 1.0) or 1.0)
            detail["mult_style"] = sm
            detail["style"] = style
            mult *= sm
        except Exception:
            pass

        # --- Exposure soft scale ---
        if dyn.get("exposure_scale", True):
            try:
                positions = getattr(self.risk_gate, "open_positions", {}) or {}
                exposure = sum(
                    float(p.size or 0) * float(p.entry_price or 0) for p in positions.values()
                )
                max_exp = equity * float(cfg.get("max_total_exposure_pct", 0.3) or 0.3)
                if max_exp > 0 and exposure > 0:
                    fill = exposure / max_exp
                    # 0% → 1.0; 80% → 0.5; 100% → 0.25
                    exp_mult = max(0.25, 1.0 - 0.75 * min(fill, 1.0))
                    detail["mult_exposure"] = round(exp_mult, 3)
                    detail["exposure_fill"] = round(fill, 3)
                    mult *= exp_mult
            except Exception:
                pass

        final_pct = base_pct * mult
        min_pct = float(dyn.get("min_risk_pct", 0.001) or 0.001)
        max_pct = float(dyn.get("max_risk_pct", base_pct * 1.2) or base_pct * 1.2)
        final_pct = max(min_pct, min(max_pct, final_pct))
        # if mult collapsed near zero via dd, already returned
        if mult < 1e-9:
            detail["final_pct"] = 0.0
            detail["risk_amount"] = 0.0
            return 0.0, detail

        risk_amount = equity * final_pct
        detail["final_pct"] = round(final_pct, 6)
        detail["risk_amount"] = round(risk_amount, 4)
        detail["mult_total"] = round(mult, 4)
        return risk_amount, detail

    def execute(self, signal: Signal) -> Optional[dict]:

        if signal.symbol in self.skip_symbols:
            logger.debug(f"Skip {signal.symbol} (in skip list)")
            return None

        # Same cycle: only one entry per symbol (prevents stacked TP/SL)
        if signal.symbol in self._cycle_symbols:
            logger.warning(
                f"Skip {signal.symbol} {signal.side.value}: already executed this cycle"
            )
            return None

        # Live exchange: already has position on this symbol
        if self._exchange_has_symbol(signal.symbol):
            logger.warning(
                f"RiskGate REJECTED {signal.symbol} {signal.side.value}: "
                f"EXCHANGE ALREADY HAS POSITION ({signal.symbol}) [live]"
            )
            self._cycle_symbols.add(signal.symbol)
            return None

        approved, reason = self.risk_gate.check(signal)
        if not approved:
            logger.warning(f"RiskGate REJECTED {signal.symbol} {signal.side.value}: {reason}")
            return None

        equity = self.exchange.get_balance()
        self.risk_gate.update_equity(equity)
        # peak equity cho drawdown động
        try:
            peak = float(getattr(self.risk_gate, "peak_equity", 0) or 0)
            if equity > peak:
                self.risk_gate.peak_equity = equity
        except Exception:
            pass

        risk_amount, dyn_detail = self._dynamic_risk_amount(signal, float(equity or 0))
        if risk_amount <= 0:
            logger.warning(
                f"Dynamic risk SKIP {signal.symbol}: {dyn_detail.get('reason') or dyn_detail}"
            )
            return None

        stop_distance = abs(signal.entry - signal.sl)
        if stop_distance <= 0:
            logger.error("Zero stop distance")
            return None

        leverage = self._choose_leverage(signal)

        size = risk_amount / stop_distance
        max_notional = equity * self.risk_gate.cfg.get("max_position_pct", 0.05) * leverage
        if signal.entry > 0 and size * signal.entry > max_notional:
            size = max_notional / signal.entry

        logger.info(
            f"Dynamic risk {signal.symbol}: amount={risk_amount:.4f} "
            f"pct={dyn_detail.get('final_pct')} mult={dyn_detail.get('mult_total')} "
            f"dd={dyn_detail.get('dd_pct')} streak={dyn_detail.get('losses')} "
            f"style={dyn_detail.get('style')}"
        )

        size = self._clamp_amount(signal.symbol, size, price=float(signal.entry or 0))
        if size is None or size <= 0:
            logger.error(f"Size invalid after market limits: {signal.symbol}")
            return None

        # Binance USDT-M: notional tối thiểu 5 USDT (-4164).
        # Sau stepSize/precision, notional có thể < 5 → target cao hơn một chút.
        entry_px = float(signal.entry or 0)
        min_notional = float(
            (self.risk_gate.cfg or {}).get("min_notional_usdt")
            or ((self.risk_gate.cfg or {}).get("dynamic_risk") or {}).get("min_notional_usdt")
            or 5.0
        )
        # buffer 15% để sau floor step vẫn >= 5
        target_notional = min_notional * 1.15
        notional = size * entry_px if entry_px > 0 else 0.0
        if entry_px > 0 and notional < min_notional:
            need_size = target_notional / entry_px
            max_size_pos = (max_notional / entry_px) if max_notional > 0 else need_size
            bumped = min(need_size, max_size_pos)
            # thử clamp nhiều lần nếu floor làm rớt dưới min
            for _try in range(4):
                bumped = self._clamp_amount(signal.symbol, bumped, price=entry_px)
                if not bumped or bumped <= 0:
                    break
                new_notional = bumped * entry_px
                if new_notional >= min_notional:
                    break
                # tăng thêm 10% rồi clamp lại
                bumped = bumped * 1.10
            new_notional = float(bumped or 0) * entry_px
            new_risk = abs(entry_px - float(signal.sl)) * float(bumped or 0)
            max_risk_cap = float(equity) * float(
                ((self.risk_gate.cfg or {}).get("dynamic_risk") or {}).get("max_risk_pct")
                or (self.risk_gate.cfg or {}).get("risk_per_trade_pct", 0.01)
                or 0.01
            ) * 2.0
            if bumped and bumped > 0 and new_notional >= min_notional and new_risk <= max_risk_cap:
                logger.info(
                    f"Size bump for min notional {signal.symbol}: "
                    f"{notional:.4f} → {new_notional:.4f} USDT (size {size} → {bumped})"
                )
                size = bumped
                notional = new_notional
            else:
                logger.warning(
                    f"SKIP {signal.symbol}: notional {notional:.4f} < min {min_notional} USDT "
                    f"(risk_amount={risk_amount:.4f}, bump_notional={new_notional:.4f}, "
                    f"new_risk={new_risk:.4f}, cap={max_risk_cap:.4f}) [-4164]"
                )
                return None

        side = "buy" if signal.side == Side.LONG else "sell"

        try:
            self.exchange.prepare_symbol(signal.symbol, leverage)

            order = None
            last_err = None
            for attempt in range(3):
                try:
                    order = self.exchange.create_order(
                        symbol=signal.symbol,
                        side=side,
                        amount=size,
                        order_type="market",
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    err_s = str(e)
                    if "-4005" in err_s or "max quantity" in err_s.lower():
                        size = self._clamp_amount(
                            signal.symbol, size * 0.5, price=float(signal.entry or 0)
                        )
                        logger.warning(
                            f"Retry -4005 {signal.symbol} attempt={attempt+1} new_size={size}"
                        )
                        if size <= 0:
                            break
                        continue
                    if "-4164" in err_s or "notional" in err_s.lower():
                        entry_px = float(signal.entry or 0)
                        if entry_px <= 0:
                            raise
                        need = (6.0 * (1.0 + 0.15 * attempt)) / entry_px
                        size = self._clamp_amount(
                            signal.symbol, max(float(size or 0), need), price=entry_px
                        )
                        # if clamp still short, push further before next attempt
                        if size and size * entry_px < 5.0:
                            size = self._clamp_amount(
                                signal.symbol, size * 1.2, price=entry_px
                            )
                        logger.warning(
                            f"Retry -4164 {signal.symbol} attempt={attempt+1} "
                            f"new_size={size} notional~={(size or 0)*entry_px:.4f}"
                        )
                        if not size or size <= 0 or size * entry_px < 5.0:
                            break
                        continue
                    raise
            if order is None:
                raise last_err if last_err else RuntimeError("create_order failed")
            order_id = str(order.get("id", "paper"))
            fill_price = float(order.get("price") or order.get("average") or signal.entry)

            # Lock symbol for this cycle immediately after fill
            self._cycle_symbols.add(signal.symbol)

            # Cancel any existing TP/SL on this symbol (avoid 2x/4x stacked orders)
            try:
                n_cancel = self.exchange.cancel_all_orders(signal.symbol)
                if n_cancel:
                    logger.info(
                        f"Canceled previous protective orders on {signal.symbol} "
                        f"before placing new pair (n~={n_cancel})"
                    )
            except Exception as e:
                logger.warning(f"cancel_all_orders before protective failed {signal.symbol}: {e}")

            # Prefer full position size on exchange (entry may have added to existing)
            full_size = self._position_size_on_exchange(signal.symbol, signal.side)
            protect_size = full_size if full_size > 0 else size
            if full_size > 0 and abs(full_size - size) > 1e-12:
                logger.info(
                    f"Protective size uses full position {full_size} "
                    f"(this fill={size}) for {signal.symbol}"
                )
            protect_size = self._clamp_amount(
                signal.symbol, protect_size, price=float(fill_price or signal.entry or 0)
            ) or protect_size

            sl_ok, tp_ok, sl_id, tp_id = self._place_protective_orders(
                signal, protect_size, fill_price
            )

            self.risk_gate.register_order(
                signal, order_id, size,
                sl_order_id=sl_id,
                tp_order_id=tp_id,
            )

            if self.ml_agent is not None:
                try:
                    df = self.ohlcv_cache.get(signal.symbol)
                    if df is None:
                        df = self.ohlcv_cache.get(
                            f"{signal.symbol}_{getattr(signal, 'timeframe', '')}"
                        )
                    actual_risk = abs(fill_price - float(signal.sl)) * size
                    notional = size * fill_price
                    self.ml_agent.on_open(
                        signal,
                        df,
                        key=f"{signal.symbol}_{signal.side.value}",
                        trade_meta={
                            "risk_amount": float(actual_risk or risk_amount),
                            "notional": float(notional),
                            "sl": float(signal.sl),
                            "tp": float(signal.tp),
                            "fill_price": float(fill_price),
                            "size": float(size),
                            "entry": float(fill_price),
                        },
                    )
                except Exception as e:
                    logger.warning(f"ML on_open skip: {e}")

            logger.info(
                f"EXECUTED {signal.side.value} {signal.symbol} size={size:.6f} lev={leverage}x "
                f"entry~={fill_price:.6f} SL={signal.sl:.6f} TP={signal.tp:.6f} "
                f"strategy={signal.strategy} | margin=ISOLATED | TP=LIMIT SL=STOP_MARKET"
            )

            if self.order_logger:
                try:
                    self.order_logger.log(
                        action="OPEN",
                        symbol=signal.symbol,
                        side=signal.side.value,
                        size=size,
                        entry=fill_price,
                        sl=signal.sl,
                        tp=signal.tp,
                        leverage=leverage,
                        strategy=signal.strategy,
                        order_id=order_id,
                        status="FILLED",
                        note=f"lev={leverage}x SL_ok={sl_ok} TP_ok={tp_ok} TP=LIMIT SL=STOP_MARKET",
                    )
                except Exception as e:
                    logger.debug(f"order_logger: {e}")

            self.risk_gate.record_api_call(True)
            return {
                "order_id": order_id,
                "symbol": signal.symbol,
                "side": signal.side.value,
                "size": size,
                "entry": fill_price,
                "leverage": leverage,
                "sl_placed": sl_ok,
                "tp_placed": tp_ok,
                "sl_order_id": sl_id,
                "tp_order_id": tp_id,
            }
        except Exception as e:
            err = str(e).lower()
            logger.error(f"Execution failed {signal.symbol}: {e}")
            soft = any(
                x in err
                for x in (
                    "invalid symbol",
                    "does not have market",
                    "invalid symbol status",
                    "symbol not found",
                    "market is closed",
                    "tradfi",
                    "agreement contract",
                    "sign tradfi",
                    "-1122",
                    "-1121",
                    "-4411",
                    "-2019",
                    "-4164",
                    "-4005",
                    "greater than max quantity",
                    "max quantity",
                )
            )
            if soft:
                self.skip_symbols.add(signal.symbol)
                self.risk_gate.record_api_call(False, count_error=False)
            else:
                self.risk_gate.record_api_call(False, count_error=True)
            return None

    def _market_info(self, symbol: str) -> dict:
        """Resolve ccxt market dict (try common symbol forms)."""
        ex = getattr(self.exchange, "exchange", None)
        markets = getattr(ex, "markets", None) or {}
        candidates = [symbol]
        if symbol.endswith(":USDT"):
            candidates.append(symbol.replace(":USDT", ""))
        if "/USDT:USDT" in symbol:
            base = symbol.split("/")[0]
            candidates.extend([f"{base}/USDT:USDT", f"{base}/USDT", f"{base}USDT"])
        for s in candidates:
            m = markets.get(s)
            if m:
                return m
        if ex is not None:
            try:
                if not markets:
                    ex.load_markets(reload=False)
                    markets = ex.markets or {}
                for s in candidates:
                    m = markets.get(s)
                    if m:
                        return m
                return ex.market(symbol) or {}
            except Exception:
                pass
        return {}

    def _limits_from_filters(self, market: dict) -> dict:
        """Parse Binance LOT_SIZE / MARKET_LOT_SIZE filters (maxQty often missing in limits)."""
        out = {"min": None, "max": None, "step": None}
        info = market.get("info") or {}
        filters = info.get("filters") or []
        for f in filters:
            ftype = str(f.get("filterType") or f.get("filter_type") or "").upper()
            if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                try:
                    if f.get("minQty") is not None:
                        v = float(f["minQty"])
                        if out["min"] is None or v > out["min"]:
                            out["min"] = v
                except (TypeError, ValueError):
                    pass
                try:
                    if f.get("maxQty") is not None:
                        v = float(f["maxQty"])
                        if v > 0 and (out["max"] is None or v < out["max"]):
                            out["max"] = v  # stricter max across filters
                except (TypeError, ValueError):
                    pass
                try:
                    if f.get("stepSize") is not None:
                        out["step"] = float(f["stepSize"])
                except (TypeError, ValueError):
                    pass
        return out

    def _clamp_amount(self, symbol: str, amount: float, price: float = 0.0) -> float:
        """Round + clamp to exchange min/max quantity & max notional (avoid -4005)."""
        import math
        try:
            amount = float(amount or 0)
            if amount <= 0:
                return 0.0

            m = self._market_info(symbol)
            limits = (m.get("limits") or {})
            amt_lim = limits.get("amount") or {}
            cost_lim = limits.get("cost") or {}
            min_amt = float(amt_lim.get("min") or 0) or 0.0
            max_amt = amt_lim.get("max")
            max_cost = cost_lim.get("max")

            filt = self._limits_from_filters(m)
            if filt.get("min") is not None:
                min_amt = max(min_amt, float(filt["min"]))
            if filt.get("max") is not None:
                try:
                    max_amt_f = float(filt["max"])
                    if max_amt is None or float(max_amt or 0) <= 0:
                        max_amt = max_amt_f
                    else:
                        max_amt = min(float(max_amt), max_amt_f)
                except (TypeError, ValueError):
                    max_amt = filt["max"]

            prec = (m.get("precision") or {}).get("amount")
            step = filt.get("step")
            if step is None and prec is not None:
                if isinstance(prec, int):
                    step = 10 ** (-prec) if prec >= 0 else None
                else:
                    try:
                        step = float(prec)
                    except (TypeError, ValueError):
                        step = None

            # Hard safety: never allow insane size if max unknown (alt coin)
            # Cap at 5% of equity * lev / price already done upstream; here absolute qty cap fallback
            if max_amt is not None:
                try:
                    max_amt_f = float(max_amt)
                    if max_amt_f > 0 and amount > max_amt_f:
                        logger.warning(
                            f"Size {amount} > max qty {max_amt_f} for {symbol} -> clamp"
                        )
                        amount = max_amt_f * 0.999  # leave tiny margin under max
                except (TypeError, ValueError):
                    pass

            if price and max_cost is not None:
                try:
                    max_cost_f = float(max_cost)
                    if max_cost_f > 0 and amount * price > max_cost_f:
                        amount = max_cost_f / price
                        logger.warning(f"Clamped by max cost for {symbol} -> {amount}")
                except (TypeError, ValueError):
                    pass

            if step and step > 0:
                amount = math.floor(amount / step) * step
                # avoid float dust above max after floor
                if max_amt is not None:
                    try:
                        max_amt_f = float(max_amt)
                        if max_amt_f > 0 and amount > max_amt_f:
                            amount = math.floor(max_amt_f / step) * step
                    except (TypeError, ValueError):
                        pass

            if min_amt and amount < min_amt:
                logger.warning(f"Size {amount} < min {min_amt} for {symbol}")
                return 0.0

            if prec is not None and isinstance(prec, int) and prec >= 0:
                amount = float(f"{amount:.{prec}f}")
            elif step and step >= 1:
                amount = float(int(amount))

            return amount if amount > 0 else 0.0
        except Exception as e:
            logger.warning(f"_clamp_amount {symbol}: {e}")
            try:
                return float(amount or 0)
            except Exception:
                return 0.0

    def _place_protective_orders(
        self, signal: Signal, size: float, fill_price: float
    ) -> Tuple[bool, bool, str, str]:
        """Place TP LIMIT + SL STOP_MARKET (never LIMIT SL)."""
        close_side = "sell" if signal.side == Side.LONG else "buy"
        entry = float(fill_price or signal.entry or 0)
        sl_px = float(signal.sl)
        tp_px = float(signal.tp)

        # Normalize sides
        if signal.side == Side.LONG:
            if sl_px >= entry:
                logger.error(f"Invalid SL for LONG {signal.symbol}: sl={sl_px} >= entry={entry}")
                return False, False, "", ""
            if tp_px <= entry:
                logger.error(f"Invalid TP for LONG {signal.symbol}: tp={tp_px} <= entry={entry}")
                return False, False, "", ""
        else:
            if sl_px <= entry:
                logger.error(f"Invalid SL for SHORT {signal.symbol}: sl={sl_px} <= entry={entry}")
                return False, False, "", ""
            if tp_px >= entry:
                logger.error(f"Invalid TP for SHORT {signal.symbol}: tp={tp_px} >= entry={entry}")
                return False, False, "", ""

        stop_pct = abs(entry - sl_px) / entry * 100 if entry else 0
        if stop_pct < 0.15:
            logger.warning(
                f"SL too tight ({stop_pct:.3f}% < 0.15%) {signal.symbol} - skip protective"
            )
            return False, False, "", ""

        sl_ok, tp_ok = False, False
        sl_id, tp_id = "", ""

        try:
            tp_order = self.exchange.create_order(
                symbol=signal.symbol,
                side=close_side,
                amount=size,
                price=tp_px,
                order_type="limit",
                params={"reduceOnly": True, "timeInForce": "GTC"},
            )
            tp_id = str(tp_order.get("id", ""))
            tp_ok = True
            logger.info(f"TP LIMIT placed {signal.symbol} @ {tp_px}")
        except Exception as e:
            logger.error(f"TP LIMIT FAILED {signal.symbol}: {e}")
            try:
                tp_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    side=close_side,
                    amount=size,
                    price=tp_px,
                    order_type="limit",
                    params={"reduceOnly": True},
                )
                tp_id = str(tp_order.get("id", ""))
                tp_ok = True
                logger.info(f"TP LIMIT retry OK {signal.symbol} @ {tp_px}")
            except Exception as e2:
                logger.error(f"TP LIMIT retry failed: {e2}")

        try:
            sl_order = self.exchange.create_order(
                symbol=signal.symbol,
                side=close_side,
                amount=size,
                order_type="STOP_MARKET",
                params={
                    "stopPrice": sl_px,
                    "reduceOnly": True,
                    "workingType": "CONTRACT_PRICE",
                },
            )
            sl_id = str(sl_order.get("id", ""))
            sl_ok = True
            logger.info(f"SL STOP_MARKET placed {signal.symbol} @ {sl_px}")
        except Exception as e:
            logger.warning(f"SL STOP_MARKET failed {signal.symbol}: {e} -> retry")
            try:
                sl_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    side=close_side,
                    amount=size,
                    order_type="STOP_MARKET",
                    params={"stopPrice": sl_px, "reduceOnly": True},
                )
                sl_id = str(sl_order.get("id", ""))
                sl_ok = True
                logger.info(f"SL STOP_MARKET retry OK {signal.symbol} @ {sl_px}")
            except Exception as e2:
                logger.error(f"SL STOP_MARKET final failed: {e2}")

        return sl_ok, tp_ok, sl_id, tp_id
