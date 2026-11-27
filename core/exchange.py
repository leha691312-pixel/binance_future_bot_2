"""
Exchange wrapper hỗ trợ 3 mode: paper / testnet / live
Sử dụng ccxt cho tính nhất quán.
"""
import os
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import ccxt
from loguru import logger
from core.models import Mode, Side, Position, OrderStatus


class ExchangeWrapper:
    def __init__(self, mode: str = "paper", config: dict = None):
        self.mode = Mode(mode)
        self.config = config or {}
        self.exchange = None
        self.paper_balance = float(self.config.get("initial_capital", 10000))
        self.paper_positions: Dict[str, Position] = {}
        self.paper_orders: Dict[str, dict] = {}
        self.paper_trades: List[dict] = []
        # isolated | cross — default CROSS
        mm = str(self.config.get("margin_mode", "cross") or "cross").lower()
        self.margin_mode = mm if mm in ("isolated", "cross") else "cross"
        self._init_exchange()
        logger.info(f"Margin mode default: {self.margin_mode.upper()}")

    def _init_exchange(self):
        if self.mode == Mode.PAPER:
            logger.info("Exchange mode: PAPER (local simulation)")
            return

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        if self.mode == Mode.TESTNET:
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_TESTNET_API_KEY", "") or os.getenv("BINANCE_DEMO_API_KEY", ""),
                "secret": os.getenv("BINANCE_TESTNET_API_SECRET", "") or os.getenv("BINANCE_DEMO_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            if hasattr(self.exchange, "enable_demo_trading"):
                self.exchange.enable_demo_trading(True)
            else:
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception as e:
                    logger.error(f"Demo/Sandbox not supported: {e}")
                    raise
            logger.info("Exchange mode: DEMO TRADING (demo.binance.com)")
        else:
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            logger.warning("Exchange mode: LIVE – REAL MONEY")

        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> List[list]:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_ohlcv(symbol, timeframe, limit=limit)
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_ticker(symbol)
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_order_book(symbol, limit)
        return self.exchange.fetch_order_book(symbol, limit)

    def fetch_funding_rate(self, symbol: str) -> dict:
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
                return public.fetch_funding_rate(symbol)
            return self.exchange.fetch_funding_rate(symbol)
        except Exception:
            return {}

    def fetch_open_interest(self, symbol: str) -> dict:
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
                return public.fetch_open_interest(symbol)
            return self.exchange.fetch_open_interest(symbol)
        except Exception:
            return {}

    def _is_tradable_market(self, market: dict, quote: str = "USDT") -> bool:
        if not market or not isinstance(market, dict):
            return False
        if not market.get("active", True):
            return False
        if market.get("spot"):
            return False
        if market.get("expiry") or market.get("expiryDatetime"):
            return False
        if market.get("swap") is False and market.get("future"):
            return False
        info = market.get("info") or {}
        status = str(info.get("status") or info.get("contractStatus") or "").upper()
        if status and status not in ("TRADING", "LIVE"):
            return False
        symbol = market.get("symbol") or ""
        if f"/{quote}" not in symbol and f":{quote}" not in symbol:
            return False
        q = quote.upper()
        quote_ccxt = (market.get("quote") or "").upper()
        settle = (market.get("settle") or "").upper()
        if quote_ccxt and quote_ccxt != q and settle != q:
            return False
        return True

    def _normalize_symbol(self, sym: str, quote: str = "USDT") -> str:
        if not sym:
            return sym
        if f":{quote}" in sym:
            return sym
        if sym.endswith(f"/{quote}"):
            return f"{sym}:{quote}"
        return sym

    def get_top_symbols(self, n: int = 20, quote: str = "USDT") -> List[str]:
        """Top N theo |% biến động 24h|, lọc market TRADING + min quote volume."""
        default_pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
            "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
            "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
            "DOT/USDT:USDT",
        ]
        min_qv = float(self.config.get("min_quote_volume_top", 500_000) or 500_000)

        markets: Dict[str, Any] = {}
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                public.load_markets()
                markets = public.markets or {}
            elif self.exchange:
                if not getattr(self.exchange, "markets", None):
                    self.exchange.load_markets()
                markets = self.exchange.markets or {}
        except Exception as e:
            logger.warning(f"load_markets for top symbols failed: {e}")

        tradable = set()
        for sym, m in (markets or {}).items():
            try:
                if self._is_tradable_market(m, quote=quote):
                    tradable.add(self._normalize_symbol(sym, quote))
            except Exception:
                continue

        tickers = {}
        try:
            if self.mode != Mode.PAPER and self.exchange:
                tickers = self.exchange.fetch_tickers() or {}
            else:
                public = ccxt.binanceusdm({
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                tickers = public.fetch_tickers() or {}
        except Exception as e:
            logger.warning(f"fetch_tickers failed: {e}")
            try:
                if self.exchange:
                    tickers = self.exchange.fetch_tickers() or {}
            except Exception as e2:
                logger.warning(f"exchange fetch_tickers failed: {e2}")

        ranked = []
        for sym, t in tickers.items():
            norm = self._normalize_symbol(sym, quote)
            if tradable and norm not in tradable and sym not in tradable:
                continue
            is_usdt = (
                sym.endswith(f"/{quote}")
                or sym.endswith(f"/{quote}:{quote}")
                or f"/{quote}:" in sym
            )
            if not is_usdt:
                continue
            try:
                qv = float(t.get("quoteVolume") or t.get("baseVolume") or 0 or 0)
            except (TypeError, ValueError):
                qv = 0.0
            if qv < min_qv:
                continue
            # 24h volatility score
            pct = t.get("percentage")
            if pct is None:
                info = t.get("info") or {}
                pct = info.get("priceChangePercent") or info.get("priceChange") or 0
            try:
                pct_f = abs(float(pct or 0))
            except (TypeError, ValueError):
                pct_f = 0.0
            try:
                last = float(t.get("last") or t.get("close") or 0 or 0)
                high = float(t.get("high") or 0 or 0)
                low = float(t.get("low") or 0 or 0)
                if last > 0 and high > low:
                    range_pct = (high - low) / last * 100.0
                    pct_f = max(pct_f, range_pct * 0.5)
            except (TypeError, ValueError):
                pass
            if pct_f <= 0:
                continue
            ranked.append((norm, pct_f, qv))

        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        seen = set()
        result = []
        head = []
        for s, pct_f, _qv in ranked:
            if s in seen:
                continue
            seen.add(s)
            result.append(s)
            if len(head) < 5:
                head.append(f"{s.split('/')[0]}={pct_f:.1f}%")
            if len(result) >= n:
                break

        if len(result) < n and tradable:
            for s in sorted(tradable):
                if s not in seen:
                    result.append(s)
                    seen.add(s)
                if len(result) >= n:
                    break

        if not result:
            logger.warning("Top symbols empty after volatility filter → default majors")
            if tradable:
                result = [s for s in default_pairs if s in tradable][:n]
            if not result:
                result = default_pairs[:n]

        logger.info(
            f"Top symbols by 24h volatility ({self.mode.value}): {len(result)}/{n} "
            f"(tradable_markets={len(tradable)}) head=[{', '.join(head)}]"
        )
        return result[:n]


    def get_balance(self) -> float:
        if self.mode == Mode.PAPER:
            return self.paper_balance
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0))

    def get_positions(self) -> List[Position]:
        if self.mode == Mode.PAPER:
            return list(self.paper_positions.values())
        positions = []
        raw = self.exchange.fetch_positions()
        for p in raw:
            if float(p.get("contracts", 0)) > 0:
                side = Side.LONG if p["side"] == "long" else Side.SHORT
                lev = p.get("leverage") or 1
                try:
                    lev = int(float(lev))
                except (TypeError, ValueError):
                    lev = 1
                ep = p.get("entryPrice") or 0
                upnl = p.get("unrealizedPnl") or 0
                positions.append(Position(
                    symbol=p["symbol"],
                    side=side,
                    entry_price=float(ep) if ep else 0.0,
                    size=float(p.get("contracts") or 0),
                    sl=0.0,
                    tp=0.0,
                    order_id=str(p.get("id") or ""),
                    strategy="unknown",
                    open_time=datetime.utcnow(),
                    leverage=lev,
                    unrealized_pnl=float(upnl) if upnl else 0.0,
                ))
        return positions

    def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        params: dict = None,
    ) -> dict:
        params = params or {}
        if self.mode == Mode.PAPER:
            return self._paper_create_order(symbol, side, amount, price, order_type, params)
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )
            logger.info(f"[{self.mode.value}] Order created: {symbol} {side} {amount}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {e}")
            raise

    def _paper_create_order(self, symbol, side, amount, price, order_type, params):
        order_id = str(uuid.uuid4())[:12]
        ot = (order_type or "market").upper()
        params = params or {}
        if ot in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"):
            stop_price = params.get("stopPrice") or price
            order = {
                "id": order_id,
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": stop_price,
                "stopPrice": stop_price,
                "type": ot,
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": {"paper": True, "reduceOnly": params.get("reduceOnly", True)},
            }
            self.paper_orders[order_id] = order
            logger.info(f"[PAPER] {ot} placed: {symbol} {side} {amount} @ stop={stop_price}")
            return order
        ticker = self.fetch_ticker(symbol)
        fill_price = price or ticker["last"]
        slip = 0.0002
        if side == "buy":
            fill_price *= (1 + slip)
        else:
            fill_price *= (1 - slip)
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": fill_price,
            "type": ot,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
            "info": {"paper": True},
        }
        self.paper_orders[order_id] = order
        logger.info(f"[PAPER] Order filled: {symbol} {side} {amount} @ {fill_price:.6f}")
        return order

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        if self.mode == Mode.PAPER:
            return True
        try:
            self.exchange.set_leverage(int(leverage), symbol)
            logger.info(f"Leverage set: {symbol} → {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"Set leverage failed {symbol}: {e}")
            return False

    def set_margin_mode(self, symbol: str, mode: str = "cross") -> bool:
        if self.mode == Mode.PAPER:
            return True
        mode = (mode or "cross").lower()
        if mode not in ("isolated", "cross"):
            mode = "cross"
        try:
            self.exchange.set_margin_mode(mode, symbol)
            logger.info(f"Margin mode set: {symbol} → {mode.upper()}")
            return True
        except Exception as e:
            err = str(e).lower()
            if "no need to change" in err or "already" in err or "margin type" in err:
                logger.debug(f"Margin mode already {mode} for {symbol}")
                return True
            logger.warning(f"Set margin mode failed {symbol}: {e}")
            return False

    def prepare_symbol(self, symbol: str, leverage: int, margin_mode: str = None) -> bool:
        mode = (margin_mode or getattr(self, "margin_mode", None) or "cross")
        mode = str(mode).lower()
        if mode not in ("isolated", "cross"):
            mode = "cross"
        ok_margin = self.set_margin_mode(symbol, mode)
        ok_lev = self.set_leverage(symbol, leverage)
        return ok_margin and ok_lev

    def _market_id(self, symbol: str) -> str:
        try:
            if self.exchange and getattr(self.exchange, "markets", None):
                m = self.exchange.market(symbol)
                return m.get("id") or symbol.replace("/", "").replace(":USDT", "")
        except Exception:
            pass
        return symbol.replace("/", "").replace(":USDT", "").replace(":USDC", "")

    def _fapi(self, path: str, method: str = "GET", params: dict = None):
        params = params or {}
        if not self.exchange:
            return None
        try:
            return self.exchange.request(path, "fapiPrivate", method, params)
        except Exception as e:
            logger.debug(f"fapi {method} {path}: {e}")
            return None

    def _algo_id_to_ccxt_symbol(self, market_id: str) -> str:
        if not market_id:
            return market_id
        try:
            by_id = getattr(self.exchange, "markets_by_id", None) or {}
            if market_id in by_id:
                return by_id[market_id]["symbol"]
        except Exception:
            pass
        if market_id.endswith("USDT") and "/" not in market_id:
            base = market_id[:-4]
            return f"{base}/USDT:USDT"
        return market_id


    def _norm_sym(self, s: str) -> str:
        if not s:
            return ""
        x = str(s).upper().replace(" ", "")
        for a, b in [(":USDT", ""), ("/USDT", "USDT"), ("/", ""), (":", "")]:
            x = x.replace(a, b)
        return x

    def _cancel_algo_conditional(self, symbol: str) -> int:
        """Hủy STOP/TP conditional (algo) trên Binance USD-M."""
        mid = self._market_id(symbol)
        n = 0
        key = self._norm_sym(symbol)
        mid_key = self._norm_sym(mid)

        algos = []
        for params in ({"symbol": mid}, {}):
            raw = self._fapi("openAlgoOrders", "GET", params)
            if raw is None:
                continue
            if isinstance(raw, dict):
                raw = raw.get("orders") or raw.get("data") or raw.get("list") or []
            if isinstance(raw, list):
                algos.extend(raw)

        seen = set()
        for a in algos:
            try:
                a_sym = str(a.get("symbol") or "")
                a_key = self._norm_sym(a_sym)
                if a_key and a_key != key and a_key != mid_key and key not in a_key and a_key not in key:
                    continue
                algo_id = a.get("algoId") or a.get("algo_id") or a.get("clientAlgoId")
                if algo_id is None or algo_id in seen:
                    continue
                seen.add(algo_id)
                ok = self._fapi("algoOrder", "DELETE", {"algoId": algo_id})
                if ok is None:
                    self._fapi("algoOrder", "DELETE", {"algoId": str(algo_id), "symbol": mid})
                n += 1
                logger.info(
                    f"Canceled algo/conditional {algo_id} on {symbol} "
                    f"({a.get('orderType') or a.get('type')})"
                )
            except Exception as e:
                logger.warning(f"cancel algo item {symbol}: {e}")

        for path in ("algoOpenOrders", "allOpenOrders", "openOrders"):
            try:
                r = self._fapi(path, "DELETE", {"symbol": mid})
                if r is not None:
                    n += 1
                    logger.info(f"Bulk DELETE {path} OK {symbol}")
            except Exception as e:
                logger.debug(f"bulk {path}: {e}")
        return n

    def cancel_all_orders(self, symbol: str = None) -> int:
        """Hủy lệnh thường + STOP/TP conditional. Gọi sau TP hit / timeout / orphan."""
        if self.mode == Mode.PAPER:
            if symbol:
                to_del = [k for k, o in self.paper_orders.items() if o.get("symbol") == symbol]
                for k in to_del:
                    del self.paper_orders[k]
                return len(to_del)
            n = len(self.paper_orders)
            self.paper_orders.clear()
            return n

        cancelled = 0
        symbols: List[str] = [symbol] if symbol else []
        if not symbols:
            try:
                open_orders = self.exchange.fetch_open_orders() or []
                symbols = list({o.get("symbol") for o in open_orders if o.get("symbol")})
            except Exception:
                pass
            try:
                algos = self._fapi("openAlgoOrders", "GET", {}) or []
                if isinstance(algos, dict):
                    algos = algos.get("orders") or algos.get("data") or []
                for a in algos or []:
                    s = a.get("symbol")
                    if not s:
                        continue
                    s = self._algo_id_to_ccxt_symbol(s)
                    if s not in symbols:
                        symbols.append(s)
            except Exception:
                pass
            try:
                for p in self.get_positions():
                    if p.symbol not in symbols:
                        symbols.append(p.symbol)
            except Exception:
                pass

        for sym in symbols:
            if not sym:
                continue
            try:
                self.exchange.cancel_all_orders(sym)
                cancelled += 1
                logger.info(f"cancel_all_orders OK: {sym}")
            except Exception as e:
                logger.debug(f"cancel_all_orders {sym}: {e}")

            try:
                for o in self.exchange.fetch_open_orders(sym) or []:
                    oid = o.get("id")
                    if not oid:
                        continue
                    try:
                        self.exchange.cancel_order(oid, sym)
                        cancelled += 1
                    except Exception as e:
                        logger.debug(f"cancel_order {sym} {oid}: {e}")
            except Exception as e:
                logger.debug(f"fetch_open_orders {sym}: {e}")

            try:
                cancelled += self._cancel_algo_conditional(sym) or 0
            except Exception as e:
                logger.warning(f"_cancel_algo_conditional {sym}: {e}")

            mid = self._market_id(sym)
            try:
                if hasattr(self.exchange, "fapiPrivateDeleteAllOpenOrders"):
                    self.exchange.fapiPrivateDeleteAllOpenOrders({"symbol": mid})
                    cancelled += 1
            except Exception as e:
                logger.debug(f"fapiPrivateDeleteAllOpenOrders {sym}: {e}")

            # Second pass – STOP algo đôi khi còn sau TP vừa khớp
            try:
                time.sleep(0.35)
                cancelled += self._cancel_algo_conditional(sym) or 0
            except Exception:
                pass

        return cancelled

    def cancel_protective_orders(self, symbol: str) -> int:
        """Alias: hủy TP LIMIT + SL STOP (thường + conditional)."""
        return self.cancel_all_orders(symbol)

    def cleanup_orphan_orders(self, extra_symbols: List[str] = None) -> int:
        """Symbol không còn position nhưng còn lệnh → hủy hết (kể cả STOP algo)."""
        if self.mode == Mode.PAPER:
            return 0
        try:
            pos_keys = {
                self._norm_sym(p.symbol)
                for p in self.get_positions()
                if abs(float(p.size or 0)) > 0
            }
        except Exception as e:
            logger.warning(f"cleanup_orphan_orders get_positions: {e}")
            return 0

        order_symbols = set(extra_symbols or [])
        try:
            for o in self.exchange.fetch_open_orders() or []:
                if o.get("symbol"):
                    order_symbols.add(o["symbol"])
        except Exception as e:
            logger.debug(f"cleanup fetch_open_orders: {e}")

        try:
            algos = self._fapi("openAlgoOrders", "GET", {}) or []
            if isinstance(algos, dict):
                algos = algos.get("orders") or algos.get("data") or []
            for a in algos or []:
                s = a.get("symbol")
                if not s:
                    continue
                order_symbols.add(self._algo_id_to_ccxt_symbol(s))
        except Exception as e:
            logger.debug(f"cleanup openAlgoOrders: {e}")

        orphans = [s for s in order_symbols if self._norm_sym(s) not in pos_keys]
        total = 0
        for sym in orphans:
            logger.warning(f"Orphan orders on {sym} (no position) → cancel all (incl. STOP_MARKET)")
            total += self.cancel_all_orders(sym) or 0
        if orphans:
            logger.info(f"Cleaned orphan SL/TP on {len(orphans)} symbol(s), ops~={total}")
        return total

    def close_position(self, symbol: str, side: Side):
        """Timeout/reverse: hủy SL/TP → market reduceOnly → hủy lại SL/TP."""
        try:
            self.cancel_all_orders(symbol)
            logger.info(f"Pre-close cancel TP/SL: {symbol}")
        except Exception as e:
            logger.warning(f"Pre-close cancel failed {symbol}: {e}")

        order = None
        try:
            positions = self.get_positions()
            for p in positions:
                if p.symbol == symbol and p.side == side:
                    close_side = "sell" if side == Side.LONG else "buy"
                    try:
                        order = self.create_order(
                            symbol,
                            close_side,
                            p.size,
                            order_type="market",
                            params={"reduceOnly": True},
                        )
                    except TypeError:
                        order = self.create_order(symbol, close_side, p.size, order_type="market")
                    break
        except Exception as e:
            logger.error(f"close_position market failed {symbol}: {e}")

        time.sleep(0.4)
        try:
            self.cancel_all_orders(symbol)
            logger.info(f"Post-close cancel TP/SL: {symbol}")
        except Exception as e:
            logger.warning(f"Post-close cancel failed {symbol}: {e}")

        try:
            time.sleep(0.6)
            self.cancel_all_orders(symbol)
        except Exception:
            pass
        return order

    def flatten_all(self):
        logger.critical("FLATTEN ALL POSITIONS")
        self.cancel_all_orders()
        for p in self.get_positions():
            self.close_position(p.symbol, p.side)
        try:
            self.cleanup_orphan_orders()
        except Exception:
            pass
        if self.mode == Mode.PAPER:
            self.paper_positions.clear()
