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
        default_pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
            "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
            "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
            "DOT/USDT:USDT",
        ]
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

        # Rank by 24h volatility (abs % change), not volume.
        # Optional min quote volume filters dead/illiquid coins.
        min_quote_vol = 0.0
        try:
            # soft floor so ranking is not pure dust (USDT)
            min_quote_vol = float(getattr(self, "min_quote_volume_top", 0) or 0)
        except (TypeError, ValueError):
            min_quote_vol = 0.0
        if min_quote_vol <= 0:
            min_quote_vol = 500_000.0  # default 500k USDT 24h quote volume

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

            info = t.get("info") or {}
            # --- 24h volatility score ---
            pct = None
            for key in ("percentage", "changePercent", "priceChangePercent"):
                if t.get(key) is not None:
                    try:
                        pct = float(t.get(key))
                        break
                    except (TypeError, ValueError):
                        pass
            if pct is None and info.get("priceChangePercent") is not None:
                try:
                    pct = float(info.get("priceChangePercent"))
                except (TypeError, ValueError):
                    pct = None

            range_pct = 0.0
            try:
                last = float(t.get("last") or t.get("close") or info.get("lastPrice") or 0)
                high = float(t.get("high") or info.get("highPrice") or 0)
                low = float(t.get("low") or info.get("lowPrice") or 0)
                if last > 0 and high > 0 and low > 0 and high >= low:
                    range_pct = (high - low) / last * 100.0
            except (TypeError, ValueError):
                range_pct = 0.0

            if pct is not None:
                vol_score = abs(float(pct))
                # blend a bit of high-low range so coins with big wick still rank up
                vol_score = max(vol_score, range_pct * 0.5)
            else:
                vol_score = range_pct

            if vol_score <= 0:
                continue

            try:
                qvol = float(t.get("quoteVolume") or info.get("quoteVolume") or 0)
            except (TypeError, ValueError):
                qvol = 0.0
            if qvol < min_quote_vol:
                continue

            ranked.append((norm, vol_score, qvol))

        # sort: higher |24h %| first; tie-break by quote volume
        ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
        seen = set()
        result = []
        top_preview = []
        for s, score, qvol in ranked:
            if s in seen:
                continue
            seen.add(s)
            result.append(s)
            if len(top_preview) < 5:
                top_preview.append(f"{s.split('/')[0]}={score:.1f}%")
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
            f"(tradable={len(tradable)}, min_qvol={min_quote_vol:.0f}) "
            f"head=[{', '.join(top_preview)}]"
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

    def cancel_all_orders(self, symbol: str = None) -> int:
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
        symbols = [symbol] if symbol else []
        if not symbols:
            try:
                open_orders = self.exchange.fetch_open_orders() or []
                symbols = list({o.get("symbol") for o in open_orders if o.get("symbol")})
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
        return cancelled

    def cleanup_orphan_orders(self, extra_symbols: List[str] = None) -> int:
        if self.mode == Mode.PAPER:
            return 0
        try:
            pos_symbols = {p.symbol for p in self.get_positions()}
        except Exception as e:
            logger.warning(f"cleanup get_positions: {e}")
            return 0
        order_symbols = set()
        try:
            for o in self.exchange.fetch_open_orders() or []:
                if o.get("symbol"):
                    order_symbols.add(o["symbol"])
        except Exception as e:
            logger.debug(f"cleanup fetch_open_orders: {e}")
        if extra_symbols:
            order_symbols.update(extra_symbols)
        orphans = [s for s in order_symbols if s not in pos_symbols]
        total = 0
        for sym in orphans:
            logger.warning(f"Orphan orders on {sym} (no position) → cancel all")
            total += self.cancel_all_orders(sym) or 0
        if orphans:
            logger.info(f"Cleaned orphan orders on {len(orphans)} symbol(s)")
        return total

    def close_position(self, symbol: str, side: Side):
        """Đóng vị thế market + reduceOnly (tránh -4164 min notional khi size nhỏ)."""
        try:
            self.cancel_all_orders(symbol)
            logger.info(f"Pre-close cancel protective orders: {symbol}")
        except Exception as e:
            logger.warning(f"Pre-close cancel failed {symbol}: {e}")

        order = None
        # Chuẩn hóa side để so khớp
        side_val = side.value if hasattr(side, "value") else str(side)
        side_u = str(side_val).upper()
        want_long = side_u in ("LONG", "BUY")

        for p in self.get_positions():
            ps = p.side.value if hasattr(p.side, "value") else str(p.side)
            ps_u = str(ps).upper()
            is_long = ps_u in ("LONG", "BUY")
            if p.symbol != symbol:
                continue
            if is_long != want_long:
                continue
            qty = abs(float(p.size or 0))
            if qty <= 0:
                continue
            close_side = "sell" if is_long else "buy"
            try:
                order = self.create_order(
                    symbol,
                    close_side,
                    qty,
                    order_type="market",
                    params={"reduceOnly": True},
                )
                logger.info(f"Close position {symbol} {ps_u} qty={qty} reduceOnly OK")
            except Exception as e:
                err = str(e)
                logger.warning(f"Close reduceOnly failed {symbol}: {e} → retry plain market")
                # Fallback: vẫn cố reduceOnly qua raw params / closePosition
                try:
                    if self.mode != Mode.PAPER and self.exchange:
                        order = self.exchange.create_order(
                            symbol,
                            type="market",
                            side=close_side,
                            amount=qty,
                            params={"reduceOnly": True},
                        )
                        logger.info(f"Close position retry OK {symbol}")
                    else:
                        raise
                except Exception as e2:
                    logger.error(f"Close position failed {symbol}: {e2}")
                    raise
            break

        try:
            self.cancel_all_orders(symbol)
            logger.info(f"Post-close cancel protective orders: {symbol}")
        except Exception as e:
            logger.warning(f"Post-close cancel failed {symbol}: {e}")
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
