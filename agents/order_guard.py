"""
Order guard: hủy SL/TP khi không còn vị thế (sau TP hit / timeout).
"""
from typing import Dict, List, Any, Optional
from loguru import logger
import time


def _symbol_key(symbol: str) -> str:
    if not symbol:
        return ""
    s = str(symbol).upper().replace(" ", "")
    s = s.replace(":USDT", "").replace("/USDT", "USDT").replace("/", "").replace(":", "")
    return s


def has_open_position(exchange, symbol: str) -> bool:
    if not symbol:
        return False
    key = _symbol_key(symbol)
    try:
        for p in exchange.get_positions() or []:
            if abs(float(getattr(p, "size", 0) or 0)) <= 0:
                continue
            if _symbol_key(getattr(p, "symbol", "")) == key:
                return True
    except Exception as e:
        logger.debug(f"has_open_position {symbol}: {e}")
    return False


def _list_symbols_with_open_orders(exchange) -> List[str]:
    symbols: List[str] = []
    seen = set()
    mode = getattr(exchange, "mode", None)
    mode_val = getattr(mode, "value", str(mode or "")).lower()
    if mode_val == "paper":
        return symbols
    try:
        for o in (exchange.exchange.fetch_open_orders() if getattr(exchange, "exchange", None) else []) or []:
            s = o.get("symbol")
            if s and s not in seen:
                seen.add(s)
                symbols.append(s)
    except Exception as e:
        logger.debug(f"list open orders: {e}")
    try:
        if hasattr(exchange, "_fapi"):
            algos = exchange._fapi("openAlgoOrders", "GET", {}) or []
            if isinstance(algos, dict):
                algos = algos.get("orders") or algos.get("data") or []
            for a in algos or []:
                s = a.get("symbol")
                if not s:
                    continue
                if hasattr(exchange, "_algo_id_to_ccxt_symbol"):
                    s = exchange._algo_id_to_ccxt_symbol(s)
                if s and s not in seen:
                    seen.add(s)
                    symbols.append(s)
    except Exception as e:
        logger.debug(f"list algo orders: {e}")
    return symbols


def cancel_sl_tp_if_no_position(exchange, symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Khi KHÔNG còn vị thế trên symbol → hủy hết LIMIT + STOP_MARKET (kể cả conditional/algo).
    Gọi mỗi cycle + sau TP/timeout.
    """
    result: Dict[str, Any] = {"checked": [], "canceled": [], "kept": [], "n_canceled": 0}
    mode = getattr(exchange, "mode", None)
    mode_val = getattr(mode, "value", str(mode or "")).lower()
    if mode_val == "paper":
        return result

    symbols = [symbol] if symbol else _list_symbols_with_open_orders(exchange)
    for sym in symbols:
        if not sym:
            continue
        result["checked"].append(sym)
        if has_open_position(exchange, sym):
            result["kept"].append(sym)
            continue
        logger.warning(f"No position on {sym} but SL/TP remain → cancel (retry)")
        n = 0
        for attempt in range(3):
            try:
                fn = getattr(exchange, "cancel_protective_orders", None) or exchange.cancel_all_orders
                n += int(fn(sym) or 0)
            except Exception as e:
                logger.error(f"cancel attempt {attempt+1} {sym}: {e}")
            time.sleep(0.35)
            # còn lệnh không?
            still = False
            try:
                if exchange.exchange.fetch_open_orders(sym):
                    still = True
            except Exception:
                pass
            try:
                if hasattr(exchange, "_fapi") and hasattr(exchange, "_market_id"):
                    mid = exchange._market_id(sym)
                    algos = exchange._fapi("openAlgoOrders", "GET", {"symbol": mid}) or []
                    if isinstance(algos, dict):
                        algos = algos.get("orders") or algos.get("data") or []
                    if algos:
                        still = True
            except Exception:
                pass
            if not still:
                break
        result["n_canceled"] += n
        result["canceled"].append(sym)
        logger.info(f"Canceled leftover SL/TP for flat {sym} (ops~={n})")

    if result["canceled"]:
        logger.info(
            f"SL/TP cleanup: canceled={len(result['canceled'])} "
            f"kept={len(result['kept'])} checked={len(result['checked'])}"
        )
    return result
