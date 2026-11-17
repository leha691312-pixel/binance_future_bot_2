"""Structure-based TP/SL helpers (support / resistance), not RR multiples.

RR chỉ dùng để *đo* sau khi đã có entry/SL/TP từ cấu trúc giá.
"""
from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


def _ohlc(df: pd.DataFrame) -> pd.DataFrame:
    need = {"high", "low", "close"}
    if not need.issubset(set(c.lower() for c in df.columns)):
        # try as-is
        pass
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}
    for std in ("open", "high", "low", "close"):
        if std not in out.columns and std in cols:
            out.rename(columns={cols[std]: std}, inplace=True)
    return out


def swing_points(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Trả về (swing_highs, swing_lows) dạng (index_pos, price)."""
    df = _ohlc(df)
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    n = len(df)
    sh, sl = [], []
    for i in range(left, n - right):
        w_high = highs[i - left : i + right + 1]
        w_low = lows[i - left : i + right + 1]
        if highs[i] >= np.max(w_high) and highs[i] == np.max(w_high):
            sh.append((i, float(highs[i])))
        if lows[i] <= np.min(w_low) and lows[i] == np.min(w_low):
            sl.append((i, float(lows[i])))
    return sh, sl


def recent_swing_high(
    df: pd.DataFrame,
    lookback: int = 50,
    left: int = 3,
    right: int = 3,
    above: Optional[float] = None,
) -> Optional[float]:
    """Swing high gần nhất trong lookback; nếu above= set thì chỉ lấy mức > above."""
    if df is None or len(df) < left + right + 5:
        return None
    tail = df.iloc[-lookback:] if lookback and len(df) > lookback else df
    sh, _ = swing_points(tail, left=left, right=right)
    if not sh:
        # fallback: max high trong lookback (trừ nến cuối)
        try:
            h = tail["high"].astype(float).iloc[:-1]
            val = float(h.max()) if len(h) else None
        except Exception:
            val = None
        if val is None:
            return None
        if above is not None and val <= above:
            return None
        return val
    # ưu tiên swing gần cuối
    for _, price in reversed(sh):
        if above is None or price > above:
            return float(price)
    return None


def recent_swing_low(
    df: pd.DataFrame,
    lookback: int = 50,
    left: int = 3,
    right: int = 3,
    below: Optional[float] = None,
) -> Optional[float]:
    if df is None or len(df) < left + right + 5:
        return None
    tail = df.iloc[-lookback:] if lookback and len(df) > lookback else df
    _, sl = swing_points(tail, left=left, right=right)
    if not sl:
        try:
            lo = tail["low"].astype(float).iloc[:-1]
            val = float(lo.min()) if len(lo) else None
        except Exception:
            val = None
        if val is None:
            return None
        if below is not None and val >= below:
            return None
        return val
    for _, price in reversed(sl):
        if below is None or price < below:
            return float(price)
    return None


def range_high_low(
    df: pd.DataFrame, lookback: int = 20
) -> Tuple[Optional[float], Optional[float]]:
    if df is None or len(df) < 5:
        return None, None
    tail = df.iloc[-lookback:] if lookback else df
    try:
        return float(tail["high"].max()), float(tail["low"].min())
    except Exception:
        return None, None


def structure_tp(
    df: pd.DataFrame,
    side: str,
    entry: float,
    sl: float,
    *,
    lookback: int = 50,
    left: int = 3,
    right: int = 3,
    min_rr: float = 1.0,
    buffer_pct: float = 0.0005,
    fallback_rr: Optional[float] = None,
) -> Optional[float]:
    """TP theo kháng cự (LONG) / hỗ trợ (SHORT).

    - LONG: swing high gần nhất phía trên entry (hoặc range high)
    - SHORT: swing low gần nhất phía dưới entry (hoặc range low)
    - buffer_pct: lùi nhẹ trong zone (không đụng exact wick)
    - min_rr: nếu structure RR < min_rr → None (bỏ setup) trừ khi fallback_rr
    - fallback_rr: nếu không có structure, TP = entry ± risk * fallback_rr
    """
    if entry <= 0 or sl <= 0 or df is None or len(df) < 10:
        return None

    side_u = str(side).upper()
    is_long = side_u in ("LONG", "BUY")
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    tp: Optional[float] = None
    if is_long:
        sh = recent_swing_high(df, lookback=lookback, left=left, right=right, above=entry)
        rh, _ = range_high_low(df, lookback=min(lookback, 30))
        candidates = [c for c in (sh, rh) if c is not None and c > entry]
        if candidates:
            # TP gần entry nhất nhưng vẫn trên entry (kháng cự gần) — realistic partial;
            # nếu muốn chạy xa hơn có thể lấy max. Dùng mức gần nhất có RR >= min_rr nếu có.
            candidates = sorted(set(candidates))
            chosen = None
            for c in candidates:
                rr = (c - entry) / risk
                if rr >= min_rr:
                    chosen = c
                    break
            if chosen is None:
                chosen = candidates[-1]  # xa nhất trong cửa sổ
            tp = chosen * (1.0 - buffer_pct)
    else:
        slw = recent_swing_low(df, lookback=lookback, left=left, right=right, below=entry)
        _, rl = range_high_low(df, lookback=min(lookback, 30))
        candidates = [c for c in (slw, rl) if c is not None and c < entry]
        if candidates:
            candidates = sorted(set(candidates), reverse=True)
            chosen = None
            for c in candidates:
                rr = (entry - c) / risk
                if rr >= min_rr:
                    chosen = c
                    break
            if chosen is None:
                chosen = candidates[-1]
            tp = chosen * (1.0 + buffer_pct)

    if tp is None and fallback_rr and fallback_rr > 0:
        if is_long:
            tp = entry + risk * fallback_rr
        else:
            tp = entry - risk * fallback_rr

    if tp is None:
        return None

    # TP phải đúng phía
    if is_long and tp <= entry:
        return None
    if not is_long and tp >= entry:
        return None

    rr = abs(tp - entry) / risk
    if rr < min_rr and fallback_rr is None:
        return None
    return float(tp)


def structure_sl(
    df: pd.DataFrame,
    side: str,
    entry: float,
    *,
    lookback: int = 30,
    left: int = 3,
    right: int = 3,
    buffer_pct: float = 0.0008,
    atr: Optional[float] = None,
) -> Optional[float]:
    """SL ngoài swing invalidation (dưới swing low với LONG, trên swing high với SHORT)."""
    if entry <= 0 or df is None or len(df) < 10:
        return None
    is_long = str(side).upper() in ("LONG", "BUY")
    if is_long:
        lvl = recent_swing_low(df, lookback=lookback, left=left, right=right, below=entry)
        if lvl is None:
            return None
        sl = lvl * (1.0 - buffer_pct)
        if atr and atr > 0:
            sl = min(sl, entry - atr * 0.5)
        return float(sl) if sl < entry else None
    else:
        lvl = recent_swing_high(df, lookback=lookback, left=left, right=right, above=entry)
        if lvl is None:
            return None
        sl = lvl * (1.0 + buffer_pct)
        if atr and atr > 0:
            sl = max(sl, entry + atr * 0.5)
        return float(sl) if sl > entry else None


def measure_rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    return abs(tp - entry) / risk
