"""ATR helpers — chuẩn hóa chu kỳ & kẹp SL theo ATR%."""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(int(period)).mean()


def clamp_stop_distance(
    entry: float,
    sl: float,
    atr: float,
    *,
    min_stop_atr: float = 0.8,
    max_stop_atr: float = 2.8,
    side: str = "LONG",
) -> float:
    """Giữ |entry-sl| trong [min_stop_atr, max_stop_atr] × ATR."""
    if entry <= 0 or atr <= 0:
        return sl
    dist = abs(entry - sl)
    lo, hi = min_stop_atr * atr, max_stop_atr * atr
    if dist < lo:
        dist = lo
    elif dist > hi:
        dist = hi
    is_long = str(side).upper() in ("LONG", "BUY")
    return (entry - dist) if is_long else (entry + dist)


def atr_pct(entry: float, atr: float) -> float:
    if entry <= 0:
        return 0.0
    return float(atr) / float(entry)
