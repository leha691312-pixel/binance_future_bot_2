"""
TrendFibVWAP — confluence TrendMA + FibPullback + VWAPEMA9
==========================================================
Tối ưu mặc định cho 15m / 1h (không phải 5m).

Ba lớp (mỗi lớp 1 điểm, cần min_score):
1) TrendMA   : EMA21/50 + giá vs EMA200
2) FibPullback: vùng OTE 0.5–0.786 từ swing
3) VWAPEMA9  : bias VWAP + EMA9 touch/reclaim

Tuning 15m/1h:
- swing_lookback rộng hơn (60)
- vwap_window 96 (~1 ngày trên 15m)
- min_score 2, prefer short hoặc score 3
- RR 1.4, sl_buffer_atr 0.55
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

try:
    from agents.strategies.base import BaseStrategy as _Base
    from core.models import Signal as _Signal, Side as _Side
    BaseStrategy = _Base
    Signal = _Signal
    Side = _Side
except Exception:
    class Side(str, Enum):
        LONG = "LONG"
        SHORT = "SHORT"

    @dataclass
    class Signal:
        symbol: str
        side: object
        entry: float
        sl: float
        tp: float
        confidence: float
        strategy: str
        reason: str
        timeframe: str
        timestamp: datetime = field(default_factory=datetime.utcnow)
        extra: dict = field(default_factory=dict)

        def risk_reward(self) -> float:
            risk = abs(self.entry - self.sl)
            if risk == 0:
                return 0.0
            return abs(self.tp - self.entry) / risk

    class BaseStrategy:
        name = "base"
        def __init__(self, params=None):
            self.params = params or {}
        def _ensure_columns(self, df):
            df = df.copy()
            for c in ("open", "high", "low", "close", "volume"):
                if c not in df.columns:
                    raise ValueError(c)
            return df


class TrendFibVWAPStrategy(BaseStrategy):
    name = "TrendFibVWAP"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        # TrendMA
        self.ema_fast = int(p.get("ema_fast", 21))
        self.ema_slow = int(p.get("ema_slow", 50))
        self.ema_trend = int(p.get("ema_trend", 200))
        # VWAP + EMA9
        self.ema9 = int(p.get("ema9", 9))
        self.vwap_window = int(p.get("vwap_window", 96))  # ~1d on 15m
        self.touch_pct = float(p.get("touch_pct", 0.0025))
        self.pullback_lookback = int(p.get("pullback_lookback", 6))
        # Fib — wider for 15m/1h
        self.swing_lookback = int(p.get("swing_lookback", 60))
        self.fib_lo = float(p.get("fib_lo", 0.5))
        self.fib_hi = float(p.get("fib_hi", 0.786))
        # Confluence
        self.min_score = int(p.get("min_score", 2))
        self.require_trend = bool(p.get("require_trend", True))  # 15m/1h: siết trend
        self.require_vwap = bool(p.get("require_vwap", False))
        # Risk tuned 15m/1h
        self.rr = float(p.get("rr", 1.5))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.55))
        self.atr_period = int(p.get("atr_period", 14))
        self.min_atr_pct = float(p.get("min_atr_pct", 0.0015))
        self.enable_long = bool(p.get("enable_long", False))
        self.enable_short = bool(p.get("enable_short", True))
        # Volume soft filter
        self.vol_filter = bool(p.get("vol_filter", True))
        self.vol_ma_period = int(p.get("vol_ma_period", 20))
        self.vol_min_ratio = float(p.get("vol_min_ratio", 0.9))
        # RSI soft
        self.rsi_filter = bool(p.get("rsi_filter", False))
        self.rsi_period = int(p.get("rsi_period", 14))
        self.rsi_long_min = float(p.get("rsi_long_min", 35))
        self.rsi_long_max = float(p.get("rsi_long_max", 58))
        self.rsi_short_min = float(p.get("rsi_short_min", 42))
        self.rsi_short_max = float(p.get("rsi_short_max", 68))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.ema_trend, self.vwap_window, self.swing_lookback) + 15
        if len(df) < need:
            return []

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float).fillna(0)

        ema9 = close.ewm(span=self.ema9, adjust=False).mean()
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
        ema_t = close.ewm(span=self.ema_trend, adjust=False).mean()
        vwap = self._vwap(df)
        atr = self._atr(df, self.atr_period)
        rsi = self._rsi(close, self.rsi_period)
        vol_ma = volume.rolling(self.vol_ma_period, min_periods=5).mean()

        i = len(df) - 1
        c = float(close.iloc[i])
        a = float(atr.iloc[i]) if atr is not None and not pd.isna(atr.iloc[i]) else 0.0
        if c <= 0 or a <= 0 or a / c < self.min_atr_pct:
            return []

        if self.vol_filter:
            vol = float(volume.iloc[i])
            vma = float(vol_ma.iloc[i]) if not pd.isna(vol_ma.iloc[i]) else 0.0
            if vma > 0 and vol < self.vol_min_ratio * vma:
                return []

        e9 = float(ema9.iloc[i])
        ef = float(ema_f.iloc[i])
        es = float(ema_s.iloc[i])
        et = float(ema_t.iloc[i])
        v = float(vwap.iloc[i])
        r = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else 50.0
        r_prev = float(rsi.iloc[i - 1]) if i >= 1 and not pd.isna(rsi.iloc[i - 1]) else r

        touched = self._touched(high, low, ema9, i, self.pullback_lookback)

        # ---- LONG score ----
        score_l, tags_l = 0, []
        if c > et and ef > es:
            score_l += 1
            tags_l.append("TrendMA")
        fib_l = self._fib_zone_long(high, low, i)
        if fib_l and fib_l[0] <= c <= fib_l[1]:
            score_l += 1
            tags_l.append("Fib")
        reclaim = c > e9 and float(low.iloc[i]) <= e9 * (1 + self.touch_pct)
        if c > v and touched and (reclaim or c > e9):
            score_l += 1
            tags_l.append("VWAPEMA9")

        # ---- SHORT score ----
        score_s, tags_s = 0, []
        if c < et and ef < es:
            score_s += 1
            tags_s.append("TrendMA")
        fib_s = self._fib_zone_short(high, low, i)
        if fib_s and fib_s[0] <= c <= fib_s[1]:
            score_s += 1
            tags_s.append("Fib")
        reject = c < e9 and float(high.iloc[i]) >= e9 * (1 - self.touch_pct)
        if c < v and touched and (reject or c < e9):
            score_s += 1
            tags_s.append("VWAPEMA9")

        signals: List[Signal] = []

        if self.enable_long and score_l >= self.min_score:
            if self.require_trend and "TrendMA" not in tags_l:
                pass
            elif self.require_vwap and "VWAPEMA9" not in tags_l:
                pass
            elif self.rsi_filter and not (
                self.rsi_long_min <= r <= self.rsi_long_max and r >= r_prev - 1.5
            ):
                pass
            else:
                s = self._build(symbol, timeframe, Side.LONG, c, a, low, i, score_l, tags_l)
                if s:
                    signals.append(s)

        if self.enable_short and score_s >= self.min_score:
            if self.require_trend and "TrendMA" not in tags_s:
                pass
            elif self.require_vwap and "VWAPEMA9" not in tags_s:
                pass
            elif self.rsi_filter and not (
                self.rsi_short_min <= r <= self.rsi_short_max and r <= r_prev + 1.5
            ):
                pass
            else:
                s = self._build(symbol, timeframe, Side.SHORT, c, a, high, i, score_s, tags_s)
                if s:
                    signals.append(s)

        return signals

    def _build(self, symbol, timeframe, side, entry, atr, swing_series, i, score, tags):
        look = min(16, i)
        if side == Side.LONG or str(side).endswith("LONG"):
            swing = float(swing_series.iloc[i - look : i + 1].min())
            sl = min(swing, entry - atr * self.sl_buffer_atr)
            if sl >= entry:
                sl = entry - atr * max(self.sl_buffer_atr, 0.8)
            risk = entry - sl
            tp = entry + risk * self.rr
        else:
            swing = float(swing_series.iloc[i - look : i + 1].max())
            sl = max(swing, entry + atr * self.sl_buffer_atr)
            if sl <= entry:
                sl = entry + atr * max(self.sl_buffer_atr, 0.8)
            risk = sl - entry
            tp = entry - risk * self.rr
        if risk <= 0:
            return None
        conf = min(0.85, max(0.5, 0.55 + 0.12 * (score - 2)))
        return Signal(
            symbol=symbol,
            side=side,
            entry=float(entry),
            sl=float(sl),
            tp=float(tp),
            confidence=conf,
            strategy=self.name,
            reason=f"score={score}/3 [{'+'.join(tags)}]",
            timeframe=timeframe,
            extra={"score": score, "tags": tags},
        )

    def _fib_zone_long(self, high, low, i) -> Optional[Tuple[float, float]]:
        start = max(0, i - self.swing_lookback)
        window_h = high.iloc[start:i]
        window_l = low.iloc[start:i]
        if len(window_h) < 12:
            return None
        sw_low = float(window_l.min())
        sw_high = float(window_h.max())
        rng = sw_high - sw_low
        if rng <= 0:
            return None
        z_hi = sw_high - rng * self.fib_lo
        z_lo = sw_high - rng * self.fib_hi
        return (min(z_lo, z_hi), max(z_lo, z_hi))

    def _fib_zone_short(self, high, low, i) -> Optional[Tuple[float, float]]:
        start = max(0, i - self.swing_lookback)
        window_h = high.iloc[start:i]
        window_l = low.iloc[start:i]
        if len(window_h) < 12:
            return None
        sw_low = float(window_l.min())
        sw_high = float(window_h.max())
        rng = sw_high - sw_low
        if rng <= 0:
            return None
        z_lo = sw_low + rng * self.fib_lo
        z_hi = sw_low + rng * self.fib_hi
        return (min(z_lo, z_hi), max(z_lo, z_hi))

    def _touched(self, high, low, ema, i, lookback) -> bool:
        for j in range(max(0, i - lookback), i + 1):
            e = float(ema.iloc[j])
            if float(low.iloc[j]) <= e * (1 + self.touch_pct) and float(high.iloc[j]) >= e * (1 - self.touch_pct):
                return True
        return False

    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        vol = df["volume"].astype(float).fillna(0).clip(lower=0)
        w = self.vwap_window if self.vwap_window > 0 else 96
        num = (tp * vol).rolling(w, min_periods=max(8, w // 5)).sum()
        den = vol.rolling(w, min_periods=max(8, w // 5)).sum().replace(0, np.nan)
        return num / den

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        prev = c.shift(1)
        tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        d = close.diff()
        gain = d.clip(lower=0).rolling(period).mean()
        loss = (-d.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
