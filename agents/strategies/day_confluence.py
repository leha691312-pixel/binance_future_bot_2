"""
Day Confluence = TrendMA + MomentumRSI + PullbackEMA + RangeBounce

Hai chế độ:
- TREND (ADX cao): TrendMA + PullbackEMA + MomentumRSI
- RANGE (ADX thấp): RangeBounce + MomentumRSI (RSI cực trị)

Score theo lớp active trong từng chế độ.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class DayConfluenceStrategy(BaseStrategy):
    name = "DayConfluence"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        self.ema_fast = int(p.get("ema_fast", 9))
        self.ema_slow = int(p.get("ema_slow", 21))
        self.ema_pull = int(p.get("ema_pull", 21))
        self.adx_period = int(p.get("adx_period", 14))
        self.adx_trend_min = float(p.get("adx_trend_min", 22))
        self.adx_range_max = float(p.get("adx_range_max", 20))
        self.rsi_period = int(p.get("rsi_period", 14))
        self.rsi_long_min = float(p.get("rsi_long_min", 45))
        self.rsi_long_max = float(p.get("rsi_long_max", 65))
        self.rsi_short_min = float(p.get("rsi_short_min", 35))
        self.rsi_short_max = float(p.get("rsi_short_max", 55))
        self.rsi_range_low = float(p.get("rsi_range_low", 35))
        self.rsi_range_high = float(p.get("rsi_range_high", 65))
        self.range_lookback = int(p.get("range_lookback", 30))
        self.zone_pct = float(p.get("zone_pct", 0.15))
        self.min_range_pct = float(p.get("min_range_pct", 0.008))
        self.min_score = int(p.get("min_score", 2))
        self.allow_trend = bool(p.get("allow_trend", True))
        self.allow_range = bool(p.get("allow_range", False))
        self.rr = float(p.get("rr", 1.4))
        self.atr_period = int(p.get("atr_period", 14))
        self.atr_sl_mult = float(p.get("atr_sl_mult", 1.6))
        self.enable_long = bool(p.get("enable_long", True))
        self.enable_short = bool(p.get("enable_short", True))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.ema_slow, self.range_lookback, self.adx_period) + 15
        if len(df) < need:
            return []

        d = df.copy()
        c = d["close"].astype(float)
        h = d["high"].astype(float)
        l = d["low"].astype(float)

        d["ema_f"] = c.ewm(span=self.ema_fast, adjust=False).mean()
        d["ema_s"] = c.ewm(span=self.ema_slow, adjust=False).mean()
        d["ema_p"] = c.ewm(span=self.ema_pull, adjust=False).mean()
        d["rsi"] = self._rsi(c, self.rsi_period)
        d["atr"] = self._atr(d, self.atr_period)
        d["adx"] = self._adx(d, self.adx_period)

        last = d.iloc[-1]
        prev = d.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        adx = float(last["adx"]) if not pd.isna(last["adx"]) else 0.0
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        if atr <= 0 or pd.isna(last["ema_f"]):
            return []

        price = float(last["close"])
        ema_f, ema_s, ema_p = float(last["ema_f"]), float(last["ema_s"]), float(last["ema_p"])

        is_trend = adx >= self.adx_trend_min
        is_range = adx <= self.adx_range_max

        # --- TrendMA ---
        bull = ema_f > ema_s
        bear = ema_f < ema_s
        cross_up = float(prev["ema_f"]) <= float(prev["ema_s"]) and ema_f > ema_s
        cross_dn = float(prev["ema_f"]) >= float(prev["ema_s"]) and ema_f < ema_s
        trend_long = is_trend and bull and (cross_up or (price > ema_f and last["close"] > last["open"]))
        trend_short = is_trend and bear and (cross_dn or (price < ema_f and last["close"] < last["open"]))

        # --- PullbackEMA ---
        pb_long = bull and float(prev["low"]) <= float(prev["ema_p"]) * 1.001 and price > ema_p and last["close"] > last["open"]
        pb_short = bear and float(prev["high"]) >= float(prev["ema_p"]) * 0.999 and price < ema_p and last["close"] < last["open"]

        # --- MomentumRSI ---
        mom_long = self.rsi_long_min <= rsi <= self.rsi_long_max and rsi >= float(prev["rsi"])
        mom_short = self.rsi_short_min <= rsi <= self.rsi_short_max and rsi <= float(prev["rsi"])

        # --- RangeBounce ---
        rng_long = rng_short = False
        if is_range and self.allow_range:
            window = d.iloc[-self.range_lookback : -1]
            res = float(window["high"].max())
            sup = float(window["low"].min())
            height = res - sup
            if sup > 0 and height / sup >= self.min_range_pct:
                zone = height * self.zone_pct
                if price <= sup + zone and rsi <= self.rsi_range_low and last["close"] > last["open"]:
                    rng_long = True
                if price >= res - zone and rsi >= self.rsi_range_high and last["close"] < last["open"]:
                    rng_short = True

        signals: List[Signal] = []

        def build(side: str, score: int, parts: list, mode: str) -> Optional[Signal]:
            entry = price
            if side == "LONG":
                sl = entry - atr * self.atr_sl_mult
                if pb_long or trend_long:
                    sl = min(sl, ema_p - atr * 0.2)
                risk = entry - sl
                if risk <= 0:
                    return None
                tp = entry + risk * self.rr
                se = Side.LONG
            else:
                sl = entry + atr * self.atr_sl_mult
                if pb_short or trend_short:
                    sl = max(sl, ema_p + atr * 0.2)
                risk = sl - entry
                if risk <= 0:
                    return None
                tp = entry - risk * self.rr
                se = Side.SHORT
            conf = min(0.55 + 0.1 * score, 0.9)
            return Signal(
                symbol=symbol, side=se, entry=float(entry), sl=float(sl), tp=float(tp),
                confidence=float(conf), strategy=self.name,
                reason=f"DayConfluence {mode} {score} | " + "+".join(parts),
                timeframe=timeframe, extra={"score": score, "mode": mode, "adx": adx},
            )

        # TREND mode scores
        if self.allow_trend and is_trend:
            if self.enable_long:
                parts = []
                if trend_long:
                    parts.append("TrendMA")
                if pb_long:
                    parts.append("PullbackEMA")
                if mom_long:
                    parts.append("MomRSI")
                if len(parts) >= self.min_score:
                    s = build("LONG", len(parts), parts, "TREND")
                    if s:
                        signals.append(s)
            if self.enable_short:
                parts = []
                if trend_short:
                    parts.append("TrendMA")
                if pb_short:
                    parts.append("PullbackEMA")
                if mom_short:
                    parts.append("MomRSI")
                if len(parts) >= self.min_score:
                    s = build("SHORT", len(parts), parts, "TREND")
                    if s:
                        signals.append(s)

        # RANGE mode scores
        if self.allow_range and is_range:
            if self.enable_long and rng_long:
                parts = ["RangeBounce"]
                if mom_long or rsi <= self.rsi_range_low:
                    parts.append("MomRSI")
                if len(parts) >= min(self.min_score, 2):
                    s = build("LONG", len(parts), parts, "RANGE")
                    if s:
                        signals.append(s)
            if self.enable_short and rng_short:
                parts = ["RangeBounce"]
                if mom_short or rsi >= self.rsi_range_high:
                    parts.append("MomRSI")
                if len(parts) >= min(self.min_score, 2):
                    s = build("SHORT", len(parts), parts, "RANGE")
                    if s:
                        signals.append(s)

        if len(signals) > 1:
            signals = [max(signals, key=lambda x: x.confidence)]
        return signals

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean().replace(0, np.nan)
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        return dx.rolling(period, min_periods=period).mean().fillna(0.0)
