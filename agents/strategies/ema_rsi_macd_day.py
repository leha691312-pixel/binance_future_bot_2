"""
EMA-RSI-MACD Day Trading (Crypto)

Long:
  1) Trend: price > EMA200, EMA21 > EMA50; pullback into EMA21–EMA50 zone
  2) Momentum: RSI near 45–50 then turns up (not deep below 40)
  3) Trigger: MACD cross above signal + histogram flips positive / rising

Short: mirror rules.

SL: beyond recent swing / below EMA50 with buffer
RR: min 1.5 (structure TP preferred when helper available)
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side

try:
    from agents.structure_tp import structure_tp
    HAS_STRUCTURE_TP = True
except Exception:
    HAS_STRUCTURE_TP = False


class EMARSIMACDDayStrategy(BaseStrategy):
    name = "EMARSIMACDDay"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.ema_fast = int(self.params.get("ema_fast", 21))
        self.ema_mid = int(self.params.get("ema_mid", 50))
        self.ema_slow = int(self.params.get("ema_slow", 200))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.macd_fast = int(self.params.get("macd_fast", 12))
        self.macd_slow = int(self.params.get("macd_slow", 26))
        self.macd_signal = int(self.params.get("macd_signal", 9))
        self.rr = float(self.params.get("rr", 1.5))
        self.min_rr = float(self.params.get("min_rr", 1.5))
        self.rsi_long_low = float(self.params.get("rsi_long_low", 40))
        self.rsi_long_high = float(self.params.get("rsi_long_high", 55))
        self.rsi_short_low = float(self.params.get("rsi_short_low", 45))
        self.rsi_short_high = float(self.params.get("rsi_short_high", 60))
        self.pullback_lookback = int(self.params.get("pullback_lookback", 5))
        self.sl_swing_lookback = int(self.params.get("sl_swing_lookback", 10))
        self.sl_buffer_atr = float(self.params.get("sl_buffer_atr", 0.25))
        self.atr_period = int(self.params.get("atr_period", 14))

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.astype(float).ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        c = close.astype(float)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _macd(self, close: pd.Series):
        ef = self._ema(close, self.macd_fast)
        es = self._ema(close, self.macd_slow)
        line = ef - es
        signal = line.ewm(span=self.macd_signal, adjust=False).mean()
        hist = line - signal
        return line, signal, hist

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.ema_slow, self.macd_slow + self.macd_signal, self.rsi_period) + 5
        if df is None or len(df) < need:
            return []

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ema21 = self._ema(close, self.ema_fast)
        ema50 = self._ema(close, self.ema_mid)
        ema200 = self._ema(close, self.ema_slow)
        rsi = self._rsi(close, self.rsi_period)
        macd_line, macd_sig, hist = self._macd(close)
        atr = self._atr(df, self.atr_period)

        i = len(df) - 1
        if any(pd.isna(x.iloc[i]) for x in (ema21, ema50, ema200, rsi, macd_line, macd_sig, hist, atr)):
            return []
        if atr.iloc[i] <= 0:
            return []

        c = float(close.iloc[i])
        e21, e50, e200 = float(ema21.iloc[i]), float(ema50.iloc[i]), float(ema200.iloc[i])
        r0, r1 = float(rsi.iloc[i]), float(rsi.iloc[i - 1])
        m0, s0 = float(macd_line.iloc[i]), float(macd_sig.iloc[i])
        m1, s1 = float(macd_line.iloc[i - 1]), float(macd_sig.iloc[i - 1])
        h0, h1 = float(hist.iloc[i]), float(hist.iloc[i - 1])
        atr_v = float(atr.iloc[i])

        # Pullback: within last N bars, price touched zone between ema21 and ema50
        lb = max(2, self.pullback_lookback)
        window_low = float(low.iloc[i - lb : i + 1].min())
        window_high = float(high.iloc[i - lb : i + 1].max())
        zone_lo = min(e21, e50)
        zone_hi = max(e21, e50)

        touched_zone = window_low <= zone_hi and window_high >= zone_lo

        macd_cross_up = m1 <= s1 and m0 > s0
        macd_cross_dn = m1 >= s1 and m0 < s0
        hist_up = h0 > 0 and h0 >= h1
        hist_dn = h0 < 0 and h0 <= h1

        signals: List[Signal] = []

        # ----- LONG -----
        trend_long = c > e200 and e21 > e50
        rsi_long_ok = (
            r0 >= self.rsi_long_low
            and r0 <= self.rsi_long_high
            and r0 > r1  # turns up
            and r1 >= self.rsi_long_low - 2  # did not crash far below 40
        )
        trigger_long = (macd_cross_up or (m0 > s0 and hist_up and h1 <= 0))

        if trend_long and touched_zone and rsi_long_ok and trigger_long:
            entry = c
            swing_low = float(low.iloc[i - self.sl_swing_lookback : i + 1].min())
            sl_candidates = [
                swing_low - atr_v * self.sl_buffer_atr,
                e50 - atr_v * self.sl_buffer_atr,
            ]
            sl = min(sl_candidates)
            if sl >= entry:
                sl = entry - atr_v * 1.0
            risk = entry - sl
            if risk > 0:
                tp = None
                if HAS_STRUCTURE_TP:
                    tp = structure_tp(
                        df, "LONG", entry, sl,
                        lookback=50, min_rr=self.min_rr,
                        fallback_rr=self.rr,
                    )
                if tp is None:
                    tp = entry + risk * max(self.rr, self.min_rr)
                if abs(tp - entry) / risk >= self.min_rr - 1e-9:
                    conf = 0.62
                    if macd_cross_up:
                        conf += 0.08
                    if 45 <= r0 <= 52:
                        conf += 0.05
                    if e21 > e50 * 1.001:
                        conf += 0.03
                    signals.append(
                        Signal(
                            symbol=symbol,
                            side=Side.LONG,
                            entry=entry,
                            sl=float(sl),
                            tp=float(tp),
                            confidence=min(0.9, conf),
                            strategy=self.name,
                            reason=(
                                f"EMA trend+pullback zone | RSI {r0:.1f}↑ | "
                                f"MACD cross/hist+ | EMA21>50> price>200"
                            ),
                            timeframe=timeframe,
                            extra={
                                "ema21": e21,
                                "ema50": e50,
                                "ema200": e200,
                                "rsi": r0,
                                "macd": m0,
                                "hist": h0,
                                "structure_tp": HAS_STRUCTURE_TP,
                            },
                        )
                    )

        # ----- SHORT -----
        trend_short = c < e200 and e21 < e50
        rsi_short_ok = (
            r0 <= self.rsi_short_high
            and r0 >= self.rsi_short_low
            and r0 < r1
            and r1 <= self.rsi_short_high + 2
        )
        trigger_short = (macd_cross_dn or (m0 < s0 and hist_dn and h1 >= 0))

        if trend_short and touched_zone and rsi_short_ok and trigger_short:
            entry = c
            swing_high = float(high.iloc[i - self.sl_swing_lookback : i + 1].max())
            sl_candidates = [
                swing_high + atr_v * self.sl_buffer_atr,
                e50 + atr_v * self.sl_buffer_atr,
            ]
            sl = max(sl_candidates)
            if sl <= entry:
                sl = entry + atr_v * 1.0
            risk = sl - entry
            if risk > 0:
                tp = None
                if HAS_STRUCTURE_TP:
                    tp = structure_tp(
                        df, "SHORT", entry, sl,
                        lookback=50, min_rr=self.min_rr,
                        fallback_rr=self.rr,
                    )
                if tp is None:
                    tp = entry - risk * max(self.rr, self.min_rr)
                if abs(tp - entry) / risk >= self.min_rr - 1e-9:
                    conf = 0.62
                    if macd_cross_dn:
                        conf += 0.08
                    if 48 <= r0 <= 55:
                        conf += 0.05
                    signals.append(
                        Signal(
                            symbol=symbol,
                            side=Side.SHORT,
                            entry=entry,
                            sl=float(sl),
                            tp=float(tp),
                            confidence=min(0.9, conf),
                            strategy=self.name,
                            reason=(
                                f"EMA trend+retest zone | RSI {r0:.1f}↓ | "
                                f"MACD cross/hist- | EMA21<50 price<200"
                            ),
                            timeframe=timeframe,
                            extra={
                                "ema21": e21,
                                "ema50": e50,
                                "ema200": e200,
                                "rsi": r0,
                                "macd": m0,
                                "hist": h0,
                                "structure_tp": HAS_STRUCTURE_TP,
                            },
                        )
                    )

        return signals
