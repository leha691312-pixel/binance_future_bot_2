"""
Supertrend Strategy (TradingView-style)
---------------------------------------
ATR Supertrend flip:
  BUY  when direction crosses under 0 (uptrend starts)
  SELL when direction crosses over  0 (downtrend starts)

SL: Supertrend line (with optional ATR buffer)
TP: structure_tp if available, else risk * rr
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from agents.strategies.base import BaseStrategy
from core.models import Signal, Side

try:
    from agents.structure_tp import structure_tp
except Exception:
    structure_tp = None


class SupertrendStrategy(BaseStrategy):
    name = "Supertrend"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        # Defaults tuned from multi-market backtest
        self.atr_period = int(p.get("atr_period", 14))
        self.factor = float(p.get("factor", 3.0))
        self.rr = float(p.get("rr", 1.5))
        self.min_rr = float(p.get("min_rr", 1.2))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.15))
        self.min_atr_pct = float(p.get("min_atr_pct", 0.001))
        self.enable_long = bool(p.get("enable_long", False))
        self.enable_short = bool(p.get("enable_short", True))
        # Only fire on the bar of flip (not every bar in trend)
        self.only_on_flip = bool(p.get("only_on_flip", True))
        self.min_hold_bars = int(p.get("min_hold_bars", 0))

    @staticmethod
    def _rma(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _supertrend(self, df: pd.DataFrame):
        """TradingView ta.supertrend-like. direction: -1 uptrend, +1 downtrend."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = self._rma(tr, self.atr_period)
        hl2 = (high + low) / 2.0
        up = (hl2 + self.factor * atr).values.copy()
        dn = (hl2 - self.factor * atr).values.copy()
        n = len(df)
        direction = np.ones(n)
        st = np.full(n, np.nan)
        c = close.values
        for i in range(1, n):
            if c[i - 1] > dn[i - 1]:
                dn[i] = max(dn[i], dn[i - 1])
            if c[i - 1] < up[i - 1]:
                up[i] = min(up[i], up[i - 1])
            if direction[i - 1] == -1:
                direction[i] = 1 if c[i] < dn[i] else -1
            else:
                direction[i] = -1 if c[i] > up[i] else 1
            st[i] = dn[i] if direction[i] == -1 else up[i]
        return st, direction, atr

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = self.atr_period + 5
        if len(df) < need:
            return []

        st, direction, atr = self._supertrend(df)
        i = len(df) - 1
        if i < 2 or np.isnan(st[i]) or np.isnan(st[i - 1]):
            return []

        atr_last = float(atr.iloc[i]) if hasattr(atr, "iloc") else float(atr[i])
        close = float(df["close"].iloc[i])
        if atr_last <= 0 or close <= 0:
            return []
        if atr_last / close < self.min_atr_pct:
            return []

        prev_d = direction[i - 1]
        cur_d = direction[i]
        buy = prev_d > 0 and cur_d < 0   # crossunder 0 -> uptrend
        sell = prev_d < 0 and cur_d > 0  # crossover 0 -> downtrend

        if self.only_on_flip and not (buy or sell):
            return []

        signals: List[Signal] = []
        st_line = float(st[i])

        if buy and self.enable_long:
            entry = close
            sl = st_line - atr_last * self.sl_buffer_atr
            if sl >= entry:
                sl = entry - atr_last * max(self.factor * 0.5, 1.0)
            risk = entry - sl
            if risk <= 0:
                return []
            tp = None
            if structure_tp is not None:
                try:
                    tp = structure_tp(
                        df, "LONG", entry, sl,
                        lookback=40, min_rr=self.min_rr, fallback_rr=None,
                    )
                except Exception:
                    tp = None
            if tp is None:
                tp = entry + risk * self.rr
            rr_now = (tp - entry) / risk if risk > 0 else 0
            if rr_now < self.min_rr:
                return []
            conf = min(0.82, 0.58 + min(0.2, (atr_last / close) * 20))
            signals.append(
                Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=float(sl),
                    tp=float(tp),
                    confidence=conf,
                    strategy=self.name,
                    reason=f"Supertrend BUY flip | ST={st_line:.6g} ATR={atr_last:.6g} factor={self.factor}",
                    timeframe=timeframe,
                    extra={
                        "st": st_line,
                        "atr": atr_last,
                        "factor": self.factor,
                        "atr_period": self.atr_period,
                        "direction": float(cur_d),
                    },
                )
            )

        if sell and self.enable_short:
            entry = close
            sl = st_line + atr_last * self.sl_buffer_atr
            if sl <= entry:
                sl = entry + atr_last * max(self.factor * 0.5, 1.0)
            risk = sl - entry
            if risk <= 0:
                return []
            tp = None
            if structure_tp is not None:
                try:
                    tp = structure_tp(
                        df, "SHORT", entry, sl,
                        lookback=40, min_rr=self.min_rr, fallback_rr=None,
                    )
                except Exception:
                    tp = None
            if tp is None:
                tp = entry - risk * self.rr
            rr_now = (entry - tp) / risk if risk > 0 else 0
            if rr_now < self.min_rr:
                return []
            conf = min(0.82, 0.58 + min(0.2, (atr_last / close) * 20))
            signals.append(
                Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=float(sl),
                    tp=float(tp),
                    confidence=conf,
                    strategy=self.name,
                    reason=f"Supertrend SELL flip | ST={st_line:.6g} ATR={atr_last:.6g} factor={self.factor}",
                    timeframe=timeframe,
                    extra={
                        "st": st_line,
                        "atr": atr_last,
                        "factor": self.factor,
                        "atr_period": self.atr_period,
                        "direction": float(cur_d),
                    },
                )
            )

        return signals
