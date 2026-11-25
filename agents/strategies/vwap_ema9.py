"""
VWAP + EMA9 Pullback Strategy
--------------------------------
Bias: VWAP (trên = ưu tiên LONG, dưới = ưu tiên SHORT)
Trigger: EMA 9 — pullback chạm rồi reclaim / reject

Long:
  1. Close > VWAP (bullish bias)
  2. Giá chạm / gần EMA9 trong vài nến gần
  3. Nến hiện tại reclaim EMA9 (close > EMA9), có rejection nhẹ từ dưới
  4. SL dưới low gần / dưới EMA9 + ATR buffer
  5. TP structure hoặc risk * rr

Short: đối xứng (close < VWAP, reject EMA9 từ trên)
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


class VWAPEMA9Strategy(BaseStrategy):
    name = "VWAPEMA9"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        self.ema_period = int(p.get("ema_period", 9))
        self.vwap_window = int(p.get("vwap_window", 48))  # rolling bars if not daily
        self.use_daily_vwap = bool(p.get("use_daily_vwap", True))
        self.touch_pct = float(p.get("touch_pct", 0.0015))  # 0.15% near EMA9
        self.pullback_lookback = int(p.get("pullback_lookback", 5))
        self.rr = float(p.get("rr", 1.5))
        self.min_rr = float(p.get("min_rr", 1.2))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.35))
        self.sl_swing_lookback = int(p.get("sl_swing_lookback", 8))
        self.min_atr_pct = float(p.get("min_atr_pct", 0.001))  # tránh range chết
        self.require_vwap_side = bool(p.get("require_vwap_side", True))
        self.enable_long = bool(p.get("enable_long", True))
        self.enable_short = bool(p.get("enable_short", True))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.vwap_window, self.ema_period) + self.pullback_lookback + 10
        if len(df) < need:
            return []

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float).replace(0, np.nan)

        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        vwap = self._vwap(df)
        atr = self._atr(df, 14)
        if atr is None or pd.isna(atr.iloc[-1]) or atr.iloc[-1] <= 0:
            return []

        i = len(df) - 1
        c = float(close.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        o = float(open_.iloc[i])
        e = float(ema.iloc[i])
        v = float(vwap.iloc[i])
        a = float(atr.iloc[i])

        if c <= 0 or e <= 0 or v <= 0:
            return []
        if a / c < self.min_atr_pct:
            return []

        # EMA slope (3 bars)
        e_prev = float(ema.iloc[i - 3]) if i >= 3 else e
        ema_up = e > e_prev
        ema_dn = e < e_prev

        touched_ema = self._touched_ema(high, low, ema, i, self.pullback_lookback)
        signals: List[Signal] = []

        # ----- LONG -----
        if self.enable_long:
            bias_ok = (c > v) if self.require_vwap_side else True
            reclaim = c > e and l <= e * (1 + self.touch_pct)
            # pullback từng chạm EMA trong lookback, nến hiện reclaim
            if bias_ok and touched_ema and (reclaim or (c > e and ema_up)):
                # nến không bearish mạnh
                body = c - o
                if body >= -0.15 * a:  # cho phép doji / mild red
                    sl = self._sl_long(low, e, a, i)
                    risk = c - sl
                    if risk > 0 and risk / c >= 0.001:
                        tp = self._tp(df, Side.LONG, c, sl)
                        if tp and tp > c:
                            rr = (tp - c) / risk
                            if rr >= self.min_rr:
                                conf = 0.58
                                if c > v:
                                    conf += 0.08
                                if ema_up:
                                    conf += 0.05
                                if body > 0:
                                    conf += 0.04
                                signals.append(
                                    Signal(
                                        symbol=symbol,
                                        side=Side.LONG,
                                        entry=c,
                                        sl=sl,
                                        tp=float(tp),
                                        confidence=min(0.82, conf),
                                        strategy=self.name,
                                        reason=(
                                            f"VWAP+EMA9 LONG | close>vwap={c>v} | "
                                            f"ema9={e:.6g} vwap={v:.6g} rr={rr:.2f}"
                                        ),
                                        timeframe=timeframe,
                                        extra={
                                            "ema9": e,
                                            "vwap": v,
                                            "setup": "long_ema9_pullback",
                                        },
                                    )
                                )

        # ----- SHORT -----
        if self.enable_short and not signals:
            bias_ok = (c < v) if self.require_vwap_side else True
            reject = c < e and h >= e * (1 - self.touch_pct)
            if bias_ok and touched_ema and (reject or (c < e and ema_dn)):
                body = o - c
                if body >= -0.15 * a:
                    sl = self._sl_short(high, e, a, i)
                    risk = sl - c
                    if risk > 0 and risk / c >= 0.001:
                        tp = self._tp(df, Side.SHORT, c, sl)
                        if tp and tp < c:
                            rr = (c - tp) / risk
                            if rr >= self.min_rr:
                                conf = 0.58
                                if c < v:
                                    conf += 0.08
                                if ema_dn:
                                    conf += 0.05
                                if body > 0:
                                    conf += 0.04
                                signals.append(
                                    Signal(
                                        symbol=symbol,
                                        side=Side.SHORT,
                                        entry=c,
                                        sl=sl,
                                        tp=float(tp),
                                        confidence=min(0.82, conf),
                                        strategy=self.name,
                                        reason=(
                                            f"VWAP+EMA9 SHORT | close<vwap={c<v} | "
                                            f"ema9={e:.6g} vwap={v:.6g} rr={rr:.2f}"
                                        ),
                                        timeframe=timeframe,
                                        extra={
                                            "ema9": e,
                                            "vwap": v,
                                            "setup": "short_ema9_reject",
                                        },
                                    )
                                )

        return signals[:1]

    def _touched_ema(
        self, high: pd.Series, low: pd.Series, ema: pd.Series, i: int, lookback: int
    ) -> bool:
        start = max(0, i - lookback)
        for j in range(start, i + 1):
            e = float(ema.iloc[j])
            if e <= 0:
                continue
            if float(low.iloc[j]) <= e * (1 + self.touch_pct) and float(high.iloc[j]) >= e * (
                1 - self.touch_pct
            ):
                return True
        return False

    def _sl_long(self, low: pd.Series, ema: float, atr: float, i: int) -> float:
        start = max(0, i - self.sl_swing_lookback)
        swing = float(low.iloc[start : i + 1].min())
        sl = min(swing, ema) - atr * self.sl_buffer_atr
        return sl

    def _sl_short(self, high: pd.Series, ema: float, atr: float, i: int) -> float:
        start = max(0, i - self.sl_swing_lookback)
        swing = float(high.iloc[start : i + 1].max())
        sl = max(swing, ema) + atr * self.sl_buffer_atr
        return sl

    def _tp(self, df: pd.DataFrame, side: Side, entry: float, sl: float) -> Optional[float]:
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        side_str = side.value if hasattr(side, "value") else str(side)
        if structure_tp is not None:
            try:
                tp = structure_tp(
                    df,
                    side_str,
                    entry,
                    sl,
                    lookback=40,
                    min_rr=self.min_rr,
                    fallback_rr=None,
                )
                if tp is not None:
                    return float(tp)
            except Exception:
                pass
        if side_str.upper() in ("LONG", "BUY"):
            return entry + risk * self.rr
        return entry - risk * self.rr

    def _vwap(self, df: pd.DataFrame) -> pd.Series:
        tp = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
        vol = df["volume"].astype(float).clip(lower=0)
        if self.use_daily_vwap and "timestamp" in df.columns:
            try:
                ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                day = ts.dt.floor("D")
                cum_pv = (tp * vol).groupby(day).cumsum()
                cum_v = vol.groupby(day).cumsum().replace(0, np.nan)
                return (cum_pv / cum_v).astype(float)
            except Exception:
                pass
        # Rolling VWAP
        w = max(5, self.vwap_window)
        pv = (tp * vol).rolling(w, min_periods=max(3, w // 4)).sum()
        vv = vol.rolling(w, min_periods=max(3, w // 4)).sum().replace(0, np.nan)
        return (pv / vv).astype(float)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()
