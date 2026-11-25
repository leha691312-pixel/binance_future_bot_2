"""
Martin-style EMA Pullback (theo backtest Closest EMA)

Long:
  - 21 EMA: bản chính (winrate/RR tốt nhất trong bảng)
  - 9 EMA: coin mạnh, vào sớm, SL hẹp (~1.2–1.5%)
  - 50 EMA: không trade đơn; chỉ khi 9/21/50 hội tụ gần nhau

Short:
  - 9 EMA: RR/avg gain mạnh
  - 21 EMA: hồi về bán
  - 50 EMA: sau khi phá xuống, lần đầu hồi về 50 EMA bị chặn rồi short lại
  - Hội tụ EMA gần nhau → resistance mạnh hơn

Nguồn ý tưởng: pullback về EMA đang dốc + rejection, rồi tiếp tục trend.
"""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side

try:
    from agents.structure_tp import structure_tp
except Exception:
    structure_tp = None


class MartinEMAPullbackStrategy(BaseStrategy):
    name = "MartinEMAPullback"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        self.ema9 = int(p.get("ema9", 9))
        self.ema21 = int(p.get("ema21", 21))
        self.ema50 = int(p.get("ema50", 50))
        self.touch_tol_pct = float(p.get("touch_tol_pct", 0.15))  # % chạm EMA
        self.converge_pct = float(p.get("converge_pct", 0.8))  # 9/21/50 trong khoảng này
        self.rr_9 = float(p.get("rr_9", 1.5))
        self.rr_21 = float(p.get("rr_21", 2.0))
        self.rr_50 = float(p.get("rr_50", 1.8))
        self.sl_pct_9 = float(p.get("sl_pct_9", 0.013))  # ~1.3% SL hẹp cho 9EMA
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.35))
        self.atr_period = int(p.get("atr_period", 14))
        self.min_atr_pct = float(p.get("min_atr_pct", 0.15))
        self.require_rejection = bool(p.get("require_rejection", True))
        self.enable_long_9 = bool(p.get("enable_long_9", True))
        self.enable_long_21 = bool(p.get("enable_long_21", True))
        self.enable_long_50_converge = bool(p.get("enable_long_50_converge", True))
        self.enable_short_9 = bool(p.get("enable_short_9", True))
        self.enable_short_21 = bool(p.get("enable_short_21", True))
        self.enable_short_50 = bool(p.get("enable_short_50", True))
        self.lookback_break = int(p.get("lookback_break", 15))

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _touch(self, low: float, high: float, ema: float, tol: float) -> bool:
        band = abs(ema) * (tol / 100.0)
        return (low <= ema + band) and (high >= ema - band)

    def _rejection_long(self, row) -> bool:
        body = abs(float(row["close"]) - float(row["open"]))
        rng = float(row["high"]) - float(row["low"])
        if rng <= 0:
            return False
        lower_wick = min(float(row["open"]), float(row["close"])) - float(row["low"])
        return float(row["close"]) >= float(row["open"]) and lower_wick >= max(body * 0.5, rng * 0.25)

    def _rejection_short(self, row) -> bool:
        body = abs(float(row["close"]) - float(row["open"]))
        rng = float(row["high"]) - float(row["low"])
        if rng <= 0:
            return False
        upper_wick = float(row["high"]) - max(float(row["open"]), float(row["close"]))
        return float(row["close"]) <= float(row["open"]) and upper_wick >= max(body * 0.5, rng * 0.25)

    def _ema_cluster(self, e9, e21, e50) -> bool:
        mx = max(e9, e21, e50)
        mn = min(e9, e21, e50)
        mid = (mx + mn) / 2.0 or 1.0
        return (mx - mn) / mid * 100.0 <= self.converge_pct

    def _sl_tp_long(self, df, entry, ema_val, atr, rr, use_tight_pct=False) -> Tuple[Optional[float], Optional[float]]:
        swing = float(df["low"].iloc[-8:].min())
        if use_tight_pct:
            sl = entry * (1.0 - self.sl_pct_9)
        else:
            sl = min(swing, ema_val) - atr * self.sl_buffer_atr
        if sl >= entry:
            sl = entry - atr * max(1.0, self.sl_buffer_atr)
        risk = entry - sl
        if risk <= 0:
            return None, None
        tp = None
        if structure_tp:
            try:
                tp = structure_tp(df, "LONG", float(entry), float(sl), lookback=40, min_rr=1.0, fallback_rr=rr)
            except Exception:
                tp = None
        if tp is None:
            tp = entry + risk * rr
        return float(sl), float(tp)

    def _sl_tp_short(self, df, entry, ema_val, atr, rr, use_tight_pct=False) -> Tuple[Optional[float], Optional[float]]:
        swing = float(df["high"].iloc[-8:].max())
        if use_tight_pct:
            sl = entry * (1.0 + self.sl_pct_9)
        else:
            sl = max(swing, ema_val) + atr * self.sl_buffer_atr
        if sl <= entry:
            sl = entry + atr * max(1.0, self.sl_buffer_atr)
        risk = sl - entry
        if risk <= 0:
            return None, None
        tp = None
        if structure_tp:
            try:
                tp = structure_tp(df, "SHORT", float(entry), float(sl), lookback=40, min_rr=1.0, fallback_rr=rr)
            except Exception:
                tp = None
        if tp is None:
            tp = entry - risk * rr
        return float(sl), float(tp)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.ema50, self.atr_period) + self.lookback_break + 5
        if len(df) < need:
            return []

        df = df.copy()
        df["ema9"] = df["close"].ewm(span=self.ema9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=self.ema21, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=self.ema50, adjust=False).mean()
        df["atr"] = self._atr(df, self.atr_period)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        entry = float(last["close"])
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0 or entry <= 0:
            return []
        if (atr / entry) * 100 < self.min_atr_pct:
            return []

        e9, e21, e50 = float(last["ema9"]), float(last["ema21"]), float(last["ema50"])
        pe9, pe21, pe50 = float(prev["ema9"]), float(prev["ema21"]), float(prev["ema50"])
        uptrend = e21 > e50 and entry > e50
        downtrend = e21 < e50 and entry < e50
        cluster = self._ema_cluster(e9, e21, e50)
        signals: List[Signal] = []

        # ----- LONG 21 EMA (primary) -----
        if self.enable_long_21 and uptrend and e21 > pe21:
            touched = self._touch(float(prev["low"]), float(prev["high"]), pe21, self.touch_tol_pct)
            reclaim = entry > e21 and float(last["close"]) > float(last["open"])
            rej = (not self.require_rejection) or self._rejection_long(prev) or self._rejection_long(last)
            if touched and reclaim and rej:
                sl, tp = self._sl_tp_long(df, entry, e21, atr, self.rr_21)
                if sl and tp:
                    signals.append(Signal(
                        symbol=symbol, side=Side.LONG, entry=entry, sl=sl, tp=tp,
                        confidence=0.72, strategy=self.name, timeframe=timeframe
                    ))

        # ----- LONG 9 EMA (strong coin, early) -----
        if self.enable_long_9 and uptrend and e9 > pe9 and not signals:
            # mạnh: giá bám 9EMA, chưa chạm 21 gần đây
            recent = df.iloc[-6:]
            touched_21 = any(
                self._touch(float(r["low"]), float(r["high"]), float(r["ema21"]), self.touch_tol_pct)
                for _, r in recent.iterrows()
            )
            touched_9 = self._touch(float(prev["low"]), float(prev["high"]), pe9, self.touch_tol_pct)
            hold_9 = entry >= e9 * 0.998 and float(last["close"]) >= float(last["open"])
            if touched_9 and hold_9 and not touched_21:
                sl, tp = self._sl_tp_long(df, entry, e9, atr, self.rr_9, use_tight_pct=True)
                if sl and tp:
                    signals.append(Signal(
                        symbol=symbol, side=Side.LONG, entry=entry, sl=sl, tp=tp,
                        confidence=0.62, strategy=self.name, timeframe=timeframe
                    ))

        # ----- LONG 50 only when EMA converge -----
        if self.enable_long_50_converge and uptrend and cluster and not signals:
            touched = self._touch(float(prev["low"]), float(prev["high"]), pe50, self.touch_tol_pct)
            reclaim = entry > e50
            if touched and reclaim:
                sl, tp = self._sl_tp_long(df, entry, e50, atr, self.rr_50)
                if sl and tp:
                    signals.append(Signal(
                        symbol=symbol, side=Side.LONG, entry=entry, sl=sl, tp=tp,
                        confidence=0.58, strategy=self.name, timeframe=timeframe
                    ))

        # ----- SHORT 50 EMA (breakdown + first retest) -----
        if self.enable_short_50 and downtrend and e50 < pe50:
            window = df.iloc[-(self.lookback_break + 2):-1]
            # từng đóng dưới 50 rồi lần đầu hồi chạm 50
            was_below = (window["close"] < window["ema50"]).sum() >= max(3, self.lookback_break // 3)
            first_retest = was_below and self._touch(float(prev["low"]), float(prev["high"]), pe50, self.touch_tol_pct)
            reject = (not self.require_rejection) or self._rejection_short(prev) or self._rejection_short(last)
            fail = entry < e50
            conf = 0.74 if cluster else 0.68
            if first_retest and reject and fail:
                sl, tp = self._sl_tp_short(df, entry, e50, atr, self.rr_50)
                if sl and tp:
                    signals.append(Signal(
                        symbol=symbol, side=Side.SHORT, entry=entry, sl=sl, tp=tp,
                        confidence=conf, strategy=self.name, timeframe=timeframe
                    ))

        # ----- SHORT 21 EMA -----
        if self.enable_short_21 and downtrend and e21 < pe21 and not any(s.side == Side.SHORT for s in signals):
            touched = self._touch(float(prev["low"]), float(prev["high"]), pe21, self.touch_tol_pct)
            fail = entry < e21 and float(last["close"]) < float(last["open"])
            rej = (not self.require_rejection) or self._rejection_short(prev) or self._rejection_short(last)
            if touched and fail and rej:
                sl, tp = self._sl_tp_short(df, entry, e21, atr, self.rr_21)
                if sl and tp:
                    signals.append(Signal(
                        symbol=symbol, side=Side.SHORT, entry=entry, sl=sl, tp=tp,
                        confidence=0.65, strategy=self.name, timeframe=timeframe
                    ))

        # ----- SHORT 9 EMA (strong RR path) -----
        if self.enable_short_9 and downtrend and e9 < pe9 and not any(s.side == Side.SHORT for s in signals):
            touched = self._touch(float(prev["low"]), float(prev["high"]), pe9, self.touch_tol_pct)
            fail = entry < e9
            rej = (not self.require_rejection) or self._rejection_short(last)
            if touched and fail and rej:
                sl, tp = self._sl_tp_short(df, entry, e9, atr, self.rr_9, use_tight_pct=True)
                if sl and tp:
                    signals.append(Signal(
                        symbol=symbol, side=Side.SHORT, entry=entry, sl=sl, tp=tp,
                        confidence=0.60, strategy=self.name, timeframe=timeframe
                    ))

        return signals[:1]  # tối đa 1 signal / symbol / TF
