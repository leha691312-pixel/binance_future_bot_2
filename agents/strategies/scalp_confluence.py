"""
Scalp Confluence = MARibbon + StochBB + PivotScalp + EMARSIScalp
Score từng lớp, vào khi score >= min_score. Phù hợp 5m / 15m.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class ScalpConfluenceStrategy(BaseStrategy):
    name = "ScalpConfluence"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        # MA Ribbon 5-8-13
        self.p1 = int(p.get("ribbon_p1", 5))
        self.p2 = int(p.get("ribbon_p2", 8))
        self.p3 = int(p.get("ribbon_p3", 13))
        self.min_spread_pct = float(p.get("min_spread_pct", 0.0003))
        # Stoch + BB
        self.stoch_k = int(p.get("stoch_k", 14))
        self.stoch_d = int(p.get("stoch_d", 3))
        self.stoch_os = float(p.get("stoch_os", 20))
        self.stoch_ob = float(p.get("stoch_ob", 80))
        self.bb_period = int(p.get("bb_period", 20))
        self.bb_std = float(p.get("bb_std", 2.0))
        # Pivot
        self.pivot_lookback = int(p.get("pivot_lookback", 5))
        self.pivot_tol = float(p.get("pivot_tol", 0.0015))
        # EMA RSI scalp
        self.ema_f = int(p.get("ema_fast", 8))
        self.ema_s = int(p.get("ema_slow", 21))
        self.rsi_period = int(p.get("rsi_period", 7))
        # Confluence / risk
        self.min_score = int(p.get("min_score", 3))
        self.require_ribbon = bool(p.get("require_ribbon", False))
        self.require_stoch = bool(p.get("require_stoch", False))
        self.rr = float(p.get("rr", 1.3))
        self.atr_period = int(p.get("atr_period", 14))
        self.atr_sl_mult = float(p.get("atr_sl_mult", 1.35))
        self.enable_long = bool(p.get("enable_long", False))
        self.enable_short = bool(p.get("enable_short", True))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.p3, self.bb_period, self.stoch_k, self.ema_s, self.pivot_lookback * 3) + 10
        if len(df) < need:
            return []

        d = df.copy()
        c = d["close"].astype(float)
        h = d["high"].astype(float)
        l = d["low"].astype(float)

        d["s1"] = c.rolling(self.p1).mean()
        d["s2"] = c.rolling(self.p2).mean()
        d["s3"] = c.rolling(self.p3).mean()
        d["ema_f"] = c.ewm(span=self.ema_f, adjust=False).mean()
        d["ema_s"] = c.ewm(span=self.ema_s, adjust=False).mean()
        d["rsi"] = self._rsi(c, self.rsi_period)
        mid = c.rolling(self.bb_period).mean()
        std = c.rolling(self.bb_period).std()
        d["bb_up"] = mid + self.bb_std * std
        d["bb_lo"] = mid - self.bb_std * std
        # Stochastic
        low_n = l.rolling(self.stoch_k).min()
        high_n = h.rolling(self.stoch_k).max()
        d["stoch_k"] = 100 * (c - low_n) / (high_n - low_n).replace(0, np.nan)
        d["stoch_d"] = d["stoch_k"].rolling(self.stoch_d).mean()
        d["atr"] = self._atr(d, self.atr_period)

        last = d.iloc[-1]
        prev = d.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0 or pd.isna(last["s3"]) or pd.isna(last["stoch_k"]):
            return []

        price = float(last["close"])

        # 1) MA Ribbon
        bull_rib = last["s1"] > last["s2"] > last["s3"]
        bear_rib = last["s1"] < last["s2"] < last["s3"]
        spread = abs(float(last["s1"] - last["s3"])) / price
        rib_ok = spread >= self.min_spread_pct
        rib_long = rib_ok and bull_rib
        rib_short = rib_ok and bear_rib

        # 2) StochBB
        sk, sd = float(last["stoch_k"]), float(last["stoch_d"])
        sk_p = float(prev["stoch_k"])
        stoch_long = sk < self.stoch_os + 15 and sk >= sk_p and price <= float(last["bb_lo"]) * 1.002
        stoch_short = sk > self.stoch_ob - 15 and sk <= sk_p and price >= float(last["bb_up"]) * 0.998
        # softer stoch zone
        if not stoch_long and sk < 35 and sk > sk_p and price < float(mid.iloc[-1]):
            stoch_long = True
        if not stoch_short and sk > 65 and sk < sk_p and price > float(mid.iloc[-1]):
            stoch_short = True

        # 3) Pivot (local swing)
        lb = self.pivot_lookback
        piv_long = piv_short = False
        if len(d) > lb * 2 + 2:
            # support: recent swing low
            sl_win = l.iloc[-(lb * 2 + 1) : -1]
            sh_win = h.iloc[-(lb * 2 + 1) : -1]
            swing_lo = float(sl_win.min())
            swing_hi = float(sh_win.max())
            if abs(price - swing_lo) / price <= self.pivot_tol * 3 and last["close"] >= last["open"]:
                piv_long = True
            if abs(price - swing_hi) / price <= self.pivot_tol * 3 and last["close"] <= last["open"]:
                piv_short = True
            # classic floor pivot from prev day-ish window
            ph, pl, pc = float(h.iloc[-lb-1]), float(l.iloc[-lb-1]), float(c.iloc[-lb-1])
            pp = (ph + pl + pc) / 3
            s1 = 2 * pp - ph
            r1 = 2 * pp - pl
            if abs(price - s1) / price <= self.pivot_tol * 2 and price > s1:
                piv_long = True
            if abs(price - r1) / price <= self.pivot_tol * 2 and price < r1:
                piv_short = True

        # 4) EMA RSI scalp
        ema_long = float(last["ema_f"]) > float(last["ema_s"]) and float(last["rsi"]) > 48
        ema_short = float(last["ema_f"]) < float(last["ema_s"]) and float(last["rsi"]) < 52
        cross_up = float(prev["ema_f"]) <= float(prev["ema_s"]) and float(last["ema_f"]) > float(last["ema_s"])
        cross_dn = float(prev["ema_f"]) >= float(prev["ema_s"]) and float(last["ema_f"]) < float(last["ema_s"])
        if cross_up and float(last["rsi"]) > 50:
            ema_long = True
        if cross_dn and float(last["rsi"]) < 50:
            ema_short = True

        signals: List[Signal] = []

        def build(side: str, score: int, parts: list) -> Optional[Signal]:
            entry = price
            if side == "LONG":
                sl = entry - atr * self.atr_sl_mult
                if rib_long:
                    sl = min(sl, float(last["s3"]))
                risk = entry - sl
                if risk <= 0:
                    return None
                tp = entry + risk * self.rr
                se = Side.LONG
            else:
                sl = entry + atr * self.atr_sl_mult
                if rib_short:
                    sl = max(sl, float(last["s3"]))
                risk = sl - entry
                if risk <= 0:
                    return None
                tp = entry - risk * self.rr
                se = Side.SHORT
            conf = min(0.55 + 0.09 * score, 0.9)
            return Signal(
                symbol=symbol, side=se, entry=float(entry), sl=float(sl), tp=float(tp),
                confidence=float(conf), strategy=self.name,
                reason=f"ScalpConfluence {score}/4 | " + "+".join(parts),
                timeframe=timeframe, extra={"score": score, "layers": parts},
            )

        if self.enable_long:
            parts = []
            if rib_long:
                parts.append("Ribbon")
            if stoch_long:
                parts.append("StochBB")
            if piv_long:
                parts.append("Pivot")
            if ema_long:
                parts.append("EMARSI")
            score = len(parts)
            ok = score >= self.min_score
            if self.require_ribbon and not rib_long:
                ok = False
            if self.require_stoch and not stoch_long:
                ok = False
            if ok:
                s = build("LONG", score, parts)
                if s:
                    signals.append(s)

        if self.enable_short:
            parts = []
            if rib_short:
                parts.append("Ribbon")
            if stoch_short:
                parts.append("StochBB")
            if piv_short:
                parts.append("Pivot")
            if ema_short:
                parts.append("EMARSI")
            score = len(parts)
            ok = score >= self.min_score
            if self.require_ribbon and not rib_short:
                ok = False
            if self.require_stoch and not stoch_short:
                ok = False
            if ok:
                s = build("SHORT", score, parts)
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
