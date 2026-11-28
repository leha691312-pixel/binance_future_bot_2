"""
Swing Confluence = OI_Funding + SwingMACross + SwingRSI + FibPullback + BBSqueeze

Mỗi lớp cộng 1 điểm (OI/Funding optional khi có data).
Vào lệnh khi score >= min_score.
Thiên swing: TF 1h / 4h.
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


class SwingConfluenceStrategy(BaseStrategy):
    name = "SwingConfluence"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        # MA cross
        self.ema_fast = int(p.get("ema_fast", 20))
        self.ema_slow = int(p.get("ema_slow", 50))
        # RSI
        self.rsi_period = int(p.get("rsi_period", 14))
        self.rsi_long_lo = float(p.get("rsi_long_lo", 40))
        self.rsi_long_hi = float(p.get("rsi_long_hi", 55))
        self.rsi_short_lo = float(p.get("rsi_short_lo", 45))
        self.rsi_short_hi = float(p.get("rsi_short_hi", 60))
        # Fib
        self.swing_lookback = int(p.get("swing_lookback", 60))
        self.fib_lo = float(p.get("fib_lo", 0.382))
        self.fib_hi = float(p.get("fib_hi", 0.618))
        self.fib_tol = float(p.get("fib_tol", 0.006))
        # BB squeeze
        self.bb_period = int(p.get("bb_period", 20))
        self.bb_std = float(p.get("bb_std", 2.0))
        self.squeeze_lookback = int(p.get("squeeze_lookback", 40))
        self.squeeze_pctile = float(p.get("squeeze_pctile", 0.25))
        self.vol_mult = float(p.get("vol_mult", 1.15))
        # OI / funding (optional)
        self.oi_lookback = int(p.get("oi_lookback", 12))
        self.min_oi_change = float(p.get("min_oi_change", 0.03))
        self.funding_extreme = float(p.get("funding_extreme", 0.0008))
        self.use_vol_as_oi_proxy = bool(p.get("use_vol_as_oi_proxy", True))
        # Confluence / risk
        self.min_score = int(p.get("min_score", 2))
        self.require_ma = bool(p.get("require_ma", True))
        self.require_fib = bool(p.get("require_fib", False))
        self.require_bb = bool(p.get("require_bb", False))
        self.rr = float(p.get("rr", 1.6))
        self.min_rr = float(p.get("min_rr", 1.2))
        self.atr_period = int(p.get("atr_period", 14))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.5))
        self.atr_sl_mult = float(p.get("atr_sl_mult", 2.0))
        self.enable_long = bool(p.get("enable_long", False))
        self.enable_short = bool(p.get("enable_short", True))

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        oi_series: Optional[pd.Series] = None,
        funding_rate: Optional[float] = None,
    ) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.swing_lookback, self.ema_slow, self.bb_period, self.squeeze_lookback) + 15
        if len(df) < need:
            return []

        d = df.copy()
        c = d["close"].astype(float)
        h = d["high"].astype(float)
        l = d["low"].astype(float)
        v = d["volume"].astype(float)

        d["ema_f"] = c.ewm(span=self.ema_fast, adjust=False).mean()
        d["ema_s"] = c.ewm(span=self.ema_slow, adjust=False).mean()
        d["rsi"] = self._rsi(d, self.rsi_period)
        d["atr"] = self._atr(d, self.atr_period)
        mid = c.rolling(self.bb_period).mean()
        std = c.rolling(self.bb_period).std()
        d["bb_up"] = mid + self.bb_std * std
        d["bb_lo"] = mid - self.bb_std * std
        d["bw"] = (d["bb_up"] - d["bb_lo"]) / mid.replace(0, np.nan)
        d["vol_ma"] = v.rolling(20).mean()

        last = d.iloc[-1]
        prev = d.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0 or pd.isna(last["ema_f"]) or pd.isna(last["rsi"]):
            return []

        price = float(last["close"])
        ema_f = float(last["ema_f"])
        ema_s = float(last["ema_s"])
        rsi = float(last["rsi"])

        # --- SwingMACross ---
        ma_long = ema_f > ema_s and price > ema_s
        ma_short = ema_f < ema_s and price < ema_s
        # cross boost
        cross_up = float(prev["ema_f"]) <= float(prev["ema_s"]) and ema_f > ema_s
        cross_dn = float(prev["ema_f"]) >= float(prev["ema_s"]) and ema_f < ema_s

        # --- SwingRSI ---
        rsi_long = self.rsi_long_lo <= rsi <= self.rsi_long_hi and rsi >= float(prev["rsi"])
        rsi_short = self.rsi_short_lo <= rsi <= self.rsi_short_hi and rsi <= float(prev["rsi"])

        # --- FibPullback ---
        window = d.iloc[-self.swing_lookback : -1]
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        rng = swing_high - swing_low
        fib_long = fib_short = False
        if rng > 0 and rng / max(swing_low, 1e-12) >= 0.012:
            hi_i = window["high"].idxmax()
            lo_i = window["low"].idxmin()
            up_impulse = hi_i > lo_i
            if up_impulse:
                for lvl in (self.fib_lo, 0.5, self.fib_hi):
                    fib = swing_high - rng * float(lvl)
                    if abs(price - fib) / price <= self.fib_tol and last["close"] >= last["open"]:
                        fib_long = True
                        break
            else:
                for lvl in (self.fib_lo, 0.5, self.fib_hi):
                    fib = swing_low + rng * float(lvl)
                    if abs(price - fib) / price <= self.fib_tol and last["close"] <= last["open"]:
                        fib_short = True
                        break

        # --- BBSqueeze (squeeze recently + expansion / break) ---
        hist_bw = d["bw"].iloc[-self.squeeze_lookback : -1].dropna()
        bw = float(last["bw"]) if not pd.isna(last["bw"]) else 1.0
        bb_long = bb_short = False
        if len(hist_bw) >= 10:
            thr = float(hist_bw.quantile(self.squeeze_pctile))
            was_sq = bw <= thr or float(d["bw"].iloc[-4:-1].min()) <= thr
            vol_ok = True
            if not pd.isna(last["vol_ma"]) and last["vol_ma"] > 0:
                vol_ok = float(last["volume"]) >= float(last["vol_ma"]) * self.vol_mult
            if was_sq and vol_ok:
                if price > float(last["bb_up"]) and last["close"] > last["open"]:
                    bb_long = True
                if price < float(last["bb_lo"]) and last["close"] < last["open"]:
                    bb_short = True
            # soft: squeeze + price near mid with trend
            elif was_sq:
                if ma_long and price > mid.iloc[-1]:
                    bb_long = True
                if ma_short and price < mid.iloc[-1]:
                    bb_short = True

        # --- OI_Funding (optional) or volume proxy ---
        oi_long = oi_short = False
        if funding_rate is not None:
            pc = (price - float(d["close"].iloc[-self.oi_lookback])) / float(d["close"].iloc[-self.oi_lookback])
            if funding_rate > self.funding_extreme and pc > 0.01:
                oi_short = True
            if funding_rate < -self.funding_extreme and pc < -0.01:
                oi_long = True
        if oi_series is not None and len(oi_series) >= self.oi_lookback:
            try:
                oi_now = float(oi_series.iloc[-1])
                oi_prev = float(oi_series.iloc[-self.oi_lookback])
                oi_ch = (oi_now - oi_prev) / oi_prev if oi_prev > 0 else 0.0
                pc = (price - float(d["close"].iloc[-self.oi_lookback])) / float(d["close"].iloc[-self.oi_lookback])
                if oi_ch > self.min_oi_change and pc > 0.008:
                    oi_long = True
                if oi_ch > self.min_oi_change and pc < -0.008:
                    oi_short = True
            except Exception:
                pass
        if self.use_vol_as_oi_proxy and not oi_long and not oi_short:
            # volume expansion + price direction as weak OI proxy
            if not pd.isna(last["vol_ma"]) and last["vol_ma"] > 0:
                if float(last["volume"]) >= float(last["vol_ma"]) * 1.4:
                    pc = (price - float(d["close"].iloc[-5])) / float(d["close"].iloc[-5])
                    if pc > 0.005 and ma_long:
                        oi_long = True
                    if pc < -0.005 and ma_short:
                        oi_short = True

        signals: List[Signal] = []

        def build(side: str, score: int, parts: List[str]) -> Optional[Signal]:
            entry = price
            if side == "LONG":
                sl = min(float(l.iloc[-8:].min()), entry - atr * self.atr_sl_mult)
                sl = sl - atr * self.sl_buffer_atr
                risk = entry - sl
                if risk <= 0:
                    return None
                tp = None
                if structure_tp:
                    try:
                        tp = structure_tp(d, "LONG", entry, sl, lookback=40, min_rr=self.min_rr, fallback_rr=None)
                    except Exception:
                        tp = None
                if tp is None:
                    tp = entry + risk * self.rr
                if tp <= entry:
                    return None
                side_e = Side.LONG
            else:
                sl = max(float(h.iloc[-8:].max()), entry + atr * self.atr_sl_mult)
                sl = sl + atr * self.sl_buffer_atr
                risk = sl - entry
                if risk <= 0:
                    return None
                tp = None
                if structure_tp:
                    try:
                        tp = structure_tp(d, "SHORT", entry, sl, lookback=40, min_rr=self.min_rr, fallback_rr=None)
                    except Exception:
                        tp = None
                if tp is None:
                    tp = entry - risk * self.rr
                if tp >= entry:
                    return None
                side_e = Side.SHORT

            conf = min(0.55 + 0.08 * score + (0.05 if cross_up or cross_dn else 0), 0.92)
            return Signal(
                symbol=symbol,
                side=side_e,
                entry=float(entry),
                sl=float(sl),
                tp=float(tp),
                confidence=float(conf),
                strategy=self.name,
                reason=f"SwingConfluence score={score}/5 | " + " + ".join(parts),
                timeframe=timeframe,
                extra={"score": score, "layers": parts},
            )

        # LONG score
        if self.enable_long:
            parts = []
            if ma_long:
                parts.append("MA_cross" + ("_fresh" if cross_up else ""))
            if rsi_long:
                parts.append("RSI")
            if fib_long:
                parts.append("Fib")
            if bb_long:
                parts.append("BB_sq")
            if oi_long:
                parts.append("OI/Vol")
            score = len(parts)
            ok = score >= self.min_score
            if self.require_ma and not ma_long:
                ok = False
            if self.require_fib and not fib_long:
                ok = False
            if self.require_bb and not bb_long:
                ok = False
            if ok:
                sig = build("LONG", score, parts)
                if sig:
                    signals.append(sig)

        # SHORT score
        if self.enable_short:
            parts = []
            if ma_short:
                parts.append("MA_cross" + ("_fresh" if cross_dn else ""))
            if rsi_short:
                parts.append("RSI")
            if fib_short:
                parts.append("Fib")
            if bb_short:
                parts.append("BB_sq")
            if oi_short:
                parts.append("OI/Vol")
            score = len(parts)
            ok = score >= self.min_score
            if self.require_ma and not ma_short:
                ok = False
            if self.require_fib and not fib_short:
                ok = False
            if self.require_bb and not bb_short:
                ok = False
            if ok:
                sig = build("SHORT", score, parts)
                if sig:
                    signals.append(sig)

        # một signal / side tốt nhất
        if len(signals) > 1:
            signals = [max(signals, key=lambda s: s.confidence)]
        return signals

    def _rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        delta = df["close"].astype(float).diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
