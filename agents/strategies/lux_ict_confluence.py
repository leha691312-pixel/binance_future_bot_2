"""
LuxICTConfluence — gộp ý tưởng LuxAlgo ICT toolkits
-------------------------------------------------
1) Market Structure Break (MSB) + momentum filter (~OB Probability Toolkit)
2) Order Block trước MSB
3) Liquidity sweep (equal high/low / pivot) — Liquidity Pools
4) Fair Value Gap retest — FVG LuxAlgo
5) Fibonacci OTE 0.5–0.786 trên impulse — Fib Confluence
6) Volume expansion trên displacement — proxy Volume Delta

LONG setup (SHORT đối xứng):
  - Bias: bullish MSB gần đây (hoặc HH structure)
  - Liquidity: quét SSL (low pivot) gần đây
  - Entry zone: retest bullish FVG và/hoặc bullish OB, ưu tiên trong Fib discount
  - Confirm: nến reclaim + volume >= vol_ma * mult
  - SL dưới zone/swing; TP structure hoặc risk * rr
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from agents.strategies.base import BaseStrategy
from core.models import Signal, Side

try:
    from agents.structure_tp import structure_tp
except Exception:
    structure_tp = None


class LuxICTConfluenceStrategy(BaseStrategy):
    name = "LuxICTConfluence"

    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        self.pivot_len = int(p.get("pivot_len", 3))
        self.msb_lookback = int(p.get("msb_lookback", 40))
        self.msb_body_atr = float(p.get("msb_body_atr", 0.85))  # displacement strength
        self.fvg_max_age = int(p.get("fvg_max_age", 30))
        self.min_gap_pct = float(p.get("min_gap_pct", 0.001))
        self.liq_lookback = int(p.get("liq_lookback", 25))
        self.fib_low = float(p.get("fib_low", 0.5))
        self.fib_high = float(p.get("fib_high", 0.786))
        self.require_fib = bool(p.get("require_fib", False))  # tuned: optional
        self.require_liq_sweep = bool(p.get("require_liq_sweep", True))  # kept
        self.require_fvg = bool(p.get("require_fvg", True))
        self.require_ob = bool(p.get("require_ob", False))
        self.vol_mult = float(p.get("vol_mult", 1.15))
        self.vol_ma = int(p.get("vol_ma", 20))
        self.rr = float(p.get("rr", 1.5))
        self.min_rr = float(p.get("min_rr", 1.2))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.35))
        self.enable_long = bool(p.get("enable_long", False))  # long weak on sample
        self.enable_short = bool(p.get("enable_short", True))
        self.min_score = int(p.get("min_score", 3))  # MSB+FVG/OB+...  # confluence score

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _swings(self, df: pd.DataFrame, left: int = 3) -> Tuple[list, list]:
        highs, lows = [], []
        h, l = df["high"].values, df["low"].values
        n = len(df)
        for i in range(left, n - left):
            if h[i] >= h[i - left : i + left + 1].max():
                highs.append((i, float(h[i])))
            if l[i] <= l[i - left : i + left + 1].min():
                lows.append((i, float(l[i])))
        return highs, lows

    def _find_msb(self, df: pd.DataFrame, atr: float) -> Optional[dict]:
        """Latest MSB in lookback with displacement filter."""
        highs, lows = self._swings(df, self.pivot_len)
        if len(highs) < 2 and len(lows) < 2:
            return None
        close = df["close"].values
        open_ = df["open"].values
        n = len(df)
        start = max(0, n - self.msb_lookback)
        # Bullish MSB: close breaks last confirmed swing high with strong body
        for hi, hv in reversed(highs):
            if hi < start:
                break
            for j in range(hi + 1, n):
                body = abs(close[j] - open_[j])
                if close[j] > hv and body >= atr * self.msb_body_atr:
                    # OB = candle before break (last down/indecision before impulse)
                    ob_i = max(0, j - 1)
                    return {
                        "side": "bull",
                        "break_i": j,
                        "level": hv,
                        "ob_high": float(df["high"].iloc[ob_i]),
                        "ob_low": float(df["low"].iloc[ob_i]),
                        "impulse_low": float(df["low"].iloc[max(0, j - 5) : j + 1].min()),
                        "impulse_high": float(df["high"].iloc[max(0, j - 1) : min(n, j + 3)].max()),
                    }
                if close[j] < hv * 0.98:
                    break
        for lo, lv in reversed(lows):
            if lo < start:
                break
            for j in range(lo + 1, n):
                body = abs(close[j] - open_[j])
                if close[j] < lv and body >= atr * self.msb_body_atr:
                    ob_i = max(0, j - 1)
                    return {
                        "side": "bear",
                        "break_i": j,
                        "level": lv,
                        "ob_high": float(df["high"].iloc[ob_i]),
                        "ob_low": float(df["low"].iloc[ob_i]),
                        "impulse_high": float(df["high"].iloc[max(0, j - 5) : j + 1].max()),
                        "impulse_low": float(df["low"].iloc[max(0, j - 1) : min(n, j + 3)].min()),
                    }
                if close[j] > lv * 1.02:
                    break
        return None

    def _liq_sweep(self, df: pd.DataFrame, side: str) -> bool:
        """SSL sweep for long / BSL sweep for short in recent bars."""
        highs, lows = self._swings(df, self.pivot_len)
        n = len(df)
        start = max(0, n - self.liq_lookback)
        if side == "bull":
            # price wicked below a prior swing low then closed back above
            for lo, lv in reversed(lows):
                if lo < start:
                    break
                for j in range(lo + 1, n):
                    if df["low"].iloc[j] < lv and df["close"].iloc[j] > lv:
                        return True
        else:
            for hi, hv in reversed(highs):
                if hi < start:
                    break
                for j in range(hi + 1, n):
                    if df["high"].iloc[j] > hv and df["close"].iloc[j] < hv:
                        return True
        return False

    def _find_fvg(self, df: pd.DataFrame, side: str) -> Optional[dict]:
        n = len(df)
        for i in range(n - 3, max(n - self.fvg_max_age - 3, 2), -1):
            c0, c2 = df.iloc[i], df.iloc[i - 2]
            if side == "bull" and c0["low"] > c2["high"]:
                gap = (c0["low"] - c2["high"]) / max(c2["high"], 1e-12)
                if gap >= self.min_gap_pct:
                    return {"low": float(c2["high"]), "high": float(c0["low"]), "i": i}
            if side == "bear" and c0["high"] < c2["low"]:
                gap = (c2["low"] - c0["high"]) / max(c2["low"], 1e-12)
                if gap >= self.min_gap_pct:
                    return {"low": float(c0["high"]), "high": float(c2["low"]), "i": i}
        return None

    def _in_fib_ote(self, price: float, impulse_low: float, impulse_high: float, side: str) -> bool:
        if impulse_high <= impulse_low:
            return False
        rng = impulse_high - impulse_low
        if side == "bull":
            # discount from high: retrace down into 0.5-0.786
            lo = impulse_high - rng * self.fib_high
            hi = impulse_high - rng * self.fib_low
            return lo <= price <= hi
        else:
            lo = impulse_low + rng * self.fib_low
            hi = impulse_low + rng * self.fib_high
            return lo <= price <= hi

    def _vol_ok(self, df: pd.DataFrame) -> bool:
        if "volume" not in df.columns:
            return True
        v = df["volume"].astype(float)
        ma = v.rolling(self.vol_ma, min_periods=5).mean()
        if pd.isna(ma.iloc[-1]) or ma.iloc[-1] <= 0:
            return True
        return float(v.iloc[-1]) >= float(ma.iloc[-1]) * self.vol_mult * 0.85  # mild on entry bar

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.msb_lookback, self.fvg_max_age, self.liq_lookback) + 15
        if len(df) < need:
            return []

        atr_s = self._atr(df, 14)
        atr = float(atr_s.iloc[-1])
        if pd.isna(atr) or atr <= 0:
            return []

        msb = self._find_msb(df, atr)
        if msb is None:
            return []

        last = df.iloc[-1]
        price = float(last["close"])
        signals: List[Signal] = []

        def build(side_name: str):
            nonlocal signals
            score = 1  # MSB itself
            fvg = self._find_fvg(df, "bull" if side_name == "LONG" else "bear")
            liq = self._liq_sweep(df, "bull" if side_name == "LONG" else "bear")
            if liq:
                score += 1
            elif self.require_liq_sweep:
                return

            in_fvg = False
            zone_lo = zone_hi = None
            if fvg:
                zone_lo, zone_hi = fvg["low"], fvg["high"]
                in_fvg = zone_lo <= price <= zone_hi or (
                    side_name == "LONG"
                    and last["low"] <= zone_hi
                    and price >= zone_lo
                ) or (
                    side_name == "SHORT"
                    and last["high"] >= zone_lo
                    and price <= zone_hi
                )
                if in_fvg:
                    score += 1
            elif self.require_fvg:
                return

            # OB zone
            in_ob = msb["ob_low"] <= price <= msb["ob_high"] or (
                side_name == "LONG"
                and last["low"] <= msb["ob_high"]
                and price >= msb["ob_low"]
            ) or (
                side_name == "SHORT"
                and last["high"] >= msb["ob_low"]
                and price <= msb["ob_high"]
            )
            if in_ob:
                score += 1
            elif self.require_ob:
                return

            # Fib OTE on impulse
            if side_name == "LONG":
                fib_ok = self._in_fib_ote(
                    price, msb.get("impulse_low", msb["ob_low"]), msb.get("impulse_high", msb["level"]), "bull"
                )
            else:
                fib_ok = self._in_fib_ote(
                    price, msb.get("impulse_low", msb["level"]), msb.get("impulse_high", msb["ob_high"]), "bear"
                )
            if fib_ok:
                score += 1
            elif self.require_fib:
                return

            if self._vol_ok(df):
                score += 1

            if score < self.min_score:
                return
            # need price interacting with FVG or OB
            if not in_fvg and not in_ob:
                return

            if side_name == "LONG":
                if not self.enable_long:
                    return
                # rejection-ish: close back up
                if float(last["close"]) < float(last["open"]):
                    return
                sl_base = min(
                    msb["ob_low"],
                    zone_lo if zone_lo is not None else msb["ob_low"],
                    float(df["low"].iloc[-8:].min()),
                )
                sl = sl_base - atr * self.sl_buffer_atr
                entry = price
                risk = entry - sl
                if risk <= 0:
                    return
                tp = None
                if structure_tp:
                    try:
                        tp = structure_tp(df, "LONG", entry, sl, lookback=40, min_rr=self.min_rr, fallback_rr=None)
                    except Exception:
                        tp = None
                if tp is None:
                    tp = entry + risk * self.rr
                if (tp - entry) / risk < self.min_rr:
                    return
                conf = min(0.88, 0.52 + 0.06 * score)
                signals.append(
                    Signal(
                        symbol=symbol,
                        side=Side.LONG,
                        entry=entry,
                        sl=float(sl),
                        tp=float(tp),
                        confidence=conf,
                        strategy=self.name,
                        reason=(
                            f"LuxICT LONG score={score} MSB+{'FVG' if in_fvg else ''}"
                            f"{'+OB' if in_ob else ''}+{'Fib' if fib_ok else ''}+{'Liq' if liq else ''}"
                        ),
                        timeframe=timeframe,
                        extra={"score": score, "msb_level": msb["level"], "in_fvg": in_fvg, "in_ob": in_ob},
                    )
                )
            else:
                if not self.enable_short:
                    return
                if float(last["close"]) > float(last["open"]):
                    return
                sl_base = max(
                    msb["ob_high"],
                    zone_hi if zone_hi is not None else msb["ob_high"],
                    float(df["high"].iloc[-8:].max()),
                )
                sl = sl_base + atr * self.sl_buffer_atr
                entry = price
                risk = sl - entry
                if risk <= 0:
                    return
                tp = None
                if structure_tp:
                    try:
                        tp = structure_tp(df, "SHORT", entry, sl, lookback=40, min_rr=self.min_rr, fallback_rr=None)
                    except Exception:
                        tp = None
                if tp is None:
                    tp = entry - risk * self.rr
                if (entry - tp) / risk < self.min_rr:
                    return
                conf = min(0.88, 0.52 + 0.06 * score)
                signals.append(
                    Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry=entry,
                        sl=float(sl),
                        tp=float(tp),
                        confidence=conf,
                        strategy=self.name,
                        reason=(
                            f"LuxICT SHORT score={score} MSB+{'FVG' if in_fvg else ''}"
                            f"{'+OB' if in_ob else ''}+{'Fib' if fib_ok else ''}+{'Liq' if liq else ''}"
                        ),
                        timeframe=timeframe,
                        extra={"score": score, "msb_level": msb["level"], "in_fvg": in_fvg, "in_ob": in_ob},
                    )
                )

        if msb["side"] == "bull":
            build("LONG")
        elif msb["side"] == "bear":
            build("SHORT")

        return signals
