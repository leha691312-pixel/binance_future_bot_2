"""LiqSweep + PremiumDiscount + Supertrend confluence"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side
try:
    from agents.structure_tp import structure_tp
except Exception:
    structure_tp = None

class LiqSweepPDSupertrendStrategy(BaseStrategy):
    name = "LiqSweepPDSupertrend"
    def __init__(self, params: dict = None):
        super().__init__(params)
        p = self.params
        self.atr_period = int(p.get("atr_period", 14))
        self.st_factor = float(p.get("st_factor", 3.0))
        self.sweep_lookback = int(p.get("sweep_lookback", 30))
        self.equal_tol = float(p.get("equal_tol", 0.0015))
        self.pd_lookback = int(p.get("pd_lookback", 50))
        self.discount_max = float(p.get("discount_max", 0.40))
        self.premium_min = float(p.get("premium_min", 0.60))
        self.min_score = int(p.get("min_score", 2))
        self.require_st = bool(p.get("require_st", True))
        self.require_sweep = bool(p.get("require_sweep", False))
        self.require_pd = bool(p.get("require_pd", True))
        self.rr = float(p.get("rr", 1.5))
        self.min_rr = float(p.get("min_rr", 1.2))
        self.sl_buffer_atr = float(p.get("sl_buffer_atr", 0.35))
        self.min_atr_pct = float(p.get("min_atr_pct", 0.001))
        self.enable_long = bool(p.get("enable_long", False))
        self.enable_short = bool(p.get("enable_short", True))
        self.only_st_flip = bool(p.get("only_st_flip", False))
    @staticmethod
    def _rma(s, period):
        return s.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    def _atr(self, df):
        h,l,c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return self._rma(tr, self.atr_period)
    def _supertrend(self, df):
        high,low,close = df["high"].astype(float),df["low"].astype(float),df["close"].astype(float)
        atr=self._atr(df); hl2=(high+low)/2.0
        up=(hl2+self.st_factor*atr).values.copy(); dn=(hl2-self.st_factor*atr).values.copy()
        n=len(df); direction=np.ones(n); st=np.full(n,np.nan); c=close.values
        for i in range(1,n):
            if c[i-1]>dn[i-1]: dn[i]=max(dn[i],dn[i-1])
            if c[i-1]<up[i-1]: up[i]=min(up[i],up[i-1])
            if c[i]>up[i-1]: direction[i]=-1
            elif c[i]<dn[i-1]: direction[i]=1
            else: direction[i]=direction[i-1]
            st[i]=dn[i] if direction[i]<0 else up[i]
        return pd.Series(st,index=df.index), pd.Series(direction,index=df.index)
    def _sweep(self, df):
        w=df.iloc[-(self.sweep_lookback+1):-1]; last=df.iloc[-1]
        if len(w)<10: return False, False
        max_high=float(w["high"].max()); min_low=float(w["low"].min())
        bear=last["high"]>max_high*(1+self.equal_tol*0.3) and last["close"]<max_high
        bull=last["low"]<min_low*(1-self.equal_tol*0.3) and last["close"]>min_low
        return bool(bull), bool(bear)
    def _pd_zone(self, df, price):
        w=df.iloc[-self.pd_lookback:]; hi=float(w["high"].max()); lo=float(w["low"].min())
        if hi<=lo: return 0.5, False, False
        pos=(price-lo)/(hi-lo)
        return pos, pos<=self.discount_max, pos>=self.premium_min
    def generate(self, df, symbol, timeframe="15m"):
        need=max(self.atr_period,self.sweep_lookback,self.pd_lookback,60)+5
        if df is None or len(df)<need: return []
        df=df.copy()
        for c in ("open","high","low","close"):
            if c not in df.columns: return []
            df[c]=df[c].astype(float)
        atr=self._atr(df); atr_v=float(atr.iloc[-1]); entry=float(df.iloc[-1]["close"])
        if not np.isfinite(atr_v) or atr_v<=0 or atr_v/entry<self.min_atr_pct: return []
        st,direction=self._supertrend(df)
        d0=float(direction.iloc[-1]); d1=float(direction.iloc[-2]) if len(direction)>1 else d0
        st_up=d0<0; st_dn=d0>0
        if self.only_st_flip:
            st_up=d1>0 and d0<0; st_dn=d1<0 and d0>0
        bull_sw,bear_sw=self._sweep(df)
        pos,in_disc,in_prem=self._pd_zone(df, entry)
        signals=[]
        if self.enable_short:
            score=0; tags=[]; ok=True
            if st_dn: score+=1; tags.append("ST_DN")
            if self.require_st and not st_dn: ok=False
            if ok and bear_sw: score+=1; tags.append("SWEEP_BSL")
            elif ok and self.require_sweep: ok=False
            if ok and in_prem: score+=1; tags.append("PREMIUM")
            elif ok and self.require_pd: ok=False
            if ok and score>=self.min_score:
                sl=max(float(df["high"].iloc[-8:].max()), float(st.iloc[-1]) if np.isfinite(st.iloc[-1]) else entry)+self.sl_buffer_atr*atr_v
                if sl>entry:
                    risk=sl-entry; tp=entry-risk*self.rr
                    signals.append(Signal(symbol=symbol,side=Side.SHORT,entry=entry,sl=float(sl),tp=float(tp),confidence=min(0.88,0.55+0.1*score),strategy=self.name,reason=f"S{score} {'+'.join(tags)}",timeframe=timeframe))
        if self.enable_long:
            score=0; tags=[]; ok=True
            if st_up: score+=1; tags.append("ST_UP")
            if self.require_st and not st_up: ok=False
            if ok and bull_sw: score+=1; tags.append("SWEEP_SSL")
            elif ok and self.require_sweep: ok=False
            if ok and in_disc: score+=1; tags.append("DISCOUNT")
            elif ok and self.require_pd: ok=False
            if ok and score>=self.min_score:
                sl=min(float(df["low"].iloc[-8:].min()), float(st.iloc[-1]) if np.isfinite(st.iloc[-1]) else entry)-self.sl_buffer_atr*atr_v
                if entry>sl:
                    risk=entry-sl; tp=entry+risk*self.rr
                    signals.append(Signal(symbol=symbol,side=Side.LONG,entry=entry,sl=float(sl),tp=float(tp),confidence=min(0.88,0.55+0.1*score),strategy=self.name,reason=f"L{score} {'+'.join(tags)}",timeframe=timeframe))
        return signals
