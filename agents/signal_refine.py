"""
Lam sach signal sau khi scan:
1) Consensus – can >= N strategy cung symbol + side
2) HTF bias – khong vao nguoc xu huong khung lon (EMA)
3) (Optional) bias strategy phai cung huong voi trigger
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, time as dtime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from loguru import logger
from core.models import Signal, Side


DEFAULT_STYLE = {
    "ORB": "scalp",
    "LiquiditySweep": "scalp",
    "MARibbon": "scalp",
    "StochBB": "scalp",
    "PivotScalp": "scalp",
    "EMARSIScalp": "scalp",
    "OrderBlock": "day",
    "PremiumDiscount": "day",
    "SupplyDemand": "day",
    "FVG": "day",
    "BOS_CHOCH": "day",
    "TrendMA": "day",
    "MomentumRSI": "day",
    "PullbackEMA": "day",
    "RangeBounce": "day",
    "OI_Funding": "swing",
    "SwingMACross": "swing",
    "SwingRSI": "swing",
    "FibPullback": "swing",
    "BBSqueeze": "swing",
}

DEFAULT_BIAS_STRATEGIES = {
    "TrendMA",
    "BOS_CHOCH",
    "SwingMACross",
    "OI_Funding",
}
DEFAULT_TRIGGER_STRATEGIES = {
    "FVG",
    "PremiumDiscount",
    "OrderBlock",
    "SupplyDemand",
    "PullbackEMA",
    "LiquiditySweep",
    "StochBB",
    "MARibbon",
    "PivotScalp",
    "EMARSIScalp",
    "ORB",
    "MomentumRSI",
    "RangeBounce",
    "FibPullback",
    "BBSqueeze",
    "SwingRSI",
}


def _side_key(sig: Signal) -> str:
    return sig.side.value if hasattr(sig.side, "value") else str(sig.side)


def apply_consensus(
    signals: List[Signal],
    min_agree: int = 2,
    same_style: bool = False,
    style_map: Optional[Dict[str, str]] = None,
) -> List[Signal]:
    if not signals or min_agree <= 1:
        return list(signals)

    style_map = style_map or DEFAULT_STYLE
    groups: Dict[Tuple[str, str], List[Signal]] = defaultdict(list)

    for s in signals:
        groups[(s.symbol, _side_key(s))].append(s)

    out: List[Signal] = []
    dropped = 0

    for (symbol, side), group in groups.items():
        by_strat: Dict[str, Signal] = {}
        for s in group:
            name = s.strategy or "unknown"
            prev = by_strat.get(name)
            if prev is None or s.confidence > prev.confidence:
                by_strat[name] = s

        if same_style:
            style_buckets: Dict[str, List[Signal]] = defaultdict(list)
            for s in by_strat.values():
                st = style_map.get(s.strategy, "day")
                style_buckets[st].append(s)
            best_style, best_list = max(
                style_buckets.items(), key=lambda kv: len(kv[1])
            )
            candidates = best_list
            style_tag = best_style
        else:
            candidates = list(by_strat.values())
            style_tag = "any"

        n = len(candidates)
        if n < min_agree:
            dropped += len(group)
            continue

        best = max(
            candidates,
            key=lambda s: (s.confidence, s.risk_reward()),
        )
        names = sorted({s.strategy for s in candidates})
        boost = min(0.15, 0.04 * (n - 1))
        new_conf = min(0.98, float(best.confidence) + boost)

        reason = best.reason or ""
        agree_note = f"CONSENSUS x{n} [{', '.join(names)}]"
        if reason:
            reason = f"{agree_note} | {reason}"
        else:
            reason = agree_note

        extra = dict(best.extra or {})
        extra["agree_count"] = n
        extra["agree_strategies"] = names
        extra["consensus_style"] = style_tag

        out.append(
            Signal(
                symbol=best.symbol,
                side=best.side,
                entry=best.entry,
                sl=best.sl,
                tp=best.tp,
                confidence=new_conf,
                strategy=best.strategy,
                reason=reason,
                timeframe=best.timeframe,
                timestamp=best.timestamp,
                extra=extra,
            )
        )

    logger.info(
        f"Consensus: in={len(signals)} out={len(out)} dropped~{dropped} "
        f"| min_agree={min_agree} same_style={same_style}"
    )
    return out


def apply_bias_trigger(
    signals: List[Signal],
    require_bias: bool = True,
    bias_strategies: Optional[set] = None,
    trigger_strategies: Optional[set] = None,
) -> List[Signal]:
    if not require_bias or not signals:
        return list(signals)

    bias_strategies = bias_strategies or DEFAULT_BIAS_STRATEGIES
    trigger_strategies = trigger_strategies or DEFAULT_TRIGGER_STRATEGIES

    groups: Dict[Tuple[str, str], List[Signal]] = defaultdict(list)
    for s in signals:
        groups[(s.symbol, _side_key(s))].append(s)

    out: List[Signal] = []
    for key, group in groups.items():
        names = {s.strategy for s in group}
        has_bias = bool(names & bias_strategies)
        has_trig = bool(names & trigger_strategies)
        unknown = names - bias_strategies - trigger_strategies
        if unknown:
            has_trig = True
        if has_bias and has_trig:
            out.extend(group)
        elif not has_bias and has_trig:
            if len(names) >= 2:
                out.extend(group)

    logger.info(f"Bias/Trigger filter: in={len(signals)} out={len(out)}")
    return out


def _htf_trend(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> Optional[str]:
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    if len(df) < max(ema_slow, ema_fast) + 2:
        return None
    close = df["close"].astype(float)
    ef = close.ewm(span=ema_fast, adjust=False).mean()
    es = close.ewm(span=ema_slow, adjust=False).mean()
    c = float(close.iloc[-1])
    f = float(ef.iloc[-1])
    s = float(es.iloc[-1])
    if f > s and c >= s * 0.998:
        return "LONG"
    if f < s and c <= s * 1.002:
        return "SHORT"
    return None



# --- Killzones (ICT-style sessions, UTC) ---
# Default windows approximate London / New York active hours.
DEFAULT_KILLZONES = {
    "asia": {"start": "00:00", "end": "09:00", "enabled": False},
    "london": {"start": "07:00", "end": "10:00", "enabled": True},
    "ny_am": {"start": "12:00", "end": "15:00", "enabled": True},
    "ny_pm": {"start": "15:00", "end": "18:00", "enabled": False},
    "london_close": {"start": "15:00", "end": "17:00", "enabled": False},
}


def _parse_hhmm(s: str) -> dtime:
    parts = str(s).strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return dtime(hour=h % 24, minute=m % 60)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def in_killzone(
    now: Optional[datetime] = None,
    zones: Optional[dict] = None,
) -> tuple:
    """Trả (active: bool, zone_name: str).

    Hỗ trợ cửa sổ qua nửa đêm (start > end).
    """
    now = now or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    t = now.timetz().replace(tzinfo=None)

    zones = zones if zones is not None else DEFAULT_KILLZONES
    active_names = []
    for name, z in (zones or {}).items():
        if isinstance(z, dict) and not z.get("enabled", True):
            continue
        if isinstance(z, dict):
            start = _parse_hhmm(z.get("start", "00:00"))
            end = _parse_hhmm(z.get("end", "23:59"))
        else:
            continue
        if start <= end:
            ok = start <= t <= end
        else:
            # wraps midnight
            ok = t >= start or t <= end
        if ok:
            active_names.append(str(name))
    if active_names:
        return True, "+".join(active_names)
    return False, ""


def apply_killzone_filter(
    signals: List[Signal],
    cfg: Optional[dict] = None,
    now: Optional[datetime] = None,
) -> List[Signal]:
    """Lọc / gắn nhãn killzone.

    mode:
      - off: bỏ qua
      - filter: chỉ giữ signal trong killzone
      - tag: giữ tất cả, gắn extra + giảm conf ngoài zone
      - boost: trong zone tăng conf, ngoài zone giữ nguyên (không chặn)
    """
    cfg = cfg or {}
    if not cfg.get("killzone_enabled", False):
        return list(signals)

    mode = str(cfg.get("killzone_mode", "filter")).lower()
    zones = cfg.get("killzones") or DEFAULT_KILLZONES
    active, zone_name = in_killzone(now=now, zones=zones)
    outside_mult = float(cfg.get("killzone_outside_conf_mult", 0.7) or 0.7)
    inside_boost = float(cfg.get("killzone_inside_conf_boost", 0.05) or 0.05)

    if mode == "filter" and not active:
        logger.info(f"Killzone filter: outside zone → drop {len(signals)} signals")
        return []

    out: List[Signal] = []
    for s in signals:
        extra = dict(s.extra or {})
        extra["killzone_active"] = active
        extra["killzone"] = zone_name or "none"
        conf = float(s.confidence or 0.5)
        if active:
            conf = min(0.98, conf + inside_boost)
        elif mode in ("tag", "boost") and mode == "tag":
            conf = max(0.1, conf * outside_mult)
        # mode boost: outside unchanged; filter already handled
        out.append(
            Signal(
                symbol=s.symbol,
                side=s.side,
                entry=s.entry,
                sl=s.sl,
                tp=s.tp,
                confidence=conf,
                strategy=s.strategy,
                reason=s.reason,
                timeframe=s.timeframe,
                timestamp=getattr(s, "timestamp", None),
                extra=extra,
            )
        )

    logger.info(
        f"Killzone mode={mode} active={active} zone={zone_name or '-'} "
        f"in={len(signals)} out={len(out)}"
    )
    return out


def apply_multi_htf_bias(
    exchange,
    signals: List[Signal],
    timeframes: Optional[List[str]] = None,
    ema_fast: int = 21,
    ema_slow: int = 50,
    df_cache: Optional[Dict[str, pd.DataFrame]] = None,
    require_all_align: bool = False,
    min_align: int = 1,
) -> List[Signal]:
    """Multi-timeframe bias: kiểm tra xu hướng trên nhiều TF (vd 1h + 4h).

    - require_all_align=True: mọi HTF phải cùng chiều với signal
    - min_align: số TF tối thiểu đồng thuận với side (khi không require_all)
    - TF không lấy được data → bỏ qua TF đó (không block)
    """
    if not signals:
        return []
    timeframes = timeframes or ["1h", "4h"]
    timeframes = [str(t) for t in timeframes if t]
    if not timeframes:
        return list(signals)

    df_cache = df_cache if df_cache is not None else {}
    # trend_cache[(sym, tf)] = LONG|SHORT|None
    trend_cache: Dict[tuple, Optional[str]] = {}
    out: List[Signal] = []
    blocked = 0

    def _trend(sym: str, tf: str) -> Optional[str]:
        key = (sym, tf)
        if key in trend_cache:
            return trend_cache[key]
        cache_key = f"{sym}_{tf}"
        df = df_cache.get(cache_key)
        if df is None or (hasattr(df, "empty") and df.empty) or len(df) < ema_slow + 2:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, tf, limit=max(120, ema_slow + 30))
                if ohlcv and len(ohlcv) >= ema_slow:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df_cache[cache_key] = df
                else:
                    df = None
            except Exception as e:
                logger.debug(f"MTF fetch {sym} {tf}: {e}")
                df = None
        trend_cache[key] = _htf_trend(df, ema_fast, ema_slow) if df is not None else None
        return trend_cache[key]

    for s in signals:
        sym = s.symbol
        side = _side_key(s)
        biases = {}
        for tf in timeframes:
            biases[tf] = _trend(sym, tf)

        known = {tf: b for tf, b in biases.items() if b is not None}
        if not known:
            # no HTF data → pass through
            out.append(s)
            continue

        agree = sum(1 for b in known.values() if b == side)
        disagree = sum(1 for b in known.values() if b != side)

        ok = False
        if require_all_align:
            ok = disagree == 0 and agree == len(known)
        else:
            ok = agree >= min(min_align, len(known)) and agree > disagree

        if ok:
            extra = dict(s.extra or {})
            extra["mtf_bias"] = biases
            extra["mtf_agree"] = agree
            # boost nhẹ khi nhiều TF đồng thuận
            conf = float(s.confidence or 0.5)
            if agree >= 2:
                conf = min(0.98, conf + 0.04 * (agree - 1))
            out.append(
                Signal(
                    symbol=s.symbol,
                    side=s.side,
                    entry=s.entry,
                    sl=s.sl,
                    tp=s.tp,
                    confidence=conf,
                    strategy=s.strategy,
                    reason=s.reason,
                    timeframe=s.timeframe,
                    timestamp=getattr(s, "timestamp", None),
                    extra=extra,
                )
            )
        else:
            blocked += 1

    logger.info(
        f"MTF bias {timeframes} require_all={require_all_align} min_align={min_align}: "
        f"in={len(signals)} out={len(out)} blocked={blocked}"
    )
    return out



def apply_htf_bias(
    exchange,
    signals: List[Signal],
    timeframe: str = "1h",
    ema_fast: int = 21,
    ema_slow: int = 50,
    df_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Signal]:
    if not signals:
        return []

    df_cache = df_cache if df_cache is not None else {}
    trend_cache: Dict[str, Optional[str]] = {}
    out: List[Signal] = []
    blocked = 0

    for s in signals:
        sym = s.symbol
        if sym not in trend_cache:
            df = df_cache.get(f"{sym}_{timeframe}")
            if df is None:
                df = df_cache.get(sym)
            if df is None or (hasattr(df, "empty") and df.empty) or len(df) < ema_slow + 2:
                try:
                    ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=max(120, ema_slow + 30))
                    if ohlcv and len(ohlcv) >= ema_slow:
                        df = pd.DataFrame(
                            ohlcv,
                            columns=["timestamp", "open", "high", "low", "close", "volume"],
                        )
                        df_cache[f"{sym}_{timeframe}"] = df
                except Exception as e:
                    logger.debug(f"HTF fetch {sym} {timeframe}: {e}")
                    df = None
            trend_cache[sym] = _htf_trend(df, ema_fast, ema_slow) if df is not None else None

        bias = trend_cache[sym]
        side = _side_key(s)
        if bias is None:
            out.append(s)
            continue
        if side == bias:
            extra = dict(s.extra or {})
            extra["htf_bias"] = bias
            extra["htf_tf"] = timeframe
            out.append(
                Signal(
                    symbol=s.symbol,
                    side=s.side,
                    entry=s.entry,
                    sl=s.sl,
                    tp=s.tp,
                    confidence=s.confidence,
                    strategy=s.strategy,
                    reason=s.reason,
                    timeframe=s.timeframe,
                    timestamp=s.timestamp,
                    extra=extra,
                )
            )
        else:
            blocked += 1

    logger.info(
        f"HTF bias ({timeframe} EMA{ema_fast}/{ema_slow}): "
        f"in={len(signals)} out={len(out)} blocked={blocked}"
    )
    return out


def refine_signals(
    signals: List[Signal],
    exchange=None,
    cfg: Optional[dict] = None,
    df_cache: Optional[Dict[str, pd.DataFrame]] = None,
    style_map: Optional[Dict[str, str]] = None,
) -> List[Signal]:
    cfg = cfg or {}
    if not signals:
        return []

    out = list(signals)

    if cfg.get("bias_trigger_enabled", False):
        out = apply_bias_trigger(
            out,
            require_bias=True,
            bias_strategies=set(cfg.get("bias_strategies") or DEFAULT_BIAS_STRATEGIES),
            trigger_strategies=set(cfg.get("trigger_strategies") or DEFAULT_TRIGGER_STRATEGIES),
        )

    if cfg.get("consensus_enabled", True):
        out = apply_consensus(
            out,
            min_agree=int(cfg.get("min_strategy_agree", 2)),
            same_style=bool(cfg.get("consensus_same_style", False)),
            style_map=style_map or DEFAULT_STYLE,
        )

    # Killzone (session filter) — trước HTF để giảm fetch thừa
    if cfg.get("killzone_enabled", False):
        out = apply_killzone_filter(out, cfg=cfg)

    # Multi-timeframe bias (1h+4h...) hoặc single HTF cũ
    if exchange is not None:
        mtf_tfs = cfg.get("mtf_timeframes") or cfg.get("htf_timeframes")
        if cfg.get("mtf_enabled", False) and mtf_tfs:
            out = apply_multi_htf_bias(
                exchange,
                out,
                timeframes=list(mtf_tfs),
                ema_fast=int(cfg.get("htf_ema_fast", 21)),
                ema_slow=int(cfg.get("htf_ema_slow", 50)),
                df_cache=df_cache,
                require_all_align=bool(cfg.get("mtf_require_all_align", False)),
                min_align=int(cfg.get("mtf_min_align", 1)),
            )
        elif cfg.get("htf_bias_enabled", True):
            out = apply_htf_bias(
                exchange,
                out,
                timeframe=str(cfg.get("htf_timeframe", "1h")),
                ema_fast=int(cfg.get("htf_ema_fast", 21)),
                ema_slow=int(cfg.get("htf_ema_slow", 50)),
                df_cache=df_cache,
            )

    out.sort(key=lambda s: s.confidence, reverse=True)
    logger.info(f"Signal refine done: final={len(out)}")
    return out
