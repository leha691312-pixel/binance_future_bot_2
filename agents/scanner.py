"""
Market Scanner - Top N symbols × strategies.
- Parallel scan (ThreadPoolExecutor)
- Per-strategy timeframe map (strategy_timeframes)
"""
from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from loguru import logger
from core.exchange import ExchangeWrapper
from agents.strategies import get_all_strategies, list_strategy_names
from core.models import Signal

try:
    from agents.signal_refine import refine_signals
except Exception:
    refine_signals = None


# TF aliases for config convenience
_TF_ALIAS = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h",
    "12H": "12h", "1D": "1d", "D": "1d",
}


def normalize_tf(tf: str) -> str:
    if not tf:
        return ""
    t = str(tf).strip()
    if t in _TF_ALIAS:
        return _TF_ALIAS[t]
    t2 = t.lower().replace(" ", "")
    if t2 in ("1", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"):
        return t2
    return t2


class ScannerAgent:
    def __init__(
        self,
        exchange: ExchangeWrapper,
        top_n: int = 20,
        timeframes: List[str] = None,
        enabled_strategies: Optional[List[str]] = None,
        strategy_params: Optional[Dict[str, dict]] = None,
        excluded_symbols: Optional[List[str]] = None,
        parallel: bool = True,
        max_workers: int = 8,
        refine_cfg: Optional[Dict[str, Any]] = None,
        style_map: Optional[Dict[str, str]] = None,
        strategy_timeframes: Optional[Dict[str, List[str]]] = None,
    ):
        self.exchange = exchange
        self.top_n = top_n
        self.timeframes = [normalize_tf(x) for x in (timeframes or ["15m", "1h"]) if normalize_tf(x)]
        if not self.timeframes:
            self.timeframes = ["15m", "1h"]
        self.enabled_strategies = enabled_strategies
        self.strategy_params = strategy_params or {}
        self.strategies = get_all_strategies(
            enabled=enabled_strategies,
            strategy_params=strategy_params,
        )
        self.last_dfs: Dict[str, pd.DataFrame] = {}
        self.excluded_symbols: set = set()
        self.set_excluded_symbols(excluded_symbols or [])
        self.parallel = bool(parallel)
        self.max_workers = max(1, int(max_workers or 1))
        self.refine_cfg = dict(refine_cfg or {})
        self.style_map = dict(style_map or {})
        self.strategy_timeframes: Dict[str, List[str]] = {}
        self.set_strategy_timeframes(strategy_timeframes or {})

        names = [s.name for s in self.strategies]
        logger.info(
            f"ScannerAgent ready | Top {top_n} | TFs={self.timeframes} | "
            f"strat_tf_map={len(self.strategy_timeframes)} | "
            f"parallel={self.parallel} workers={self.max_workers} | "
            f"Excluded={len(self.excluded_symbols)} | "
            f"Strategies ({len(names)}): {names}"
        )
        if self.strategy_timeframes:
            for k, v in sorted(self.strategy_timeframes.items()):
                logger.info(f"  strategy_timeframes[{k}] = {v}")

    def set_strategy_timeframes(self, mapping: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Normalize map: {StrategyName: [tf, ...]}."""
        out: Dict[str, List[str]] = {}
        for raw_name, tfs in (mapping or {}).items():
            name = str(raw_name).strip()
            if not name:
                continue
            if isinstance(tfs, str):
                raw_list = [x.strip() for x in tfs.replace(";", ",").split(",") if x.strip()]
            elif isinstance(tfs, (list, tuple)):
                raw_list = list(tfs)
            else:
                continue
            norm = []
            for x in raw_list:
                t = normalize_tf(str(x))
                if t and t not in norm:
                    norm.append(t)
            if norm:
                out[name] = norm
                # case-insensitive lookup helper keys
                out.setdefault(name.lower(), norm)
        self.strategy_timeframes = out
        return {k: v for k, v in out.items() if k[0].isupper() or k in [s.name for s in self.strategies]}

    def _tfs_for_strategy(self, strat_name: str) -> List[str]:
        """Per-strategy TFs, else global timeframes."""
        m = self.strategy_timeframes or {}
        if strat_name in m and m[strat_name]:
            return list(m[strat_name])
        low = (strat_name or "").lower()
        if low in m and m[low]:
            return list(m[low])
        # try without underscores
        alt = low.replace("_", "")
        for k, v in m.items():
            if k.replace("_", "").lower() == alt and v:
                return list(v)
        return list(self.timeframes)

    def _all_needed_timeframes(self) -> List[str]:
        """Union of global + all strategy-specific TFs (stable order)."""
        ordered: List[str] = []
        for tf in self.timeframes:
            if tf not in ordered:
                ordered.append(tf)
        for strat in self.strategies:
            for tf in self._tfs_for_strategy(strat.name):
                if tf not in ordered:
                    ordered.append(tf)
        return ordered or list(self.timeframes)

    @staticmethod
    def normalize_symbol(sym: str, quote: str = "USDT") -> str:
        if not sym:
            return ""
        s = str(sym).strip().upper().replace(" ", "")
        if not s:
            return ""
        if f":{quote}" in s:
            return s
        if s.endswith(f"/{quote}"):
            return f"{s}:{quote}"
        if s.endswith(quote) and "/" not in s:
            base = s[: -len(quote)]
            if base:
                return f"{base}/{quote}:{quote}"
        if "/" in s and ":" not in s:
            return f"{s}:{quote}"
        if "/" not in s and ":" not in s:
            return f"{s}/{quote}:{quote}"
        return s

    def set_excluded_symbols(self, symbols) -> int:
        out = set()
        for x in symbols or []:
            n = self.normalize_symbol(str(x))
            if n:
                out.add(n)
                base = n.split("/")[0]
                if base:
                    out.add(base)
        self.excluded_symbols = out
        return len(self.excluded_symbols)

    def is_excluded(self, symbol: str) -> bool:
        if not self.excluded_symbols:
            return False
        n = self.normalize_symbol(symbol)
        if n in self.excluded_symbols:
            return True
        base = n.split("/")[0] if n else ""
        return bool(base and base in self.excluded_symbols)

    def excluded_list_display(self) -> list:
        return sorted(self.excluded_symbols)

    def reload_strategies(
        self,
        enabled: Optional[List[str]] = None,
        strategy_params: Optional[Dict[str, dict]] = None,
    ):
        if strategy_params is not None:
            self.strategy_params = strategy_params
        if enabled is not None:
            self.enabled_strategies = enabled
        self.strategies = get_all_strategies(
            enabled=self.enabled_strategies,
            strategy_params=self.strategy_params,
        )
        logger.info(f"Strategies reloaded: {[s.name for s in self.strategies]}")

    def _scan_one_symbol(self, symbol: str) -> Tuple[List[Signal], Dict[str, pd.DataFrame], Counter]:
        local_signals: List[Signal] = []
        local_dfs: Dict[str, pd.DataFrame] = {}
        per_strat: Counter = Counter()

        funding_rate = None
        oi_value = None
        try:
            fr = self.exchange.fetch_funding_rate(symbol)
            if fr and "fundingRate" in fr:
                funding_rate = float(fr["fundingRate"])
        except Exception:
            pass
        try:
            oi = self.exchange.fetch_open_interest(symbol)
            if oi and "openInterestAmount" in oi:
                oi_value = float(oi["openInterestAmount"])
            elif oi and "openInterest" in oi:
                oi_value = float(oi["openInterest"])
        except Exception:
            pass

        # Precompute which strategies run on which TF
        strat_tfs: Dict[str, List[str]] = {
            s.name: self._tfs_for_strategy(s.name) for s in self.strategies
        }
        needed_tfs: List[str] = []
        for tlist in strat_tfs.values():
            for tf in tlist:
                if tf not in needed_tfs:
                    needed_tfs.append(tf)
        if not needed_tfs:
            needed_tfs = list(self.timeframes)

        for tf in needed_tfs:
            # strategies that want this TF
            strats_for_tf = [s for s in self.strategies if tf in strat_tfs.get(s.name, [])]
            if not strats_for_tf:
                continue
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=200)
                if not ohlcv or len(ohlcv) < 50:
                    continue
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                local_dfs[symbol] = df
                local_dfs[f"{symbol}_{tf}"] = df

                for strat in strats_for_tf:
                    try:
                        if strat.name == "OI_Funding":
                            oi_series = None
                            if oi_value is not None:
                                oi_series = pd.Series(
                                    [oi_value * 0.97] * (len(df) - 1) + [oi_value]
                                )
                            signals = strat.generate(
                                df, symbol, tf,
                                oi_series=oi_series,
                                funding_rate=funding_rate,
                            )
                        else:
                            signals = strat.generate(df, symbol, tf)

                        if signals:
                            local_signals.extend(signals)
                            per_strat[strat.name] += len(signals)
                    except Exception as e:
                        logger.debug(f"Strategy {strat.name} error {symbol} {tf}: {e}")
            except Exception as e:
                logger.warning(f"Scan error {symbol} {tf}: {e}")
                continue

        return local_signals, local_dfs, per_strat

    def scan(self) -> List[Signal]:
        symbols = self.exchange.get_top_symbols(self.top_n)
        before = len(symbols)
        symbols = [s for s in symbols if not self.is_excluded(s)]
        if before != len(symbols):
            logger.info(f"Excluded {before - len(symbols)} symbol(s) from scan blacklist")

        needed = self._all_needed_timeframes()
        logger.info(
            f"Scanning {len(symbols)} symbols | TFs={needed} | "
            f"strategies={len(self.strategies)} | parallel={self.parallel} "
            f"workers={self.max_workers if self.parallel else 1} | "
            f"blacklist={len(self.excluded_symbols)} | "
            f"per_strategy_tf={bool(self.strategy_timeframes)}"
        )

        all_signals: List[Signal] = []
        per_strat: Counter = Counter()
        self.last_dfs = {}

        if not symbols:
            logger.info("Found 0 raw signals")
            return []

        workers = self.max_workers if self.parallel else 1
        workers = max(1, min(workers, len(symbols)))

        if workers == 1:
            for symbol in symbols:
                sigs, dfs, cnt = self._scan_one_symbol(symbol)
                all_signals.extend(sigs)
                self.last_dfs.update(dfs)
                per_strat.update(cnt)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self._scan_one_symbol, sym): sym for sym in symbols}
                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        sigs, dfs, cnt = fut.result()
                        all_signals.extend(sigs)
                        self.last_dfs.update(dfs)
                        per_strat.update(cnt)
                    except Exception as e:
                        logger.warning(f"Parallel scan failed {sym}: {e}")

        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        if per_strat:
            detail = ", ".join(
                f"{k}={v}" for k, v in sorted(per_strat.items(), key=lambda x: -x[1])
            )
            logger.info(f"Found {len(all_signals)} raw signals | by strategy: {detail}")
        else:
            logger.info("Found 0 raw signals")

        if all_signals and self.refine_cfg and refine_signals is not None:
            try:
                all_signals = refine_signals(
                    all_signals,
                    exchange=self.exchange,
                    cfg=self.refine_cfg,
                    df_cache=self.last_dfs,
                    style_map=self.style_map or None,
                )
            except Exception as e:
                logger.warning(f"signal refine failed (using raw): {e}")

        return all_signals

    @staticmethod
    def available_strategies() -> List[str]:
        return list_strategy_names()
