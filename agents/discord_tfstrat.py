"""
Discord helpers: per-strategy timeframe map
Import/call from discord_control if desired.

Commands idea:
  !tfstrat                     → list map
  !tfstrat VWAPEMA9 15m 1h     → set map for one strategy
  !tfstrat clear VWAPEMA9      → remove override (use global TFs)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml
from loguru import logger

try:
    from agents.scanner import normalize_tf
except Exception:
    def normalize_tf(tf: str) -> str:
        return str(tf).strip().lower()


def _settings_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def load_strategy_timeframes(settings: Optional[dict] = None) -> Dict[str, List[str]]:
    if settings is None:
        try:
            with open(_settings_path(), encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
        except Exception:
            settings = {}
    raw = settings.get("strategy_timeframes") or {}
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            tfs = [normalize_tf(x) for x in v.replace(";", ",").split(",") if x.strip()]
        else:
            tfs = [normalize_tf(str(x)) for x in (v or [])]
        tfs = [t for t in tfs if t]
        if tfs:
            out[str(k)] = tfs
    return out


def save_strategy_timeframes(mapping: Dict[str, List[str]], settings: Optional[dict] = None) -> bool:
    path = _settings_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = settings or {}
    # only keep TitleCase keys for yaml cleanliness
    clean = {}
    for k, v in mapping.items():
        if not k or k != k.strip():
            continue
        if k[0].islower() and k.lower() == k:
            continue  # skip pure lowercase alias keys
        clean[k] = list(v)
    data["strategy_timeframes"] = clean
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return True
    except Exception as e:
        logger.error(f"save strategy_timeframes failed: {e}")
        return False


def format_tfstrat_list(scanner, settings: Optional[dict] = None) -> str:
    global_tfs = list(getattr(scanner, "timeframes", None) or ["15m", "1h"])
    mapping = load_strategy_timeframes(settings)
    lines = [f"**Global TFs:** `{', '.join(global_tfs)}`", "", "**Per-strategy:**"]
    names = [s.name for s in getattr(scanner, "strategies", [])]
    if not names:
        names = sorted({k for k in mapping if k[:1].isupper()})
    for n in names:
        tfs = mapping.get(n) or mapping.get(n.lower())
        if tfs:
            lines.append(f"• **{n}**: `{', '.join(tfs)}`")
        else:
            lines.append(f"• **{n}**: _(global)_ `{', '.join(global_tfs)}`")
    if not names:
        lines.append("_No strategies loaded_")
    return "\n".join(lines)


def apply_tfstrat_command(
    scanner,
    settings: dict,
    args: List[str],
) -> str:
    """
    args examples:
      [] -> list
      ['VWAPEMA9', '15m', '1h'] -> set
      ['clear', 'VWAPEMA9'] -> remove override
    """
    mapping = load_strategy_timeframes(settings)
    if not args:
        return format_tfstrat_list(scanner, settings)

    if args[0].lower() in ("clear", "rm", "remove", "reset") and len(args) >= 2:
        name = args[1]
        mapping.pop(name, None)
        mapping.pop(name.lower(), None)
        settings["strategy_timeframes"] = {k: v for k, v in mapping.items() if k[:1].isupper()}
        save_strategy_timeframes(settings["strategy_timeframes"], settings)
        if hasattr(scanner, "set_strategy_timeframes"):
            scanner.set_strategy_timeframes(settings["strategy_timeframes"])
        return f"✅ Cleared TF map for **{name}** (uses global TFs)"

    # set: Name tf1 tf2 ...
    name = args[0]
    tfs = [normalize_tf(x) for x in args[1:]]
    tfs = [t for t in tfs if t]
    if not tfs:
        return "Usage: `!tfstrat StrategyName 15m 1h` or `!tfstrat clear StrategyName`"
    mapping[name] = tfs
    settings["strategy_timeframes"] = {k: v for k, v in mapping.items() if not (k.islower() and k == k.lower() and k[:1].islower())}
    # always keep the explicit name
    settings["strategy_timeframes"][name] = tfs
    save_strategy_timeframes(settings["strategy_timeframes"], settings)
    if hasattr(scanner, "set_strategy_timeframes"):
        scanner.set_strategy_timeframes(settings["strategy_timeframes"])
    return f"✅ **{name}** → TF `{', '.join(tfs)}`"
