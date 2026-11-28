"""Discord helpers: map timeframe per strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

VALID_TFS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}


def _settings_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _save_settings(settings: dict) -> None:
    path = _settings_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    data["strategy_timeframes"] = settings.get("strategy_timeframes") or {}
    if "timeframes" in settings:
        data["timeframes"] = settings["timeframes"]
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def apply_tfstrat_command(scanner, settings: dict, args: List[str]) -> str:
    """
    !tfstrat
    !tfstrat VWAPEMA9 15m 1h
    !tfstrat clear Supertrend
    """
    settings = settings if isinstance(settings, dict) else {}
    st_map: Dict[str, List[str]] = dict(settings.get("strategy_timeframes") or {})

    # resolve strategy names from scanner if possible
    known = []
    try:
        for s in getattr(scanner, "strategies", []) or []:
            n = getattr(s, "name", None) or type(s).__name__.replace("Strategy", "")
            if n:
                known.append(str(n))
    except Exception:
        pass

    if not args:
        lines = ["**strategy_timeframes** (TF riêng từng strategy):", ""]
        if not st_map:
            lines.append("_Chưa cấu hình — mọi strategy dùng `timeframes` chung._")
        else:
            for k in sorted(st_map.keys()):
                tfs = st_map[k]
                lines.append(f"• **{k}**: {', '.join(tfs) if tfs else '—'}")
        lines.append("")
        lines.append("```\n!tfstrat VWAPEMA9 15m 1h\n!tfstrat clear Supertrend\n!tfstrat clear all\n```")
        if known:
            lines.append("Strategies: " + ", ".join(known[:25]))
        return "\n".join(lines)

    a0 = str(args[0]).strip().lower()
    if a0 in ("clear", "rm", "delete", "reset"):
        if len(args) < 2:
            return "Dùng: `!tfstrat clear Supertrend` hoặc `!tfstrat clear all`"
        target = str(args[1]).strip()
        if target.lower() == "all":
            st_map = {}
            settings["strategy_timeframes"] = st_map
            try:
                if hasattr(scanner, "strategy_timeframes"):
                    scanner.strategy_timeframes = {}
            except Exception:
                pass
            try:
                _save_settings(settings)
            except Exception as e:
                logger.warning(f"save strategy_timeframes: {e}")
            return "✅ Đã xóa **toàn bộ** strategy_timeframes (mọi strategy dùng TF chung)."
        # match name
        key = target
        for k in list(st_map.keys()):
            if k.lower() == target.lower():
                key = k
                break
        if key in st_map:
            del st_map[key]
            settings["strategy_timeframes"] = st_map
            try:
                if hasattr(scanner, "strategy_timeframes"):
                    scanner.strategy_timeframes = dict(st_map)
            except Exception:
                pass
            try:
                _save_settings(settings)
            except Exception as e:
                logger.warning(f"save: {e}")
            return f"✅ Đã xóa TF riêng của **{key}**."
        return f"⚠️ Không thấy `{target}` trong map."

    # set: name tf1 tf2 ...
    name = str(args[0]).strip()
    tfs = []
    for x in args[1:]:
        t = str(x).strip().lower()
        if t in VALID_TFS and t not in tfs:
            tfs.append(t)
    if not tfs:
        return f"Cần TF hợp lệ: {', '.join(sorted(VALID_TFS))}\nVD: `!tfstrat {name} 15m 1h`"

    # canonicalize name
    key = name
    for n in known:
        if n.lower() == name.lower():
            key = n
            break
    st_map[key] = tfs
    settings["strategy_timeframes"] = st_map
    try:
        if hasattr(scanner, "strategy_timeframes"):
            scanner.strategy_timeframes = dict(st_map)
    except Exception as e:
        logger.warning(f"scanner update: {e}")
    try:
        _save_settings(settings)
    except Exception as e:
        logger.warning(f"save settings: {e}")
        return f"✅ Runtime: **{key}** → {tfs}\n⚠️ Không ghi được settings.yaml: {e}"
    return f"✅ **{key}** chỉ quét TF: `{', '.join(tfs)}` (đã lưu settings.yaml)"
