"""
Discord Control – UI tối ưu (Embed + Buttons + Select + Modal)

.env:
  DISCORD_BOT_TOKEN=...
  DISCORD_ALLOWED_IDS=123,456
  DISCORD_PREFIX=!

Dùng: !panel  (hoặc !menu / !ui)
"""
from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple
from pathlib import Path

import yaml
from loguru import logger

try:
    from agents import discord_ml as _discord_ml
except Exception:
    try:
        import discord_ml as _discord_ml  # type: ignore
    except Exception:
        _discord_ml = None

try:
    import discord
    from discord.ext import commands

    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    discord = None
    commands = None


ALLOWED_RISK_KEYS = {
    "default_leverage": (int, 1, 125),
    "max_leverage": (int, 1, 125),
    "risk_per_trade_pct": (float, 0.0005, 0.1),
    "max_position_pct": (float, 0.01, 1.0),
    "max_total_exposure_pct": (float, 0.05, 20.0),
    "max_open_positions": (int, 1, 50),
    "min_risk_reward": (float, 0.5, 10.0),
    "min_risk_reward_structure": (float, 0.5, 10.0),
    "max_daily_loss_pct": (float, 0.005, 0.5),
    "max_drawdown_pct": (float, 0.01, 0.5),
    "duplicate_window_seconds": (int, 0, 7200),
    "kill_on_consecutive_losses": (int, 1, 50),
    "cooldown_seconds": (int, 0, 3600),
    "atr_multiplier_sl": (float, 0.5, 5.0),
    "kill_on_api_error_rate": (float, 0.05, 1.0),
    "kill_on_high_volatility_atr": (float, 1.0, 20.0),
    "min_notional_usdt": (float, 1.0, 50.0),
    "max_hold_minutes": (int, 5, 20000),
}

ALLOWED_SETTINGS_KEYS = {
    "top_n": (int, 5, 500),
    "scan_interval_seconds": (int, 15, 3600),
    "min_quote_volume_top": (float, 0, 50_000_000),
    "scan_workers": (int, 1, 32),
    "ml_prob_threshold": (float, 0.3, 0.95),
    "ml_min_samples": (int, 20, 500),
    "ml_min_auc_deploy": (float, 0.45, 0.95),
    "ml_retrain_every_new": (int, 5, 200),
    "ml_retrain_hours": (float, 1, 168),
    # signal_refine / consensus
    "min_strategy_agree": (int, 1, 10),
    "mtf_min_align": (int, 1, 5),
    "htf_ema_fast": (int, 5, 100),
    "htf_ema_slow": (int, 10, 200),
}

# String / bool settings
ALLOWED_SETTINGS_STR = {
    "margin_mode": ("isolated", "cross"),
    "mode": ("paper", "testnet", "live"),
    "ml_collect_only": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "ml_filter_enabled": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "ml_per_style": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "scan_parallel": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "killzone_enabled": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "killzone_mode": ("filter", "tag", "boost"),

    "consensus_enabled": ("Consensus (bật/tắt)", "true/false — cần ≥N strategy cùng chiều"),
    "consensus_mode": ("Consensus mode", "on/off — alias consensus_enabled"),
    "min_strategy_agree": ("Consensus: số strategy tối thiểu", "1–10 (vd 2)"),
    "consensus_same_style": ("Consensus cùng style", "true = chỉ gộp scalp với scalp"),
    "bias_trigger_enabled": ("Bias+Trigger filter", "true/false"),
    "htf_bias_enabled": ("HTF bias (EMA khung lớn)", "true/false"),
    "mtf_enabled": ("MTF bias đa khung", "true/false"),
    "mtf_require_all_align": ("MTF: bắt buộc mọi TF cùng chiều", "true/false"),
    "mtf_min_align": ("MTF: số TF đồng thuận tối thiểu", "1–5"),
    "htf_ema_fast": ("HTF EMA nhanh", "vd 21"),
    "htf_ema_slow": ("HTF EMA chậm", "vd 50"),
    # consensus / refine
    "consensus_enabled": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "consensus_mode": ("on", "off", "true", "false", "1", "0"),  # alias → consensus_enabled
    "consensus_same_style": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "bias_trigger_enabled": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "htf_bias_enabled": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "mtf_enabled": ("true", "false", "1", "0", "yes", "no", "on", "off"),
    "mtf_require_all_align": ("true", "false", "1", "0", "yes", "no", "on", "off"),
}

# Nhãn tiếng Việt cho select / modal
PARAM_LABELS = {
    "default_leverage": ("Đòn bẩy mặc định", "1–125"),
    "max_leverage": ("Đòn bẩy tối đa", "1–125"),
    "risk_per_trade_pct": ("Risk mỗi lệnh", "vd 0.01 = 1%"),
    "max_position_pct": ("% vốn 1 vị thế", "0.01–1.0"),
    "max_total_exposure_pct": ("Tổng exposure", "0.05–20 (1=100% equity)"),
    "max_open_positions": ("Số vị thế tối đa", "1–50"),
    "min_risk_reward": ("RR tối thiểu", "0.5–10"),
    "min_risk_reward_structure": ("RR tối thiểu (structure TP)", "0.5–10"),
    "max_daily_loss_pct": ("Lỗ ngày tối đa", "0.005–0.5"),
    "max_drawdown_pct": ("Max drawdown", "0.01–0.5"),
    "duplicate_window_seconds": ("Chống lệnh trùng (giây)", "0–7200"),
    "kill_on_consecutive_losses": ("Kill sau N lệnh thua", "1–50"),
    "cooldown_seconds": ("Cooldown sau lệnh (giây)", "0–3600"),
    "atr_multiplier_sl": ("ATR SL multiplier", "0.5–5"),
    "kill_on_api_error_rate": ("Kill API error rate", "0.05–1.0"),
    "kill_on_high_volatility_atr": ("Kill high ATR%", "1–20"),
    "min_notional_usdt": ("Min notional USDT", "1–50"),
    "max_hold_minutes": ("Max hold (fallback phút)", "5–20000"),
    "top_n": ("Số symbol quét (top N vol)", "5–500"),
    "scan_interval_seconds": ("Chu kỳ scan (giây)", "15–3600"),
    "min_quote_volume_top": ("Min volume lọc top N", "0–50M"),
    "scan_workers": ("Số worker scan song song", "1–32"),
    "ml_prob_threshold": ("ML P(WIN) threshold", "0.3–0.95"),
    "ml_min_samples": ("ML min samples train", "20–500"),
    "ml_min_auc_deploy": ("ML min AUC deploy", "0.45–0.95"),
    "ml_retrain_every_new": ("ML retrain mỗi N sample", "5–200"),
    "ml_retrain_hours": ("ML retrain hours", "1–168"),
    "margin_mode": ("Margin mode", "isolated | cross"),
    "mode": ("Bot mode", "paper | testnet | live"),
    "ml_collect_only": ("ML chỉ gom data", "true/false"),
    "ml_filter_enabled": ("ML filter bật", "true/false"),
    "ml_per_style": ("ML theo style", "true/false"),
    "scan_parallel": ("Scan song song", "true/false"),
    "killzone_enabled": ("Killzone bật", "true/false"),
    "killzone_mode": ("Killzone mode", "filter | tag | boost"),
}

COLOR_OK = 0x2ECC71
COLOR_WARN = 0xF1C40F
COLOR_DANGER = 0xE74C3C
COLOR_INFO = 0x3498DB
COLOR_IDLE = 0x95A5A6


class BotControlState:
    def __init__(self):
        self.risk_gate = None
        self.executor = None
        self.exchange = None
        self.settings: Dict[str, Any] = {}
        self.risk_cfg: Dict[str, Any] = {}
        self.mode: str = "paper"
        self.paused: bool = False
        self.running: bool = True
        self.request_restart: bool = False
        self.last_cycle_info: str = ""
        self.scanner = None
        self.ml_agent = None

    def set_refs(self, risk_gate, executor, exchange, settings, risk_cfg, mode: str):
        self.risk_gate = risk_gate
        self.executor = executor
        self.exchange = exchange
        self.settings = settings or {}
        self.risk_cfg = risk_cfg or {}
        self.mode = mode


control_state = BotControlState()


def _parse_allowed_ids() -> Set[int]:
    raw = os.getenv("DISCORD_ALLOWED_IDS", "").strip()
    ids: Set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if part.isdigit():
            ids.add(int(part))
    return ids


def _user_authorized(user_id: int) -> bool:
    allowed = _parse_allowed_ids()
    return bool(allowed) and user_id in allowed


def _is_authorized(ctx) -> bool:
    return _user_authorized(ctx.author.id)


async def _check_interaction(interaction) -> bool:
    if not _user_authorized(interaction.user.id):
        msg = "⛔ Bạn không có quyền điều khiển bot."
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
        return False
    return True


def _save_risk_yaml(risk_cfg: dict):
    path = Path(__file__).parent.parent / "config" / "risk.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                risk_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
    except Exception as e:
        logger.warning(f"Failed to save risk.yaml: {e}")


def _save_settings_yaml(settings: dict):
    path = Path(__file__).parent.parent / "config" / "settings.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
    except Exception as e:
        logger.warning(f"Failed to save settings.yaml: {e}")


if _discord_ml is not None:
    try:
        _discord_ml.bind(lambda: control_state, _save_settings_yaml)
    except Exception as e:
        logger.warning(f"discord_ml bind failed: {e}")


def _normalize_symbol(sym: str, quote: str = "USDT") -> str:
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


def _get_excluded_list() -> list:
    st = control_state
    raw = list(st.settings.get("excluded_symbols") or [])
    sc = getattr(st, "scanner", None)
    if sc is not None and hasattr(sc, "excluded_list_display"):
        try:
            live = sc.excluded_list_display()
            if live is not None:
                return list(live)
        except Exception:
            pass
    out, seen = [], set()
    for x in raw:
        n = _normalize_symbol(str(x))
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _persist_excluded(symbols: list) -> list:
    st = control_state
    cleaned, seen = [], set()
    for x in symbols:
        n = _normalize_symbol(str(x))
        if n and n not in seen:
            seen.add(n)
            cleaned.append(n)
    cleaned.sort()
    st.settings["excluded_symbols"] = cleaned
    _save_settings_yaml(st.settings)
    sc = getattr(st, "scanner", None)
    if sc is not None and hasattr(sc, "set_excluded_symbols"):
        sc.set_excluded_symbols(cleaned)
    return cleaned


def _format_excluded_msg() -> str:
    items = _get_excluded_list()
    if not items:
        return "📭 Blacklist trống — bot quét tất cả top_n."
    lines = "\n".join(f"• `{s}`" for s in items[:40])
    more = f"\n… +{len(items) - 40}" if len(items) > 40 else ""
    return f"🚫 **Blacklist ({len(items)}):**\n{lines}{more}"




# Strategy names known to bot (fallback if import fails)
_KNOWN_STRATEGIES = [
    "ORB", "LiquiditySweep", "MARibbon", "StochBB", "PivotScalp", "EMARSIScalp",
    "OrderBlock", "SupplyDemand", "FVG", "BOS_CHOCH", "TrendMA", "MomentumRSI",
    "PullbackEMA", "RangeBounce", "OI_Funding", "SwingMACross", "SwingRSI",
    "FibPullback", "BBSqueeze", "EMARSIMACDDay", "PremiumDiscount",
]


def _all_strategy_names() -> list:
    try:
        from agents.strategies import list_strategy_names
        names = list(list_strategy_names() or [])
        if names:
            return sorted(set(names))
    except Exception:
        pass
    try:
        from agents.strategies import STRATEGY_MAP
        return sorted(STRATEGY_MAP.keys())
    except Exception:
        pass
    return list(_KNOWN_STRATEGIES)


def _get_enabled_strategies() -> list:
    st = control_state
    # Ưu tiên list runtime từ scanner
    sc = getattr(st, "scanner", None)
    if sc is not None and getattr(sc, "strategies", None):
        return [s.name for s in sc.strategies]
    raw = st.settings.get("enabled_strategies")
    if raw is None:
        return _all_strategy_names()
    return [str(x).strip() for x in raw if str(x).strip()]


def _normalize_strategy_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    all_names = _all_strategy_names()
    lower_map = {n.lower(): n for n in all_names}
    if name in all_names:
        return name
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    # partial match
    matches = [n for n in all_names if name.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    return name  # keep as-is; may fail validation


def _persist_strategies(enabled: list) -> list:
    """Ghi settings.yaml + reload scanner runtime."""
    st = control_state
    cleaned = []
    seen = set()
    for n in enabled:
        n = _normalize_strategy_name(str(n))
        if not n or n in seen:
            continue
        seen.add(n)
        cleaned.append(n)
    st.settings["enabled_strategies"] = cleaned
    _save_settings_yaml(st.settings)
    sc = getattr(st, "scanner", None)
    if sc is not None and hasattr(sc, "reload_strategies"):
        try:
            sc.reload_strategies(
                enabled=cleaned if cleaned else None,
                strategy_params=st.settings.get("strategy_params") or {},
            )
        except Exception as e:
            logger.warning(f"reload_strategies failed: {e}")
    return cleaned


def _format_strategies_msg() -> str:
    all_n = _all_strategy_names()
    on = set(_get_enabled_strategies())
    lines_on = [f"✅ `{n}`" for n in all_n if n in on]
    lines_off = [f"⬜ `{n}`" for n in all_n if n not in on]
    # unknown enabled
    extra = [f"✅ `{n}` (custom)" for n in sorted(on) if n not in set(all_n)]
    body_on = "\n".join(lines_on + extra) or "_(không có)_"
    body_off = "\n".join(lines_off) or "_(không có)_"
    return (
        f"**Strategies bật ({len(on)}):**\n{body_on}\n\n"
        f"**Tắt / chưa bật ({len(lines_off)}):**\n{body_off}"
    )


def _log_file_path() -> Path:
    """Đường dẫn file log: settings.log_file hoặc logs/bot.log."""
    root = _repo_root()
    st = control_state.settings or {}
    rel = st.get("log_file") or "logs/bot.log"
    p = Path(str(rel))
    if not p.is_absolute():
        p = root / p
    return p


def _read_log_tail(n: int = 50) -> Tuple[bool, str]:
    """Đọc n dòng cuối log. Trả về (ok, text_for_discord)."""
    n = max(1, min(int(n), 200))
    path = _log_file_path()
    if not path.is_file():
        return False, f"Không thấy file log: `{path}`"
    try:
        max_bytes = min(512_000, n * 500)
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes), 0)
            raw = f.read().decode("utf-8", errors="replace")
        lines = raw.splitlines()
        if size > max_bytes and lines:
            lines = lines[1:]
        tail = lines[-n:] if len(lines) > n else lines
        if not tail:
            return True, "(file log trống)"
        body = chr(10).join(tail)
        header = f"📜 **{path.name}** — {len(tail)} dòng cuối"
        return True, header + chr(10) + "```" + chr(10) + body + chr(10) + "```"
    except Exception as e:
        return False, f"Đọc log lỗi: `{e}`"


def _repo_root() -> Path:
    """Thư mục gốc repo (chứa main.py, .git)."""
    return Path(__file__).resolve().parent.parent


def _git_pull() -> Tuple[bool, str]:
    """Pull code an toàn: stash config → pull --ff-only → restore config.

    Giữ nguyên config/settings.yaml và config/risk.yaml local để tránh conflict.
    """
    root = _repo_root()
    if not (root / ".git").exists():
        return False, f"Không thấy .git tại `{root}` — bot chưa clone bằng git?"

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    cfg_files = ["config/settings.yaml", "config/risk.yaml"]
    logs: list = []

    def run(cmd: list, timeout: int = 120) -> Tuple[int, str]:
        try:
            p = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            out = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
            return p.returncode, out
        except FileNotFoundError:
            return 127, "Không tìm thấy lệnh `git` trên máy bot."
        except subprocess.TimeoutExpired:
            return 124, "git timeout."
        except Exception as e:
            return 1, str(e)[:500]

    # 1) Stash chỉ 2 file config (nếu có thay đổi)
    code, out = run(
        ["git", "stash", "push", "-m", "discord-pull-keep-cfg", "--"] + cfg_files
    )
    logs.append(f"[stash] {out or 'ok'}")
    stashed = code == 0 and "Saved working directory" in (out or "")

    # 2) Pull fast-forward
    code, out = run(["git", "pull", "--ff-only", "origin", "main"])
    pull_err = ""
    if code != 0:
        code2, out2 = run(["git", "pull", "--ff-only"])
        logs.append(f"[pull] {out2 or out}")
        ok = code2 == 0
        if not ok:
            pull_err = out2 or out
        else:
            out = out2
    else:
        logs.append(f"[pull] {out or 'Already up to date.'}")
        ok = True

    # 3) Restore config local từ stash
    if stashed:
        for f in cfg_files:
            c2, o2 = run(["git", "checkout", "stash@{0}", "--", f])
            logs.append(f"[restore {f}] {o2 or ('ok' if c2 == 0 else str(c2))}")
        run(["git", "stash", "drop"])
        logs.append("[stash drop] done")

    msg = "\n".join(logs)[-1800:]
    if not ok:
        return False, (pull_err or msg)[:1800]
    return True, msg or "Already up to date."



def _snapshot() -> Dict[str, Any]:
    """Lấy snapshot trạng thái an toàn."""
    st = control_state
    rg = st.risk_gate
    equity = float(getattr(rg, "equity", 0) or 0) if rg else 0.0
    try:
        if st.exchange:
            equity = float(st.exchange.get_balance())
            if rg:
                rg.update_equity(equity)
    except Exception:
        pass

    pos_n = len(rg.open_positions) if rg else 0
    max_pos = (rg.cfg.get("max_open_positions") if rg else st.risk_cfg.get("max_open_positions")) or "?"
    daily = float(getattr(rg, "daily_pnl", 0) or 0) if rg else 0.0
    kill = bool(getattr(rg, "kill_switch", False)) if rg else False
    kill_reason = str(getattr(rg, "kill_reason", "") or "") if rg else ""
    losses = int(getattr(rg, "consecutive_losses", 0) or 0) if rg else 0
    dlev = getattr(st.executor, "default_leverage", st.risk_cfg.get("default_leverage", "?"))
    mlev = getattr(st.executor, "max_leverage", st.risk_cfg.get("max_leverage", "?"))

    return {
        "mode": st.mode,
        "paused": st.paused,
        "equity": equity,
        "daily_pnl": daily,
        "pos_n": pos_n,
        "max_pos": max_pos,
        "kill": kill,
        "kill_reason": kill_reason,
        "losses": losses,
        "dlev": dlev,
        "mlev": mlev,
        "scan": st.settings.get("scan_interval_seconds", "?"),
        "top_n": st.settings.get("top_n", "?"),
        "excluded": len(_get_excluded_list()),
        "last_cycle": st.last_cycle_info or "—",
        "ready": rg is not None,
    }


def _panel_color(snap: dict) -> int:
    if not snap["ready"]:
        return COLOR_IDLE
    if snap["kill"]:
        return COLOR_DANGER
    if snap["paused"]:
        return COLOR_WARN
    return COLOR_OK


def _status_embed() -> "discord.Embed":
    snap = _snapshot()
    if not snap["ready"]:
        return discord.Embed(
            title="Bot chưa sẵn sàng",
            description="Đợi main loop khởi tạo xong rồi bấm **Làm mới**.",
            color=COLOR_IDLE,
        )

    if snap["kill"]:
        state = f"🛑 KILL — {snap['kill_reason'] or 'active'}"
    elif snap["paused"]:
        state = "⏸ Tạm dừng (pause)"
    else:
        state = "▶️ Đang chạy"

    pnl = snap["daily_pnl"]
    pnl_s = f"{'+' if pnl >= 0 else ''}{pnl:.4f}"

    emb = discord.Embed(
        title="📊 Trạng thái bot",
        color=_panel_color(snap),
        timestamp=datetime.now(timezone.utc),
    )
    emb.add_field(name="Trạng thái", value=state, inline=True)
    emb.add_field(name="Mode", value=f"`{snap['mode']}`", inline=True)
    emb.add_field(name="Equity", value=f"`{snap['equity']:.2f}` USDT", inline=True)
    emb.add_field(name="PnL ngày", value=f"`{pnl_s}`", inline=True)
    emb.add_field(
        name="Vị thế",
        value=f"`{snap['pos_n']}` / `{snap['max_pos']}`",
        inline=True,
    )
    emb.add_field(name="Thua liên tiếp", value=f"`{snap['losses']}`", inline=True)
    emb.add_field(
        name="Leverage",
        value=f"default `{snap['dlev']}` · max `{snap['mlev']}`",
        inline=True,
    )
    emb.add_field(
        name="Scan",
        value=f"mỗi `{snap['scan']}`s · top `{snap['top_n']}`",
        inline=True,
    )
    emb.add_field(
        name="Blacklist",
        value=f"`{snap.get('excluded', 0)}` symbol loại khỏi scan",
        inline=True,
    )
    emb.add_field(
        name="Cycle gần nhất",
        value=str(snap["last_cycle"])[:200],
        inline=False,
    )
    if _discord_ml is not None:
        try:
            emb.add_field(name="ML", value=_discord_ml.apply_ml_runtime()[:1020], inline=False)
        except Exception as e:
            emb.add_field(name="ML", value=str(e)[:200], inline=False)
    emb.set_footer(text="Binance Futures Bot")
    return emb


def _lookup_position_meta(symbol: str, side: str) -> dict:
    """Lấy strategy / trade_style từ RiskGate (lệnh bot mở)."""
    st = control_state
    rg = st.risk_gate
    if not rg:
        return {}
    side_u = (side or "").upper()
    key = f"{symbol}_{side_u}"
    pos = (rg.open_positions or {}).get(key)
    if pos is None:
        # fallback: match symbol only
        for k, p in (rg.open_positions or {}).items():
            if str(k).startswith(str(symbol) + "_"):
                pos = p
                break
    if pos is None:
        return {}
    strategy = getattr(pos, "strategy", None) or ""
    style = getattr(pos, "trade_style", None) or ""
    if not style and strategy and hasattr(rg, "resolve_trade_style"):
        try:
            style = rg.resolve_trade_style(strategy, getattr(pos, "timeframe", "") or "")
        except Exception:
            style = ""
    tf = getattr(pos, "timeframe", None) or ""
    return {"strategy": strategy, "trade_style": style, "timeframe": tf}


def _positions_embed() -> "discord.Embed":
    st = control_state
    emb = discord.Embed(title="📈 Vị thế đang mở", color=COLOR_INFO, timestamp=datetime.now(timezone.utc))
    if not st.exchange:
        emb.description = "Exchange chưa sẵn sàng."
        emb.color = COLOR_IDLE
        return emb
    try:
        positions = st.exchange.get_positions()
    except Exception as e:
        emb.description = f"Lỗi: `{e}`"
        emb.color = COLOR_DANGER
        return emb
    if not positions:
        emb.description = "Không có vị thế nào."
        emb.color = COLOR_IDLE
        return emb

    total_upnl = 0.0
    lines = []
    for p in positions[:20]:
        side = p.side.value if hasattr(p.side, "value") else str(p.side)
        upnl = float(getattr(p, "unrealized_pnl", 0) or 0)
        total_upnl += upnl
        flag = "🟢" if str(side).upper() == "LONG" else "🔴"
        meta = _lookup_position_meta(getattr(p, "symbol", ""), side)
        # strategy từ exchange object (hiếm) hoặc RiskGate
        strategy = (
            meta.get("strategy")
            or getattr(p, "strategy", None)
            or "—"
        )
        style = meta.get("trade_style") or getattr(p, "trade_style", None) or ""
        tf = meta.get("timeframe") or getattr(p, "timeframe", None) or ""
        strat_s = str(strategy) if strategy else "—"
        style_s = f" · `{style}`" if style else ""
        tf_s = f" · `{tf}`" if tf else ""
        lines.append(
            f"{flag} `{p.symbol}` **{side}** size=`{p.size}`\n"
            f"　strategy **`{strat_s}`**{style_s}{tf_s}\n"
            f"　entry `{p.entry_price:.6g}` · uPnL `{upnl:+.4f}`"
        )
    emb.description = "\n".join(lines)[:4000]
    emb.add_field(name="Tổng uPnL", value=f"`{total_upnl:+.4f}`", inline=True)
    emb.add_field(name="Số lệnh", value=f"`{len(positions)}`", inline=True)
    emb.set_footer(text="Strategy lấy từ RiskGate · tối đa 20 dòng")
    return emb


def _params_embed() -> "discord.Embed":
    st = control_state
    rg = st.risk_gate
    cfg = (rg.cfg if rg else st.risk_cfg) or {}
    emb = discord.Embed(
        title="⚙️ Thông số",
        description="Chọn trong menu **Chỉnh thông số** hoặc dùng preset đòn bẩy.",
        color=COLOR_INFO,
        timestamp=datetime.now(timezone.utc),
    )
    risk_lines = []
    for k in sorted(ALLOWED_RISK_KEYS.keys()):
        label = PARAM_LABELS.get(k, (k, ""))[0]
        risk_lines.append(f"• **{label}** (`{k}`): `{cfg.get(k, 'N/A')}`")
    emb.add_field(name="Risk", value="\n".join(risk_lines)[:1020], inline=False)
    set_lines = []
    for k in sorted(ALLOWED_SETTINGS_KEYS.keys()):
        label = PARAM_LABELS.get(k, (k, ""))[0]
        set_lines.append(f"• **{label}** (`{k}`): `{st.settings.get(k, 'N/A')}`")
    for k in sorted(ALLOWED_SETTINGS_STR.keys()):
        label = PARAM_LABELS.get(k, (k, ""))[0]
        set_lines.append(f"• **{label}** (`{k}`): `{st.settings.get(k, 'N/A')}`")
    emb.add_field(name="Settings", value="\n".join(set_lines), inline=False)
    
    # signal_refine / consensus
    sr = st.settings.get("signal_refine") or {}
    if isinstance(sr, dict) and sr:
        lines = []
        for k in (
            "consensus_enabled", "min_strategy_agree", "consensus_same_style",
            "bias_trigger_enabled", "htf_bias_enabled", "mtf_enabled",
            "mtf_require_all_align", "mtf_min_align", "killzone_enabled", "killzone_mode",
        ):
            if k in sr:
                lab = PARAM_LABELS.get(k, (k,))[0]
                lines.append(f"• **{lab}**: `{sr[k]}`")
        if lines:
            emb.add_field(name="Signal refine / Consensus", value="\n".join(lines)[:1020], inline=False)

    return emb


def _panel_embed() -> "discord.Embed":
    snap = _snapshot()
    if snap["kill"]:
        state = f"🛑 KILL ({snap['kill_reason'] or 'on'})"
    elif snap["paused"]:
        state = "⏸ PAUSED"
    elif snap["ready"]:
        state = "▶️ RUNNING"
    else:
        state = "⏳ STARTING"

    emb = discord.Embed(
        title="🎛️ Control Panel",
        description=(
            f"**{state}** · mode `{snap['mode']}`\n"
            f"Equity **{snap['equity']:.2f}** USDT · "
            f"PnL ngày `{snap['daily_pnl']:+.4f}` · "
            f"Positions **{snap['pos_n']}/{snap['max_pos']}**\n\n"
            f"Dùng nút bên dưới. Chỉ user trong `DISCORD_ALLOWED_IDS`."
        ),
        color=_panel_color(snap),
        timestamp=datetime.now(timezone.utc),
    )
    emb.add_field(
        name="Leverage",
        value=f"`{snap['dlev']}`x (max `{snap['mlev']}`)",
        inline=True,
    )
    emb.add_field(
        name="Scan",
        value=f"`{snap['scan']}`s · top `{snap['top_n']}`",
        inline=True,
    )
    emb.set_footer(text="Làm mới để cập nhật · Panel ~30 phút")
    return emb


def _apply_param(key: str, value: str) -> str:
    key = key.strip().lower()
    st = control_state

    if key in ALLOWED_RISK_KEYS:
        typ, lo, hi = ALLOWED_RISK_KEYS[key]
        try:
            val = typ(value)
        except ValueError:
            return f"❌ Giá trị không hợp lệ cho `{key}`"
        if not (lo <= val <= hi):
            return f"❌ `{key}` phải trong [{lo}, {hi}]"
        if st.risk_gate:
            st.risk_gate.cfg[key] = val
        st.risk_cfg[key] = val
        if key == "default_leverage" and st.executor:
            st.executor.default_leverage = int(val)
        if key == "max_leverage" and st.executor:
            st.executor.max_leverage = max(int(getattr(st.executor, "default_leverage", 1)), int(val))
        _save_risk_yaml(st.risk_gate.cfg if st.risk_gate else st.risk_cfg)
        label = PARAM_LABELS.get(key, (key,))[0]
        return f"✅ **{label}** = `{val}`"

    if key in ALLOWED_SETTINGS_KEYS:
        typ, lo, hi = ALLOWED_SETTINGS_KEYS[key]
        try:
            val = typ(value)
        except ValueError:
            return f"❌ Giá trị không hợp lệ cho `{key}`"
        if not (lo <= val <= hi):
            return f"❌ `{key}` phải trong [{lo}, {hi}]"
        refine_nums = {
            "min_strategy_agree", "mtf_min_align", "htf_ema_fast", "htf_ema_slow",
        }
        if key in refine_nums:
            sr = st.settings.setdefault("signal_refine", {})
            if not isinstance(sr, dict):
                sr = {}
                st.settings["signal_refine"] = sr
            sr[key] = val
            _save_settings_yaml(st.settings)
            _reload_refine_runtime(st)
            label = PARAM_LABELS.get(key, (key,))[0]
            return f"✅ **{label}** = `{val}` (signal_refine)"
        st.settings[key] = val
        if key == "ml_prob_threshold" and st.ml_agent is not None:
            try:
                st.ml_agent.prob_threshold = float(val)
            except Exception as e:
                logger.warning(f"Apply ml_prob_threshold: {e}")
        _save_settings_yaml(st.settings)
        label = PARAM_LABELS.get(key, (key,))[0]
        return f"✅ **{label}** = `{val}`"

    # String / bool settings
    if key in ALLOWED_SETTINGS_STR:
        allowed = ALLOWED_SETTINGS_STR[key]
        raw = str(value).strip().lower()
        if raw not in allowed:
            return f"❌ `{key}` chỉ nhận: {', '.join(allowed)}"

        bool_keys = {
            "ml_collect_only", "ml_filter_enabled", "ml_per_style",
            "scan_parallel", "killzone_enabled",
            "consensus_enabled", "consensus_mode", "consensus_same_style",
            "bias_trigger_enabled", "htf_bias_enabled", "mtf_enabled",
            "mtf_require_all_align",
        }
        if key in bool_keys:
            val = raw in ("true", "1", "yes", "on")
        else:
            val = raw

        # Nested signal_refine keys (consensus / HTF / MTF / killzone)
        refine_bool_or_str = {
            "killzone_enabled", "killzone_mode",
            "consensus_enabled", "consensus_same_style",
            "bias_trigger_enabled", "htf_bias_enabled", "mtf_enabled",
            "mtf_require_all_align",
        }
        if key == "consensus_mode":
            key = "consensus_enabled"
            val = bool(val) if isinstance(val, bool) else raw in ("true", "1", "yes", "on")
        if key in refine_bool_or_str:
            sr = st.settings.setdefault("signal_refine", {})
            if not isinstance(sr, dict):
                sr = {}
                st.settings["signal_refine"] = sr
            sr[key] = val
            _save_settings_yaml(st.settings)
            _reload_refine_runtime(st)
            label = PARAM_LABELS.get(key, (key,))[0]
            return f"✅ **{label}** = `{val}` (signal_refine)"
        else:
            st.settings[key] = val

        if st.exchange is not None and key == "margin_mode":
            try:
                st.exchange.margin_mode = val
                if hasattr(st.exchange, "config") and isinstance(st.exchange.config, dict):
                    st.exchange.config["margin_mode"] = val
            except Exception as e:
                logger.warning(f"Apply margin_mode runtime failed: {e}")

        if st.ml_agent is not None and key in (
            "ml_collect_only", "ml_filter_enabled", "ml_per_style"
        ):
            try:
                if key == "ml_collect_only":
                    st.ml_agent.collect_only = bool(val)
                elif key == "ml_filter_enabled":
                    st.ml_agent.filter_enabled = bool(val)
                elif key == "ml_per_style":
                    if hasattr(st.ml_agent, "per_style"):
                        st.ml_agent.per_style = bool(val)
            except Exception as e:
                logger.warning(f"Apply ML runtime {key}: {e}")

        if key == "ml_prob_threshold" and st.ml_agent is not None:
            try:
                st.ml_agent.prob_threshold = float(val)
            except Exception:
                pass

        _save_settings_yaml(st.settings)
        label = PARAM_LABELS.get(key, (key,))[0]
        note = ""
        if key == "margin_mode":
            note = (
                "\n⚠️ Chỉ áp dụng cho **vị thế mới**. "
                "Vị thế đang mở giữ nguyên margin type trên sàn."
            )
        if key == "mode":
            note = "\n⚠️ Đổi mode cần **restart** bot để Exchange nạp lại."
        return f"✅ **{label}** = `{val}`{note}"

    return (
        f"❌ Key `{key}` không được phép.\n"
        "Dùng `!params` để xem danh sách, hoặc `!set <key> <value>`."
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

if HAS_DISCORD:

    class ConfirmKillView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=45)

        @discord.ui.button(
            label="Xác nhận KILL", style=discord.ButtonStyle.danger, emoji="🛑"
        )
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            st = control_state
            if st.risk_gate:
                st.risk_gate.activate_kill_switch("DISCORD_KILL")
            emb = discord.Embed(
                title="🛑 Kill switch ON",
                description="Bot sẽ flatten và tạm dừng vòng lặp.",
                color=COLOR_DANGER,
            )
            await interaction.response.edit_message(content=None, embed=emb, view=None)

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content="Đã hủy kill.", embed=None, view=None
            )

    class ConfirmRestartView(discord.ui.View):
        """Xác nhận restart bot (không flatten vị thế)."""

        def __init__(self):
            super().__init__(timeout=45)

        @discord.ui.button(
            label="Xác nhận RESTART", style=discord.ButtonStyle.danger, emoji="🔁"
        )
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.request_restart = True
            emb = discord.Embed(
                title="🔁 Restart đã yêu cầu",
                description=(
                    "Bot sẽ **khởi động lại process** trong vài giây.\n"
                    "Vị thế trên sàn **không** bị đóng.\n"
                    "Discord có thể mất kết nối tạm rồi online lại."
                ),
                color=COLOR_WARN,
            )
            await interaction.response.edit_message(content=None, embed=emb, view=None)
            logger.warning(
                f"Discord restart requested by user {interaction.user.id}"
            )

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content="Đã hủy restart.", embed=None, view=None
            )

    class ConfirmPullRestartView(discord.ui.View):
        """git pull rồi restart process."""

        def __init__(self):
            super().__init__(timeout=60)

        @discord.ui.button(
            label="Pull & Restart", style=discord.ButtonStyle.danger, emoji="📥"
        )
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            ok, log = _git_pull()
            logger.info(f"Discord git pull ok={ok} | {log[:300]}")
            if not ok:
                emb = discord.Embed(
                    title="❌ git pull thất bại",
                    description=f"```\n{log}\n```\nBot **không** restart. Xử lý conflict trên máy bot rồi thử lại.",
                    color=COLOR_DANGER,
                )
                await interaction.followup.send(embed=emb, ephemeral=True)
                return
            control_state.request_restart = True
            emb = discord.Embed(
                title="📥 Pull OK — đang Restart",
                description=(
                    f"```\n{log}\n```\n"
                    "Bot sẽ **khởi động lại** trong vài giây.\n"
                    "Vị thế trên sàn **không** bị đóng."
                ),
                color=COLOR_OK,
            )
            await interaction.followup.send(embed=emb, ephemeral=True)
            logger.warning(f"Discord pull+restart by user {interaction.user.id}")

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content="Đã hủy pull.", embed=None, view=None
            )

    class ExcludeSymbolModal(discord.ui.Modal):
        """Modal nhập symbol để thêm vào blacklist (không quét)."""

        def __init__(self):
            super().__init__(title="Loại symbol khỏi scan")
            self.symbols_input = discord.ui.TextInput(
                label="Symbol (cách nhau bằng dấu cách)",
                placeholder="BTC DOGE PEPE  hoặc  BTC/USDT:USDT",
                required=True,
                max_length=200,
                style=discord.TextStyle.short,
            )
            self.add_item(self.symbols_input)

        async def on_submit(self, interaction: discord.Interaction):
            if not _user_authorized(interaction.user.id):
                await interaction.response.send_message("⛔ Không có quyền.", ephemeral=True)
                return
            raw = str(self.symbols_input.value or "").strip()
            parts = [p for p in raw.replace(",", " ").split() if p.strip()]
            if not parts:
                await interaction.response.send_message(
                    "❌ Chưa nhập symbol nào.", ephemeral=True
                )
                return
            current = list(_get_excluded_list())
            added = []
            for p in parts:
                n = _normalize_symbol(p)
                if n and n not in current:
                    current.append(n)
                    added.append(n)
            _persist_excluded(current)
            if added:
                msg = (
                    f"✅ Đã loại **{len(added)}** symbol khỏi scan:\n"
                    + "\n".join(f"• `{s}`" for s in added)
                    + f"\n\n{_format_excluded_msg()}"
                )
                color = COLOR_OK
            else:
                msg = f"ℹ️ Symbol đã có trong blacklist.\n\n{_format_excluded_msg()}"
                color = COLOR_INFO
            emb = discord.Embed(description=msg[:4000], color=color)
            await interaction.response.send_message(embed=emb, ephemeral=True)


    class StrategyEnableSelect(discord.ui.Select):
        """Chọn strategy đang tắt để BẬT (multi, max 25 option)."""

        def __init__(self):
            on = set(_get_enabled_strategies())
            all_n = _all_strategy_names()
            off = [n for n in all_n if n not in on]
            options = [
                discord.SelectOption(label=n[:100], value=n, description="Chạm để bật")
                for n in off[:25]
            ]
            if not options:
                options = [
                    discord.SelectOption(
                        label="(không còn strategy tắt)",
                        value="__none__",
                        description="Tất cả đã bật",
                    )
                ]
            super().__init__(
                placeholder="✅ Chọn strategy để BẬT…",
                min_values=1,
                max_values=min(len(options), 25),
                options=options,
                row=1,
            )

        async def callback(self, interaction: discord.Interaction):
            if not await _check_interaction(interaction):
                return
            vals = [v for v in self.values if v != "__none__"]
            if not vals:
                await interaction.response.send_message("Không có gì để bật.", ephemeral=True)
                return
            current = list(_get_enabled_strategies())
            cleaned = _persist_strategies(list(dict.fromkeys(current + vals)))
            await interaction.response.edit_message(
                content=_format_strategies_msg()[:1900],
                view=StrategyPanelView(),
            )
            await interaction.followup.send(
                f"✅ Đã bật: {', '.join(f'`{v}`' for v in vals)} · total **{len(cleaned)}**",
                ephemeral=True,
            )

    class StrategyDisableSelect(discord.ui.Select):
        """Chọn strategy đang bật để TẮT."""

        def __init__(self):
            on = list(_get_enabled_strategies())
            options = [
                discord.SelectOption(label=n[:100], value=n, description="Chạm để tắt")
                for n in on[:25]
            ]
            if not options:
                options = [
                    discord.SelectOption(
                        label="(không còn strategy bật)",
                        value="__none__",
                        description="Tất cả đã tắt",
                    )
                ]
            super().__init__(
                placeholder="⬜ Chọn strategy để TẮT…",
                min_values=1,
                max_values=min(len(options), 25),
                options=options,
                row=2,
            )

        async def callback(self, interaction: discord.Interaction):
            if not await _check_interaction(interaction):
                return
            vals = [v for v in self.values if v != "__none__"]
            if not vals:
                await interaction.response.send_message("Không có gì để tắt.", ephemeral=True)
                return
            remove = set(vals)
            current = [n for n in _get_enabled_strategies() if n not in remove]
            cleaned = _persist_strategies(current)
            await interaction.response.edit_message(
                content=_format_strategies_msg()[:1900],
                view=StrategyPanelView(),
            )
            await interaction.followup.send(
                f"⬜ Đã tắt: {', '.join(f'`{v}`' for v in vals)} · còn **{len(cleaned)}**",
                ephemeral=True,
            )

    class StrategyPanelView(discord.ui.View):
        """Panel strategy — tối đa 3 hàng: nút + 2 Select (mỗi Select = 1 hàng đầy đủ)."""

        def __init__(self):
            super().__init__(timeout=1800)
            self.add_item(StrategyEnableSelect())
            self.add_item(StrategyDisableSelect())

        @discord.ui.button(label="Làm mới", style=discord.ButtonStyle.primary, emoji="🔄", row=0)
        async def btn_refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content=_format_strategies_msg()[:1900],
                view=StrategyPanelView(),
            )

        @discord.ui.button(label="Bật hết", style=discord.ButtonStyle.success, emoji="✅", row=0)
        async def btn_all_on(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            cleaned = _persist_strategies(list(_all_strategy_names()))
            await interaction.response.edit_message(
                content=_format_strategies_msg()[:1900],
                view=StrategyPanelView(),
            )
            await interaction.followup.send(f"✅ Bật tất cả **{len(cleaned)}** strategy.", ephemeral=True)

        @discord.ui.button(label="Tắt hết", style=discord.ButtonStyle.secondary, emoji="⬜", row=0)
        async def btn_all_off(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            _persist_strategies([])
            await interaction.response.edit_message(
                content=_format_strategies_msg()[:1900],
                view=StrategyPanelView(),
            )
            await interaction.followup.send("⬜ Đã tắt hết strategy.", ephemeral=True)


    class ParamSelect(discord.ui.Select):
        def __init__(self):
            options = []
            for key in [
                "default_leverage",
                "max_leverage",
                "risk_per_trade_pct",
                "max_open_positions",
                "min_risk_reward",
                "max_daily_loss_pct",
                "top_n",
                "scan_interval_seconds",
            ]:
                label, hint = PARAM_LABELS.get(key, (key, ""))
                st = control_state
                cfg = (st.risk_gate.cfg if st.risk_gate else st.risk_cfg) or {}
                cur = cfg.get(key, st.settings.get(key, "?"))
                options.append(
                    discord.SelectOption(
                        label=label[:100],
                        description=f"Hiện: {cur} · {hint}"[:100],
                        value=key,
                    )
                )
            super().__init__(
                placeholder="✏️ Chỉnh thông số…",
                min_values=1,
                max_values=1,
                options=options,
                row=3,
            )

        async def callback(self, interaction: discord.Interaction):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_modal(SetParamModal(self.values[0]))

    class SetParamModal(discord.ui.Modal):
        def __init__(self, key: str):
            label, hint = PARAM_LABELS.get(key, (key, ""))
            super().__init__(title=f"Chỉnh: {label}"[:45])
            self.key = key
            st = control_state
            cfg = (st.risk_gate.cfg if st.risk_gate else st.risk_cfg) or {}
            current = cfg.get(key, st.settings.get(key, ""))
            self.value_input = discord.ui.TextInput(
                label=f"Giá trị ({hint})"[:45],
                placeholder=str(current),
                default=str(current) if current != "" else None,
                required=True,
                max_length=32,
            )
            self.add_item(self.value_input)

        async def on_submit(self, interaction: discord.Interaction):
            if not _user_authorized(interaction.user.id):
                await interaction.response.send_message("⛔ Không có quyền.", ephemeral=True)
                return
            msg = _apply_param(self.key, str(self.value_input.value).strip())
            emb = discord.Embed(
                description=msg,
                color=COLOR_OK if msg.startswith("✅") else COLOR_DANGER,
            )
            await interaction.response.send_message(embed=emb, ephemeral=True)

    class ControlPanelView(discord.ui.View):
        """Panel chính – timeout 30 phút."""

        def __init__(self):
            super().__init__(timeout=1800)
            self.add_item(ParamSelect())

        async def on_timeout(self):
            # Không edit được message ở đây dễ dàng không có ref; user gõ !panel lại
            pass

        async def _refresh_panel(self, interaction: discord.Interaction):
            await interaction.response.edit_message(
                content=None,
                embed=_panel_embed(),
                view=ControlPanelView(),
            )

        @discord.ui.button(label="Làm mới", style=discord.ButtonStyle.primary, emoji="🔄", row=0)
        async def btn_refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await self._refresh_panel(interaction)

        @discord.ui.button(label="Trạng thái", style=discord.ButtonStyle.secondary, emoji="📊", row=0)
        async def btn_status(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(embed=_status_embed(), ephemeral=True)

        @discord.ui.button(label="Vị thế", style=discord.ButtonStyle.secondary, emoji="📈", row=0)
        async def btn_positions(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(embed=_positions_embed(), ephemeral=True)

        @discord.ui.button(label="Thông số", style=discord.ButtonStyle.secondary, emoji="⚙️", row=0)
        async def btn_params(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(embed=_params_embed(), ephemeral=True)

        @discord.ui.button(label="Log", style=discord.ButtonStyle.secondary, emoji="📜", row=0)
        async def btn_log(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            ok, payload = _read_log_tail(40)
            if not ok:
                await interaction.response.send_message(payload, ephemeral=True)
                return
            if len(payload) <= 1900:
                await interaction.response.send_message(payload, ephemeral=True)
                return
            if "```" in payload:
                parts = payload.split("```")
                header = parts[0].strip()
                body = parts[1].strip("\n") if len(parts) > 1 else payload
            else:
                header = "📜 log"
                body = payload
            lines = body.splitlines()
            chunk, size, chunks = [], 0, []
            for line in lines:
                add = len(line) + 1
                if size + add > 1700 and chunk:
                    chunks.append("\n".join(chunk))
                    chunk, size = [line], add
                else:
                    chunk.append(line)
                    size += add
            if chunk:
                chunks.append("\n".join(chunk))
            await interaction.response.send_message(
                f"{header} · {len(lines)} dòng (nút Log ≈ 40 dòng)", ephemeral=True
            )
            for ch in chunks[:4]:
                await interaction.followup.send(f"```\n{ch}\n```", ephemeral=True)


        @discord.ui.button(label="Tạm dừng", style=discord.ButtonStyle.secondary, emoji="⏸", row=1)
        async def btn_pause(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.paused = True
            await interaction.response.edit_message(
                embed=_panel_embed(), view=ControlPanelView()
            )

        @discord.ui.button(label="Tiếp tục", style=discord.ButtonStyle.success, emoji="▶️", row=1)
        async def btn_resume(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.paused = False
            await interaction.response.edit_message(
                embed=_panel_embed(), view=ControlPanelView()
            )

        @discord.ui.button(label="Kill", style=discord.ButtonStyle.danger, emoji="🛑", row=1)
        async def btn_kill(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="⚠️ Xác nhận Kill",
                description=(
                    "Bật kill switch sẽ **flatten** vị thế và dừng bot.\n"
                    "Chỉ bấm xác nhận nếu bạn chắc chắn."
                ),
                color=COLOR_DANGER,
            )
            await interaction.response.send_message(
                embed=emb, view=ConfirmKillView(), ephemeral=True
            )

        @discord.ui.button(label="Gỡ Kill", style=discord.ButtonStyle.success, emoji="✅", row=1)
        async def btn_unkill(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            if control_state.risk_gate:
                control_state.risk_gate.reset_kill_switch()
            await interaction.response.edit_message(
                embed=_panel_embed(), view=ControlPanelView()
            )

        @discord.ui.button(label="Lev 3x", style=discord.ButtonStyle.secondary, row=2)
        async def btn_lev3(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            msg = _apply_param("default_leverage", "3")
            await interaction.response.send_message(msg, ephemeral=True)

        @discord.ui.button(label="Lev 5x", style=discord.ButtonStyle.secondary, row=2)
        async def btn_lev5(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            msg = _apply_param("default_leverage", "5")
            await interaction.response.send_message(msg, ephemeral=True)

        @discord.ui.button(label="Lev 10x", style=discord.ButtonStyle.secondary, row=2)
        async def btn_lev10(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            msg = _apply_param("default_leverage", "10")
            await interaction.response.send_message(msg, ephemeral=True)

        @discord.ui.button(
            label="Strategy", style=discord.ButtonStyle.primary, emoji="🎯", row=1
        )
        async def btn_strategy_panel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(
                content=_format_strategies_msg()[:1900],
                view=StrategyPanelView(),
                ephemeral=True,
            )

        # ---- Row 4: blacklist + restart ----
        
        @discord.ui.button(label="ML ON", style=discord.ButtonStyle.success, emoji="🧠", row=2)
        async def btn_ml_on(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            if _discord_ml is None:
                await interaction.response.send_message("discord_ml missing", ephemeral=True)
                return
            await interaction.response.send_message(_discord_ml.set_ml_filter(True), ephemeral=True)

        @discord.ui.button(label="ML OFF", style=discord.ButtonStyle.secondary, emoji="🧠", row=2)
        async def btn_ml_off(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            if _discord_ml is None:
                await interaction.response.send_message("discord_ml missing", ephemeral=True)
                return
            await interaction.response.send_message(_discord_ml.set_ml_filter(False), ephemeral=True)

        @discord.ui.button(
            label="Loại symbol", style=discord.ButtonStyle.secondary, emoji="🚫", row=4
        )
        async def btn_exclude(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_modal(ExcludeSymbolModal())

        @discord.ui.button(
            label="Xem blacklist", style=discord.ButtonStyle.secondary, emoji="📋", row=4
        )
        async def btn_exclude_list(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="Blacklist (không quét)",
                description=_format_excluded_msg(),
                color=COLOR_INFO,
            )
            await interaction.response.send_message(embed=emb, ephemeral=True)

        @discord.ui.button(
            label="Xóa blacklist", style=discord.ButtonStyle.secondary, emoji="♻️", row=4
        )
        async def btn_exclude_clear(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            _persist_excluded([])
            await interaction.response.send_message(
                "✅ Đã xóa toàn bộ blacklist — bot quét lại top_n.",
                ephemeral=True,
            )

        @discord.ui.button(
            label="Restart bot", style=discord.ButtonStyle.danger, emoji="🔁", row=4
        )
        async def btn_restart(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="⚠️ Xác nhận Restart",
                description=(
                    "Bot sẽ **tắt process và chạy lại** (load code/config mới).\n"
                    "• Vị thế trên sàn **không** bị đóng\n"
                    "• Discord offline vài giây rồi online lại\n"
                    "Chỉ dùng khi cần nạp lại code sau `git pull`."
                ),
                color=COLOR_WARN,
            )
            await interaction.response.send_message(
                embed=emb, view=ConfirmRestartView(), ephemeral=True
            )

        @discord.ui.button(
            label="Pull & Restart", style=discord.ButtonStyle.danger, emoji="📥", row=4
        )
        async def btn_pull_restart(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="⚠️ Xác nhận Pull & Restart",
                description=(
                    "Bot sẽ chạy **`git pull`** từ GitHub rồi **restart** process.\n"
                    "• Cần máy bot đã `git clone` repo và có mạng\n"
                    "• Vị thế trên sàn **không** bị đóng\n"
                    "• Nếu pull conflict → **không** restart, báo lỗi tại đây"
                ),
                color=COLOR_WARN,
            )
            await interaction.response.send_message(
                embed=emb, view=ConfirmPullRestartView(), ephemeral=True
            )


def create_bot() -> Optional[Any]:
    if not HAS_DISCORD:
        logger.warning("discord.py not installed – Discord control disabled")
        return None

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        logger.info("DISCORD_BOT_TOKEN not set – Discord control disabled")
        return None

    prefix = os.getenv("DISCORD_PREFIX", "!").strip() or "!"
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=prefix, intents=intents, help_command=None)

    @bot.event
    async def on_ready():
        logger.info(f"Discord control online as {bot.user} (prefix={prefix})")

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return
        logger.warning(f"Discord command error: {error}")
        try:
            await ctx.send(f"⚠️ Lỗi: `{error}`", delete_after=15)
        except Exception:
            pass

    @bot.command(name="panel", aliases=["menu", "ui"])
    async def cmd_panel(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Bạn không có quyền điều khiển bot.", delete_after=10)
            return
        await ctx.send(embed=_panel_embed(), view=ControlPanelView())

    @bot.command(name="help")
    async def cmd_help(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="Trợ giúp điều khiển",
            description=(
                f"**Khuyến nghị:** `{prefix}panel` — bảng nút + embed\n\n"
                f"`{prefix}panel` · `{prefix}menu` · `{prefix}ui`\n"
                f"`{prefix}status` · `{prefix}positions` · `{prefix}params`\n"
                f"`{prefix}set <key> <value>`\n"
                f"`{prefix}pause` · `{prefix}resume`\n"
                f"`{prefix}kill` · `{prefix}unkill`\n"
                f"`{prefix}restart` / `{prefix}pull` (có xác nhận)\n"
                f"`{prefix}skip clear` · `{prefix}exclude list|add|remove|clear`\n"
                f"`{prefix}log [n]` — n dòng cuối bot.log (mặc định 50)\n"
                f"`{prefix}strategy list|on|off <name...>`\n"
                f"`{prefix}ml` · `{prefix}ml on|off`\n\n"
                "**Panel:** 🚫 Loại symbol · 📋 Blacklist · 🔁 Restart · 📥 Pull & Restart"
            ),
            color=COLOR_INFO,
        )
        await ctx.send(embed=emb)




    @bot.command(name="ml")
    async def cmd_ml(ctx, action: str = None, sub: str = None):
        if not _is_authorized(ctx):
            await ctx.send("No permission.", delete_after=10)
            return
        if _discord_ml is None:
            await ctx.send("Module agents/discord_ml.py missing — git pull.")
            return
        act = (action or "status").strip().lower()
        if act in ("status", "st", "info"):
            await ctx.send(_discord_ml.status_text())
            return
        if act in ("on", "enable", "true", "1"):
            await ctx.send(_discord_ml.set_ml_filter(True))
            return
        if act in ("off", "disable", "false", "0"):
            await ctx.send(_discord_ml.set_ml_filter(False))
            return
        if act == "collect":
            sub_v = (sub or "").strip().lower()
            if sub_v in ("on", "true", "1"):
                await ctx.send(_discord_ml.set_ml_collect(True))
            elif sub_v in ("off", "false", "0"):
                await ctx.send(_discord_ml.set_ml_collect(False))
            else:
                await ctx.send(f"Use: `{ctx.prefix}ml collect on|off`")
            return
        if act in ("threshold", "thr", "prob"):
            if not sub:
                await ctx.send(f"Use: `{ctx.prefix}ml threshold 0.50`")
                return
            await ctx.send(_discord_ml.set_ml_threshold(sub))
            return
        await ctx.send(
            f"`{ctx.prefix}ml` | `{ctx.prefix}ml on|off` | "
            f"`{ctx.prefix}ml collect on|off` | `{ctx.prefix}ml threshold 0.5`"
        )


    @bot.command(name="log", aliases=["logs", "tail"])
    async def cmd_log(ctx, n: str = "50"):
        """Xem n dòng cuối logs/bot.log (mặc định 50, tối đa 200)."""
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        try:
            num = int(str(n).strip())
        except ValueError:
            await ctx.send(f"Dùng: `{prefix}log` hoặc `{prefix}log 80` (1–200)")
            return
        num = max(1, min(num, 200))
        ok, payload = _read_log_tail(num)
        if not ok:
            await ctx.send(payload)
            return
        if len(payload) <= 1900:
            await ctx.send(payload)
            return
        # Tách header / body code block
        if "```" in payload:
            parts = payload.split("```")
            header = parts[0].strip()
            body = parts[1].strip("\n") if len(parts) > 1 else payload
        else:
            header = "📜 log"
            body = payload
        lines = body.splitlines()
        chunk = []
        size = 0
        chunks = []
        for line in lines:
            add = len(line) + 1
            if size + add > 1700 and chunk:
                chunks.append("\n".join(chunk))
                chunk = [line]
                size = add
            else:
                chunk.append(line)
                size += add
        if chunk:
            chunks.append("\n".join(chunk))
        await ctx.send(f"{header} · {len(lines)} dòng · {len(chunks)} phần")
        for ch in chunks[:5]:
            await ctx.send(f"```\n{ch}\n```")
        if len(chunks) > 5:
            await ctx.send(f"… còn {len(chunks) - 5} phần — thử `{prefix}log 40`")


    @bot.command(name="status")
    async def cmd_status(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        await ctx.send(embed=_status_embed())

    @bot.command(name="positions")
    async def cmd_positions(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        await ctx.send(embed=_positions_embed())

    @bot.command(name="params")
    async def cmd_params(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        await ctx.send(embed=_params_embed())

    @bot.command(name="set")
    async def cmd_set(ctx, key: str = None, value: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        if not key or value is None:
            await ctx.send(f"Dùng: `{prefix}set <key> <value>` hoặc `{prefix}panel`")
            return
        await ctx.send(_apply_param(key, value))

    @bot.command(name="pause")
    async def cmd_pause(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        control_state.paused = True
        await ctx.send("⏸ Bot đã **pause**.")

    @bot.command(name="resume")
    async def cmd_resume(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        control_state.paused = False
        await ctx.send("▶️ Bot đã **resume**.")

    @bot.command(name="kill")
    async def cmd_kill(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="⚠️ Xác nhận Kill",
            description="Bật kill = flatten + dừng.",
            color=COLOR_DANGER,
        )
        await ctx.send(embed=emb, view=ConfirmKillView())

    @bot.command(name="unkill")
    async def cmd_unkill(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        if control_state.risk_gate:
            control_state.risk_gate.reset_kill_switch()
        await ctx.send("✅ Kill switch **OFF**.")

    @bot.command(name="restart")
    async def cmd_restart(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="⚠️ Xác nhận Restart",
            description=(
                "Bot sẽ tắt process và chạy lại.\n"
                "Vị thế trên sàn **không** bị đóng."
            ),
            color=COLOR_WARN,
        )
        await ctx.send(embed=emb, view=ConfirmRestartView())

    @bot.command(name="pull")
    async def cmd_pull(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="⚠️ Xác nhận Pull & Restart",
            description=(
                "Bot sẽ **`git pull`** từ GitHub rồi **restart**.\n"
                "Vị thế trên sàn **không** bị đóng.\n"
                "Conflict → không restart, báo lỗi."
            ),
            color=COLOR_WARN,
        )
        await ctx.send(embed=emb, view=ConfirmPullRestartView())

    @bot.command(name="skip")
    async def cmd_skip(ctx, action: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        st = control_state
        if action == "clear" and st.executor:
            n = len(st.executor.skip_symbols)
            st.executor.skip_symbols.clear()
            await ctx.send(f"Đã xóa {n} symbol khỏi skip list.")
        else:
            n = len(getattr(st.executor, "skip_symbols", []) or [])
            await ctx.send(
                f"Skip runtime: **{n}**. `{prefix}skip clear` | `{prefix}exclude list`"
            )

    
    @bot.command(name="strategy", aliases=["strategies", "strats", "strat"])
    async def cmd_strategy(ctx, action: str = None, *names: str):
        """Bật/tắt strategy: list | on|add | off|remove | all | none|clear"""
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        act = (action or "panel").strip().lower()
        all_names = set(_all_strategy_names())
        current = list(_get_enabled_strategies())

        if act in ("list", "ls", "status", "show", "panel", "ui", "menu"):
            msg = _format_strategies_msg()
            await ctx.send(msg[:1900], view=StrategyPanelView())
            return

        if act in ("all", "enable_all", "on_all"):
            cleaned = _persist_strategies(list(_all_strategy_names()))
            await ctx.send(f"✅ Bật **tất cả** ({len(cleaned)}):\n" + ", ".join(f"`{n}`" for n in cleaned))
            return

        if act in ("none", "clear", "off_all", "disable_all"):
            # Giữ ít nhất 1 nếu muốn — nhưng user hỏi clear thì clear
            _persist_strategies([])
            await ctx.send("⬜ Đã **tắt hết** strategy (scan sẽ không có signal strategy). Dùng `!strategy all` hoặc `!strategy on ORB` để bật lại.")
            return

        if act in ("on", "add", "enable", "bat"):
            if not names:
                await ctx.send(f"Dùng: `{prefix}strategy on ORB FVG TrendMA`")
                return
            resolved, unknown = [], []
            for raw in names:
                n = _normalize_strategy_name(raw)
                if n in all_names or n:
                    # accept known; warn unknown
                    if n not in all_names and n.lower() not in {x.lower() for x in all_names}:
                        unknown.append(raw)
                    else:
                        resolved.append(n if n in all_names else _normalize_strategy_name(raw))
            # re-resolve strictly
            resolved = []
            for raw in names:
                n = _normalize_strategy_name(raw)
                if n in all_names:
                    resolved.append(n)
                else:
                    unknown.append(raw)
            if not resolved and unknown:
                await ctx.send(
                    "Không nhận tên: " + ", ".join(f"`{u}`" for u in unknown)
                    + f"\nDanh sách: `{prefix}strategy list`"
                )
                return
            new_list = list(dict.fromkeys(current + resolved))
            cleaned = _persist_strategies(new_list)
            extra = f"\n⚠️ Bỏ qua: {', '.join(unknown)}" if unknown else ""
            await ctx.send(
                f"✅ Bật: {', '.join(f'`{n}`' for n in resolved)}\n"
                f"Đang active **{len(cleaned)}**: " + ", ".join(f"`{n}`" for n in cleaned) + extra
            )
            return

        if act in ("off", "remove", "disable", "rm", "tat"):
            if not names:
                await ctx.send(f"Dùng: `{prefix}strategy off ORB FVG`")
                return
            resolved, unknown = [], []
            for raw in names:
                n = _normalize_strategy_name(raw)
                if n in all_names or n in current:
                    resolved.append(n if n in current or n in all_names else raw)
                else:
                    unknown.append(raw)
            resolved = []
            for raw in names:
                n = _normalize_strategy_name(raw)
                if n in current or n in all_names:
                    resolved.append(n)
                else:
                    unknown.append(raw)
            remove_set = set(resolved)
            new_list = [n for n in current if n not in remove_set]
            cleaned = _persist_strategies(new_list)
            extra = f"\n⚠️ Không thấy: {', '.join(unknown)}" if unknown else ""
            await ctx.send(
                f"⬜ Tắt: {', '.join(f'`{n}`' for n in resolved)}\n"
                f"Còn **{len(cleaned)}**: " + (", ".join(f"`{n}`" for n in cleaned) or "_(rỗng)_") + extra
            )
            return

        await ctx.send(
            f"`{prefix}strategy list`\n"
            f"`{prefix}strategy on ORB FVG`\n"
            f"`{prefix}strategy off TrendMA`\n"
            f"`{prefix}strategy all` · `{prefix}strategy none`"
        )


    @bot.command(name="exclude", aliases=["blacklist", "ban"])
    async def cmd_exclude(ctx, action: str = None, *symbols: str):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        act = (action or "list").lower()
        current = list(_get_excluded_list())
        if act in ("list", "ls", "show"):
            await ctx.send(_format_excluded_msg())
            return
        if act == "clear":
            _persist_excluded([])
            await ctx.send("🧹 Đã xóa hết blacklist.")
            return
        if act in ("add", "+") or (action and act not in ("list", "remove", "clear", "add", "rm", "-")):
            syms = list(symbols) if act in ("add", "+") else ([action] + list(symbols or []))
            if act in ("add", "+") and not symbols:
                await ctx.send(f"Dùng: `{prefix}exclude add BTC DOGE`")
                return
            added = []
            for raw in syms:
                if not raw:
                    continue
                n = _normalize_symbol(raw)
                if n and n not in current:
                    current.append(n)
                    added.append(n)
            cleaned = _persist_excluded(current)
            msg = ", ".join(f"`{s}`" for s in added) if added else "đã có"
            await ctx.send(f"✅ Thêm: {msg} | tổng **{len(cleaned)}**")
            return
        if act in ("remove", "rm", "-"):
            if not symbols:
                await ctx.send(f"Dùng: `{prefix}exclude remove BTC`")
                return
            removed, targets, bases = [], set(), set()
            for raw in symbols:
                n = _normalize_symbol(raw)
                if n:
                    targets.add(n)
                    bases.add(n.split("/")[0])
                bases.add(str(raw).strip().upper())
            new_list = []
            for s in current:
                if s in targets or s.split("/")[0] in bases:
                    removed.append(s)
                else:
                    new_list.append(s)
            cleaned = _persist_excluded(new_list)
            msg = ", ".join(f"`{s}`" for s in removed) if removed else "—"
            await ctx.send(f"🗑️ Gỡ: {msg} | còn **{len(cleaned)}**")
            return
        await ctx.send(f"`{prefix}exclude list|add|remove|clear`")

    return bot


def start_discord_in_thread() -> Optional[threading.Thread]:
    if not HAS_DISCORD:
        return None
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        return None
    bot = create_bot()
    if bot is None:
        return None

    def _run():
        try:
            bot.run(token, log_handler=None)
        except Exception as e:
            logger.error(f"Discord bot crashed: {e}")

    t = threading.Thread(target=_run, name="discord-control", daemon=True)
    t.start()
    logger.info("Discord control thread started")
    return t
