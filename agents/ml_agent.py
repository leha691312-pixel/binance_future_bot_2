"""ML Agent - collect samples, train filter (global or per style: scalp/day/swing)."""
from typing import List, Optional, Dict, Any
import json
import math
from pathlib import Path
from datetime import datetime
from loguru import logger
from core.models import Signal, Side

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    np = None

FEATURE_DIM = 18
STYLES = ("scalp", "day", "swing")


class MLAgent:
    def __init__(
        self,
        model_path: str = "data/models/signal_filter.joblib",
        samples_path: str = "data/models/train_samples.jsonl",
        min_samples: int = 80,
        retrain_every_new: int = 25,
        min_retrain_hours: float = 12.0,
        collect_only: bool = False,
        filter_enabled: bool = True,
        min_auc_deploy: float = 0.55,
        min_class_count: int = 10,
        fee_rate: float = 0.0004,
        min_r_multiple: float = 0.25,
        prob_threshold: float = 0.55,
        test_ratio: float = 0.25,
        per_style: bool = False,
        min_samples_per_style: int = 40,
        style_map: Optional[Dict[str, str]] = None,
        prob_threshold_by_style: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = self.model_path.with_suffix(".scaler.joblib")
        self.metrics_path = self.model_path.with_suffix(".metrics.json")
        self.samples_path = Path(samples_path)
        self.min_samples = int(min_samples)
        self.retrain_every_new = int(retrain_every_new)
        self.min_retrain_hours = float(min_retrain_hours)
        self.collect_only = bool(collect_only)
        self.filter_enabled = bool(filter_enabled) and not self.collect_only
        self.min_auc_deploy = float(min_auc_deploy)
        self.min_class_count = int(min_class_count)
        self.fee_rate = float(fee_rate)
        self.min_r_multiple = float(min_r_multiple)
        self.prob_threshold = float(prob_threshold)
        self.test_ratio = float(test_ratio)
        self.per_style = bool(per_style)
        self.min_samples_per_style = int(min_samples_per_style)
        self.style_map = {str(k): str(v).lower() for k, v in (style_map or {}).items()}
        self.prob_threshold_by_style = {
            str(k).lower(): float(v) for k, v in (prob_threshold_by_style or {}).items()
        }

        self.model = None
        self.scaler = None
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.style_metrics: Dict[str, dict] = {}
        self.enabled = HAS_ML
        self.last_metrics: Dict[str, Any] = {}
        self.pending_features: Dict[str, List[float]] = {}
        self.pending_meta: Dict[str, dict] = {}
        self._samples_since_train = 0
        self._samples_since_train_style: Dict[str, int] = {s: 0 for s in STYLES}
        self._last_train_time: Optional[datetime] = None
        self._last_train_time_style: Dict[str, Optional[datetime]] = {s: None for s in STYLES}

        self._load_pending()
        self._load_meta()
        self._load_models()

        if self.enabled and not self.collect_only:
            if self.per_style:
                for st in STYLES:
                    if st not in self.models:
                        n = self.count_samples(style=st)
                        if n >= self.min_samples_per_style:
                            logger.info(f"ML startup train [{st}]: {n} samples")
                            self.maybe_retrain(force=True, style=st)
            elif self.model is None:
                n = self.count_samples()
                if n >= self.min_samples:
                    logger.info(f"ML startup train: {n} samples >= {self.min_samples}")
                    self.maybe_retrain(force=True)

    # ----- paths -----
    def _style_samples_path(self, style: str) -> Path:
        # train_samples_scalp.jsonl next to train_samples.jsonl
        p = self.samples_path
        return p.with_name(f"{p.stem}_{style}{p.suffix}")

    def _style_model_path(self, style: str) -> Path:
        p = self.model_path
        return p.with_name(f"{p.stem}_{style}{p.suffix}")

    def _style_scaler_path(self, style: str) -> Path:
        return self._style_model_path(style).with_suffix(".scaler.joblib")

    def _style_metrics_path(self, style: str) -> Path:
        return self._style_model_path(style).with_suffix(".metrics.json")

    def _pending_path(self) -> Path:
        return self.samples_path.with_suffix(".pending.json")

    def _pending_meta_path(self) -> Path:
        return self.samples_path.with_suffix(".pending_meta.json")

    def _meta_path(self) -> Path:
        return self.samples_path.with_suffix(".meta.json")

    def _load_pending(self):
        try:
            p = self._pending_path()
            if p.exists():
                self.pending_features = json.loads(p.read_text(encoding="utf-8"))
            mp = self._pending_meta_path()
            if mp.exists():
                self.pending_meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _save_pending(self):
        try:
            self.samples_path.parent.mkdir(parents=True, exist_ok=True)
            self._pending_path().write_text(json.dumps(self.pending_features), encoding="utf-8")
            self._pending_meta_path().write_text(json.dumps(self.pending_meta), encoding="utf-8")
        except Exception:
            pass

    def _load_meta(self):
        try:
            mp = self._meta_path()
            if mp.exists():
                data = json.loads(mp.read_text(encoding="utf-8"))
                self._samples_since_train = int(data.get("samples_since_train", 0))
                ts = data.get("last_train_time")
                if ts:
                    self._last_train_time = datetime.fromisoformat(ts)
                sst = data.get("samples_since_train_style") or {}
                for s in STYLES:
                    self._samples_since_train_style[s] = int(sst.get(s, 0))
                lts = data.get("last_train_time_style") or {}
                for s in STYLES:
                    if lts.get(s):
                        try:
                            self._last_train_time_style[s] = datetime.fromisoformat(lts[s])
                        except Exception:
                            pass
        except Exception:
            pass

    def _save_meta(self):
        try:
            self.samples_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "samples_since_train": self._samples_since_train,
                "last_train_time": self._last_train_time.isoformat() if self._last_train_time else None,
                "samples_since_train_style": self._samples_since_train_style,
                "last_train_time_style": {
                    s: (t.isoformat() if t else None) for s, t in self._last_train_time_style.items()
                },
            }
            self._meta_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_models(self):
        if not self.enabled:
            return
        # global
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                if self.scaler_path.exists():
                    self.scaler = joblib.load(self.scaler_path)
                if self.metrics_path.exists():
                    self.last_metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                logger.info(f"ML global model loaded | metrics={self.last_metrics}")
            except Exception as e:
                logger.warning(f"ML global model load failed: {e}")
        # per-style
        for st in STYLES:
            mp = self._style_model_path(st)
            if not mp.exists():
                continue
            try:
                self.models[st] = joblib.load(mp)
                sp = self._style_scaler_path(st)
                if sp.exists():
                    self.scalers[st] = joblib.load(sp)
                metap = self._style_metrics_path(st)
                if metap.exists():
                    self.style_metrics[st] = json.loads(metap.read_text(encoding="utf-8"))
                logger.info(f"ML model [{st}] loaded | auc={self.style_metrics.get(st, {}).get('auc')}")
            except Exception as e:
                logger.warning(f"ML model [{st}] load failed: {e}")
        if not self.model and not self.models:
            logger.info("ML Agent: no model yet (collecting / pass-through)")

    # ----- style resolve -----
    def resolve_style(self, strategy: str = "", timeframe: str = "", explicit: str = "") -> str:
        if explicit:
            s = str(explicit).lower().strip()
            if s in STYLES:
                return s
        strat = (strategy or "").strip()
        if strat in self.style_map:
            st = self.style_map[strat]
            if st in STYLES:
                return st
        # case-insensitive map
        low = {k.lower(): v for k, v in self.style_map.items()}
        if strat.lower() in low and low[strat.lower()] in STYLES:
            return low[strat.lower()]
        tf = (timeframe or "").lower().strip()
        if tf in ("1m", "3m", "5m"):
            return "scalp"
        if tf in ("15m", "30m", "1h"):
            return "day"
        if tf in ("2h", "4h", "6h", "12h", "1d"):
            return "swing"
        return "day"

    def _threshold_for(self, style: str) -> float:
        return float(self.prob_threshold_by_style.get(style, self.prob_threshold))

    # ----- features -----
    def extract_features(self, signal: Signal, df=None, context: Optional[dict] = None) -> List[float]:
        context = context or {}
        conf = float(getattr(signal, "confidence", 0.5) or 0.5)
        try:
            rr = float(signal.risk_reward()) if hasattr(signal, "risk_reward") else 1.5
        except Exception:
            rr = 1.5
        is_long = 1.0 if getattr(signal, "side", None) == Side.LONG or str(getattr(signal, "side", "")).upper() in ("LONG", "BUY") else 0.0
        hour = datetime.utcnow().hour
        body_ratio = 0.5
        atr_pct = 0.01
        atr_pctile = 0.5
        volume_ratio = 1.0
        rsi_n = 0.5
        dist_from_high = 0.0
        dist_from_low = 0.0
        trend_strength = 0.0
        try:
            entry = float(signal.entry)
            sl = float(signal.sl)
            tp = float(signal.tp)
            sl_dist = abs(entry - sl) / entry if entry else 0.01
            tp_dist = abs(tp - entry) / entry if entry else 0.02
        except Exception:
            sl_dist, tp_dist = 0.01, 0.02

        if df is not None and hasattr(df, "__len__") and len(df) >= 30:
            try:
                last = df.iloc[-1]
                window = df.iloc[-20:]
                body = abs(float(last["close"]) - float(last["open"]))
                full = float(last["high"]) - float(last["low"])
                body_ratio = float(body / full) if full > 0 else 0.5
                closes = df["close"].astype(float)
                highs = df["high"].astype(float)
                lows = df["low"].astype(float)
                tr = (highs - lows).iloc[-14:]
                atr = float(tr.mean()) if len(tr) else 0.0
                atr_pct = atr / float(last["close"]) if float(last["close"]) > 0 else 0.01
                vol = df["volume"].astype(float).iloc[-20:]
                volume_ratio = float(vol.iloc[-1] / vol.mean()) if vol.mean() > 0 else 1.0
                delta = closes.diff()
                gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                if loss and loss > 0:
                    rsi_n = float(100 - (100 / (1 + gain / loss))) / 100.0
                high_20 = float(window["high"].max())
                low_20 = float(window["low"].min())
                dist_from_high = (high_20 - float(last["close"])) / float(last["close"])
                dist_from_low = (float(last["close"]) - low_20) / float(last["close"])
                sma_f = float(closes.iloc[-8:].mean())
                sma_s = float(closes.iloc[-20:].mean())
                trend_strength = (sma_f - sma_s) / float(last["close"])
                # ATR percentile proxy
                atr_series = (highs - lows).rolling(14).mean().dropna()
                if len(atr_series) >= 20:
                    atr_pctile = float((atr_series.iloc[-20:] < atr).mean()) if atr else 0.5
            except Exception:
                pass
        feats = [
            conf, rr, is_long, 0.5,
            body_ratio, atr_pct, atr_pctile, volume_ratio, rsi_n,
            dist_from_high, dist_from_low, trend_strength,
            sl_dist, tp_dist,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            float(context.get("funding_rate") or 0),
            float(context.get("htf_trend") or 0),
        ]
        while len(feats) < FEATURE_DIM:
            feats.append(0.0)
        return feats[:FEATURE_DIM]

    def on_open(self, signal: Signal, df=None, key: str = "", context=None, trade_meta=None):
        if not key:
            key = f"{signal.symbol}_{signal.side.value}"
        try:
            feats = self.extract_features(signal, df, context=context)
            style = self.resolve_style(
                getattr(signal, "strategy", "") or "",
                getattr(signal, "timeframe", "") or "",
                (trade_meta or {}).get("trade_style", ""),
            )
            self.pending_features[key] = feats
            self.pending_meta[key] = {
                "entry": float(signal.entry),
                "sl": float(signal.sl),
                "tp": float(signal.tp),
                "strategy": getattr(signal, "strategy", ""),
                "timeframe": getattr(signal, "timeframe", ""),
                "trade_style": style,
                **(trade_meta or {}),
            }
            self.pending_meta[key]["trade_style"] = style
            self._save_pending()
            logger.info(
                f"ML on_open {key} style={style} feats_dim={len(feats)} | pending={len(self.pending_features)}"
            )
        except Exception as e:
            logger.warning(f"ML on_open failed {key}: {e}")

    def on_close(self, key: str, pnl: float, meta: Optional[dict] = None) -> Optional[dict]:
        feats = self.pending_features.pop(key, None)
        pmeta = self.pending_meta.pop(key, {}) or {}
        if feats is None and key:
            for ak in (
                key.replace("_LONG", "_long"),
                key.replace("_SHORT", "_short"),
                key.replace("_long", "_LONG"),
                key.replace("_short", "_SHORT"),
            ):
                if ak in self.pending_features:
                    feats = self.pending_features.pop(ak)
                    pmeta = self.pending_meta.pop(ak, {}) or {}
                    key = ak
                    break
        self._save_pending()
        if feats is None:
            logger.warning(
                f"ML on_close: no pending features for {key} | "
                f"pending_keys={list(self.pending_features.keys())[:8]}"
            )
            return None

        m = {**pmeta, **(meta or {})}
        risk_amount = float(m.get("risk_amount") or 0)
        notional = float(m.get("notional") or 0)
        entry = float(m.get("entry") or 0)
        size = float(m.get("size") or 0)
        if risk_amount <= 0 and entry and size and m.get("sl") is not None:
            risk_amount = abs(entry - float(m["sl"])) * abs(size)

        fee_est = abs(notional) * self.fee_rate * 2
        r_mult = (pnl / risk_amount) if risk_amount > 0 else None
        win = pnl > fee_est
        if r_mult is not None and self.min_r_multiple > 0:
            win = win and (r_mult >= self.min_r_multiple)

        style = self.resolve_style(
            m.get("strategy", ""),
            m.get("timeframe", ""),
            m.get("trade_style", ""),
        )
        label = 1 if win else 0
        rec = {
            "features": list(feats)[:FEATURE_DIM],
            "label": label,
            "pnl": float(pnl),
            "r_multiple": float(r_mult) if r_mult is not None else None,
            "style": style,
            "strategy": m.get("strategy", ""),
            "timeframe": m.get("timeframe", ""),
            "key": key,
            "ts": datetime.utcnow().isoformat(),
        }

        # always append global jsonl
        self._append_sample(self.samples_path, rec)
        # per-style file
        if self.per_style:
            self._append_sample(self._style_samples_path(style), rec)
            self._samples_since_train_style[style] = self._samples_since_train_style.get(style, 0) + 1

        self._samples_since_train += 1
        self._save_meta()
        logger.info(
            f"ML sample | {key} style={style} label={'WIN' if label else 'LOSS'} "
            f"pnl={pnl:.4f} R={r_mult if r_mult is not None else 'n/a'} | "
            f"n_global={self.count_samples()} n_{style}={self.count_samples(style=style)}"
        )

        if self.collect_only:
            return rec

        if self.per_style:
            self.maybe_retrain(style=style)
        else:
            self.maybe_retrain()
        return rec

    def _append_sample(self, path: Path, rec: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def count_samples(self, style: Optional[str] = None) -> int:
        path = self._style_samples_path(style) if style else self.samples_path
        if not path.exists():
            return 0
        n = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
        except Exception:
            pass
        return n

    def load_all_samples(self, style: Optional[str] = None) -> List[dict]:
        path = self._style_samples_path(style) if style else self.samples_path
        records = []
        if not path.exists():
            return records
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "features" in rec and "label" in rec:
                        feats = list(rec["features"])
                        if len(feats) < FEATURE_DIM:
                            feats = feats + [0.0] * (FEATURE_DIM - len(feats))
                        rec["features"] = feats[:FEATURE_DIM]
                        records.append(rec)
                except Exception:
                    continue
        return records

    def maybe_retrain(self, force: bool = False, style: Optional[str] = None) -> Optional[dict]:
        if not HAS_ML:
            logger.warning("ML retrain skip: sklearn not installed")
            return None

        if self.per_style and style:
            return self._maybe_retrain_style(style, force=force)
        if self.per_style and style is None:
            # retrain all styles that need it
            reports = {}
            for st in STYLES:
                r = self._maybe_retrain_style(st, force=force)
                if r:
                    reports[st] = r
            return reports or None

        n = self.count_samples()
        if n < self.min_samples:
            logger.debug(f"ML retrain skip: {n}/{self.min_samples}")
            return None

        if not force:
            if self.model is None:
                force = True
            else:
                if self._samples_since_train < self.retrain_every_new:
                    return None
                if self._last_train_time is not None:
                    hours = (datetime.utcnow() - self._last_train_time).total_seconds() / 3600
                    if hours < self.min_retrain_hours and self._samples_since_train < self.retrain_every_new * 2:
                        return None

        records = self.load_all_samples()
        report = self.train_from_trades(records, min_samples=self.min_samples, style=None)
        if report.get("deployed"):
            self._samples_since_train = 0
            self._last_train_time = datetime.utcnow()
            self._save_meta()
            logger.info(f"ML deploy OK: {report}")
        else:
            logger.warning(f"ML train not deployed: {report}")
        return report

    def _maybe_retrain_style(self, style: str, force: bool = False) -> Optional[dict]:
        style = style if style in STYLES else "day"
        n = self.count_samples(style=style)
        if n < self.min_samples_per_style:
            logger.debug(f"ML retrain skip [{style}]: {n}/{self.min_samples_per_style}")
            return None
        if not force:
            has = style in self.models
            if not has:
                force = True
            else:
                if self._samples_since_train_style.get(style, 0) < self.retrain_every_new:
                    return None
                lt = self._last_train_time_style.get(style)
                if lt is not None:
                    hours = (datetime.utcnow() - lt).total_seconds() / 3600
                    if hours < self.min_retrain_hours and self._samples_since_train_style.get(style, 0) < self.retrain_every_new * 2:
                        return None

        records = self.load_all_samples(style=style)
        report = self.train_from_trades(records, min_samples=self.min_samples_per_style, style=style)
        if report.get("deployed"):
            self._samples_since_train_style[style] = 0
            self._last_train_time_style[style] = datetime.utcnow()
            self._save_meta()
            logger.info(f"ML deploy OK [{style}]: {report}")
        else:
            logger.warning(f"ML train not deployed [{style}]: {report}")
        return report

    def train_from_trades(
        self, trade_records: List[dict], min_samples: int = 50, style: Optional[str] = None
    ) -> dict:
        if not HAS_ML:
            return {"error": "sklearn not installed", "deployed": False, "style": style}

        if len(trade_records) < min_samples:
            return {
                "error": f"Need at least {min_samples} samples, got {len(trade_records)}",
                "deployed": False,
                "style": style,
            }

        # sort by time for time-based split
        def _ts(r):
            t = r.get("ts") or r.get("time") or ""
            return str(t)

        records = sorted(trade_records, key=_ts)
        labels = [int(r["label"]) for r in records]
        n_pos = sum(1 for x in labels if x == 1)
        n_neg = sum(1 for x in labels if x == 0)
        if n_pos < self.min_class_count or n_neg < self.min_class_count:
            return {
                "error": f"Need >= {self.min_class_count} per class (win={n_pos} loss={n_neg})",
                "deployed": False,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "style": style,
            }

        n = len(records)
        split = max(1, int(n * (1 - self.test_ratio)))
        if split >= n:
            split = n - 1
        train_recs = records[:split]
        test_recs = records[split:]
        if len(test_recs) < 5:
            test_recs = records[-max(5, n // 5) :]
            train_recs = records[: max(1, n - len(test_recs))]

        X_train = np.array([r["features"] for r in train_recs], dtype=float)
        y_train = np.array([int(r["label"]) for r in train_recs], dtype=int)
        X_test = np.array([r["features"] for r in test_recs], dtype=float)
        y_test = np.array([int(r["label"]) for r in test_recs], dtype=int)

        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=100,
                learning_rate=0.08,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=42,
            )
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)[:, 1]
            if len(set(y_test.tolist())) > 1:
                auc = float(roc_auc_score(y_test, proba))
            else:
                auc = 0.5

            report = {
                "style": style or "global",
                "n": n,
                "n_train": len(train_recs),
                "n_test": len(test_recs),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc": round(auc, 4),
                "deployed": False,
            }

            if auc < self.min_auc_deploy:
                report["deploy_reason"] = f"AUC {auc:.3f} < min {self.min_auc_deploy}"
                if style:
                    self.style_metrics[style] = report
                else:
                    self.last_metrics = report
                return report

            if style:
                mp = self._style_model_path(style)
                sp = self._style_scaler_path(style)
                metap = self._style_metrics_path(style)
                mp.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, mp)
                joblib.dump(scaler, sp)
                self.models[style] = model
                self.scalers[style] = scaler
                report["deployed"] = True
                report["deploy_reason"] = "ok"
                self.style_metrics[style] = report
                metap.write_text(json.dumps(report, indent=2), encoding="utf-8")
                logger.info(f"ML model DEPLOYED [{style}]: AUC={auc:.3f} n={n} pos={n_pos} neg={n_neg}")
            else:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, self.model_path)
                joblib.dump(scaler, self.scaler_path)
                self.model = model
                self.scaler = scaler
                report["deployed"] = True
                report["deploy_reason"] = "ok"
                self.last_metrics = report
                self.metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                logger.info(f"ML model DEPLOYED: AUC={auc:.3f} n={n} pos={n_pos} neg={n_neg}")
            return report
        except Exception as e:
            logger.error(f"ML train failed [{style or 'global'}]: {e}")
            return {"error": str(e), "deployed": False, "style": style}

    def filter(self, signals: List[Signal], dfs=None) -> List[Signal]:
        if self.collect_only or not self.filter_enabled:
            return list(signals)

        out = []
        for s in signals:
            try:
                style = self.resolve_style(
                    getattr(s, "strategy", "") or "",
                    getattr(s, "timeframe", "") or "",
                )
                thr = self._threshold_for(style)

                if self.per_style:
                    model = self.models.get(style)
                    scaler = self.scalers.get(style)
                    # fallback: global model if style model missing
                    if model is None:
                        model = self.model
                        scaler = self.scaler
                    if model is None:
                        out.append(s)  # no model for this style → pass-through
                        continue
                else:
                    if self.model is None:
                        return list(signals)
                    model = self.model
                    scaler = self.scaler

                feats = self.extract_features(s)
                X = [feats]
                if scaler is not None:
                    X = scaler.transform(X)
                proba = float(model.predict_proba(X)[0][1])
                if proba >= thr:
                    s.confidence = max(float(getattr(s, "confidence", 0) or 0), proba)
                    out.append(s)
            except Exception:
                out.append(s)
        out.sort(key=lambda x: getattr(x, "confidence", 0), reverse=True)
        return out

    def status(self) -> dict:
        st = {
            "has_model": self.model is not None or bool(self.models),
            "per_style": self.per_style,
            "collect_only": self.collect_only,
            "filter_enabled": self.filter_enabled,
            "n_samples": self.count_samples(),
            "pending_open": len(self.pending_features),
            "samples_since_train": self._samples_since_train,
            "last_train": self._last_train_time.isoformat() if self._last_train_time else None,
            "min_samples": self.min_samples,
            "prob_threshold": self.prob_threshold,
            "last_metrics": self.last_metrics,
        }
        if self.per_style:
            st["styles"] = {
                s: {
                    "n_samples": self.count_samples(style=s),
                    "has_model": s in self.models,
                    "samples_since_train": self._samples_since_train_style.get(s, 0),
                    "last_train": (
                        self._last_train_time_style[s].isoformat()
                        if self._last_train_time_style.get(s)
                        else None
                    ),
                    "threshold": self._threshold_for(s),
                    "metrics": self.style_metrics.get(s, {}),
                }
                for s in STYLES
            }
        return st
