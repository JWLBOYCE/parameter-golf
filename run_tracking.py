from __future__ import annotations

import json
import math
import os
import platform
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def env_flag(name: str, default: str = "0") -> bool:
    return bool(int(os.environ.get(name, default)))


def extract_config(args: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name in dir(args):
        if name.startswith("_"):
            continue
        value = getattr(args, name)
        if callable(value):
            continue
        config[name] = _jsonable(value)
    return config


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def host_info() -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cwd": str(Path.cwd()),
    }
    try:
        uname = os.uname()
        info["uname"] = {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        }
    except AttributeError:
        pass
    return info


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


class RunTracker:
    def __init__(
        self,
        *,
        run_id: str,
        trainer_name: str,
        backend: str,
        config: dict[str, Any],
        runs_dir: str = "artifacts/runs",
        best_dir: str = "artifacts/best",
        promote_max_bytes: int = 16_000_000,
        promote_metric: str = "final_int8_zlib_roundtrip_exact_val_bpb",
        retain_top_k: int = 1,
        keep_nonbest_artifacts: bool = False,
    ) -> None:
        self.run_id = run_id
        self.trainer_name = trainer_name
        self.backend = backend
        self.runs_dir = Path(runs_dir)
        self.best_dir = Path(best_dir)
        self.promote_max_bytes = int(promote_max_bytes)
        self.promote_metric = promote_metric
        self.retain_top_k = max(1, int(retain_top_k))
        self.keep_nonbest_artifacts = keep_nonbest_artifacts
        self.root_dir = self._derive_root_dir()
        self.run_dir = self.runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.log"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        self.promotion_history_path = self.root_dir / "promotion_history.jsonl"
        _write_json(
            self.config_path,
            {
                "run_id": run_id,
                "trainer_name": trainer_name,
                "backend": backend,
                "git_sha": git_sha(),
                "started_at": _now_iso(),
                "host": host_info(),
                "config": config,
            },
        )

    def _derive_root_dir(self) -> Path:
        runs_root = self.runs_dir.parent
        best_root = self.best_dir.parent
        return runs_root if runs_root == best_root else Path(os.path.commonpath([runs_root, best_root]))

    def log(
        self,
        message: str,
        *,
        console: bool = True,
        event_type: str = "log",
        step: int | None = None,
        train_time_ms: float | None = None,
        **payload: Any,
    ) -> None:
        if console:
            print(message)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        self.event(
            event_type=event_type,
            message=message,
            step=step,
            train_time_ms=train_time_ms,
            **payload,
        )

    def event(
        self,
        *,
        event_type: str,
        step: int | None = None,
        train_time_ms: float | None = None,
        **payload: Any,
    ) -> None:
        _append_jsonl(
            self.events_path,
            {
                "timestamp": _now_iso(),
                "run_id": self.run_id,
                "backend": self.backend,
                "event_type": event_type,
                "step": step,
                "train_time_ms": train_time_ms,
                **payload,
            },
        )

    def finalize(
        self,
        *,
        summary: dict[str, Any],
        artifact_paths: dict[str, Path],
    ) -> dict[str, Any]:
        enriched = dict(summary)
        enriched.update(
            {
                "run_id": self.run_id,
                "backend": self.backend,
                "trainer_name": self.trainer_name,
                "run_dir": str(self.run_dir),
                "log_path": str(self.log_path),
                "summary_path": str(self.summary_path),
            }
        )
        promoted, reason = self._should_promote(enriched)
        enriched["promoted"] = promoted
        enriched["promotion_reason"] = reason
        if promoted:
            self._promote(enriched, artifact_paths)
        retained = self._ranked_summaries(current_summary=enriched)[: self.retain_top_k]
        retained_ids = {summary["run_id"] for summary in retained}
        enriched["retained_top_k"] = self.retain_top_k
        enriched["retained_top_rank"] = next(
            (rank for rank, retained_summary in enumerate(retained, start=1) if retained_summary["run_id"] == self.run_id),
            None,
        )
        pruned_paths = [] if self.keep_nonbest_artifacts else self._enforce_retention(retained_ids, current_summary=enriched)
        enriched["artifacts_pruned"] = pruned_paths
        _write_json(self.summary_path, enriched)
        self.event(event_type="summary", summary=enriched)
        return enriched

    def _metric_tuple(self, summary: dict[str, Any]) -> tuple[float, int, float]:
        metric = float(summary.get(self.promote_metric, math.inf))
        bytes_total = int(summary.get("bytes_total", 1 << 62))
        loss = float(summary.get("final_int8_zlib_roundtrip_exact_val_loss", math.inf))
        return metric, bytes_total, loss

    def _load_best(self) -> dict[str, Any] | None:
        path = self.best_dir / "best.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_run_summaries(self, *, current_summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for path in self.runs_dir.glob("*/summary.json"):
            summaries[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
        if current_summary is not None:
            summaries[current_summary["run_id"]] = dict(current_summary)
        return list(summaries.values())

    def _eligible_for_retention(self, summary: dict[str, Any]) -> bool:
        bytes_total = int(summary.get("bytes_total", 1 << 62))
        metric = float(summary.get(self.promote_metric, math.inf))
        return bytes_total <= self.promote_max_bytes and math.isfinite(metric)

    def _ranked_summaries(self, *, current_summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        eligible = [summary for summary in self._load_run_summaries(current_summary=current_summary) if self._eligible_for_retention(summary)]
        return sorted(eligible, key=self._metric_tuple)

    def _should_promote(self, summary: dict[str, Any]) -> tuple[bool, str]:
        bytes_total = int(summary.get("bytes_total", 1 << 62))
        if bytes_total > self.promote_max_bytes:
            return False, f"bytes_total_exceeds_limit:{bytes_total}>{self.promote_max_bytes}"
        current = self._load_best()
        if current is None:
            return True, "no_existing_best"
        if self._metric_tuple(summary) < self._metric_tuple(current):
            return True, "better_exact_metric"
        return False, "score_not_better"

    def _promote(self, summary: dict[str, Any], artifact_paths: dict[str, Path]) -> None:
        self.best_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_best = Path(tempfile.mkdtemp(prefix=f".best-{self.run_id}-", dir=self.best_dir.parent))
        raw_path = artifact_paths["raw_model"]
        quant_path = artifact_paths["quantized_model"]
        raw_name = f"model.raw{''.join(raw_path.suffixes) or raw_path.suffix}"
        shutil.copy2(self.log_path, tmp_best / "train.log")
        shutil.copy2(raw_path, tmp_best / raw_name)
        shutil.copy2(quant_path, tmp_best / "model.int8.ptz")
        best_payload = dict(summary)
        best_payload["best_paths"] = {
            "train_log": str(self.best_dir / "train.log"),
            "raw_model": str(self.best_dir / raw_name),
            "quantized_model": str(self.best_dir / "model.int8.ptz"),
        }
        _write_json(tmp_best / "best.json", best_payload)

        backup_dir: Path | None = None
        if self.best_dir.exists():
            backup_dir = self.best_dir.parent / f".best-old-{self.run_id}"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            self.best_dir.rename(backup_dir)
        tmp_best.rename(self.best_dir)
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
        _append_jsonl(
            self.promotion_history_path,
            {
                "timestamp": _now_iso(),
                "event_type": "promote_best",
                "run_id": self.run_id,
                "metric": summary.get(self.promote_metric),
                "bytes_total": summary.get("bytes_total"),
            },
        )

    def _artifact_paths_from_summary(self, summary: dict[str, Any]) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for key, summary_key in (("raw_model", "raw_model_path"), ("quantized_model", "quantized_model_path")):
            value = summary.get(summary_key)
            if value:
                paths[key] = Path(value)
        return paths

    def _enforce_retention(
        self,
        retained_ids: set[str],
        *,
        current_summary: dict[str, Any] | None = None,
    ) -> list[str]:
        pruned: list[str] = []
        for summary in self._load_run_summaries(current_summary=current_summary):
            run_id = str(summary.get("run_id", ""))
            if run_id in retained_ids:
                continue
            pruned.extend(self._prune_artifacts(self._artifact_paths_from_summary(summary), run_id=run_id))
        return pruned

    def _prune_artifacts(self, artifact_paths: dict[str, Path], *, run_id: str | None = None) -> list[str]:
        pruned: list[str] = []
        for path in artifact_paths.values():
            if not isinstance(path, Path):
                path = Path(path)
            if path.is_file():
                path.unlink()
                pruned.append(str(path))
        if pruned:
            _append_jsonl(
                self.promotion_history_path,
                {
                    "timestamp": _now_iso(),
                    "event_type": "prune_artifacts",
                    "run_id": run_id or self.run_id,
                    "paths": pruned,
                },
            )
        return pruned
