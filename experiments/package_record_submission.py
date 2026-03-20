from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from experiments.snapshot_record_candidate import ROOT, WORKBENCH, snapshot_candidate
except ImportError:  # pragma: no cover - direct script execution from experiments/
    from snapshot_record_candidate import ROOT, WORKBENCH, snapshot_candidate


@dataclass
class ParsedMetrics:
    variant: str | None
    layout: str | None
    train_cfg: str | None
    optimizer_cfg: str | None
    quant_cfg: str | None
    final_val_loss: float | None
    final_val_bpb: float | None
    sliding_val_loss: float | None
    sliding_val_bpb: float | None
    sliding_stride: int | None
    bytes_total: int | None
    bytes_code: int | None
    bytes_quantized: int | None
    bytes_payload: int | None


def _last_match(text: str, pattern: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return matches[-1] if matches else None


def parse_train_log_metrics(log_path: Path) -> ParsedMetrics:
    text = log_path.read_text(encoding="utf-8")
    final_match = _last_match(
        text,
        r"final_(?:roundtrip_exact|int8_zlib_roundtrip_exact) val_loss:(?P<loss>[-+0-9.eE]+) val_bpb:(?P<bpb>[-+0-9.eE]+)",
    )
    sliding_match = _last_match(
        text,
        r"final_sliding_window_exact val_loss:(?P<loss>[-+0-9.eE]+) val_bpb:(?P<bpb>[-+0-9.eE]+) stride:(?P<stride>\d+)",
    )
    quantized_match = _last_match(
        text,
        r"Serialized quantized model: (?P<bytes>\d+) bytes \(payload:(?P<payload>\d+)\)",
    )
    total_match = _last_match(text, r"Total submission size: (?P<bytes>\d+) bytes")
    code_match = _last_match(text, r"Code size: (?P<bytes>\d+) bytes")
    variant_match = _last_match(text, r"^variant:(?P<variant>[a-z0-9_-]+)$")
    layout_match = _last_match(text, r"^layout:(?P<value>.+)$")
    train_match = _last_match(text, r"^train:(?P<value>.+)$")
    optimizer_match = _last_match(text, r"^optimizer:(?P<value>.+)$")
    quant_match = _last_match(text, r"^quant:(?P<value>.+)$")
    return ParsedMetrics(
        variant=variant_match.group("variant") if variant_match else None,
        layout=layout_match.group("value") if layout_match else None,
        train_cfg=train_match.group("value") if train_match else None,
        optimizer_cfg=optimizer_match.group("value") if optimizer_match else None,
        quant_cfg=quant_match.group("value") if quant_match else None,
        final_val_loss=float(final_match.group("loss")) if final_match else None,
        final_val_bpb=float(final_match.group("bpb")) if final_match else None,
        sliding_val_loss=float(sliding_match.group("loss")) if sliding_match else None,
        sliding_val_bpb=float(sliding_match.group("bpb")) if sliding_match else None,
        sliding_stride=int(sliding_match.group("stride")) if sliding_match else None,
        bytes_total=int(total_match.group("bytes")) if total_match else None,
        bytes_code=int(code_match.group("bytes")) if code_match else None,
        bytes_quantized=int(quantized_match.group("bytes")) if quantized_match else None,
        bytes_payload=int(quantized_match.group("payload")) if quantized_match else None,
    )


def build_submission_json(
    *,
    target_name: str,
    author: str,
    github_id: str,
    blurb: str | None,
    metrics: ParsedMetrics,
) -> dict[str, object]:
    variant = metrics.variant or "candidate"
    return {
        "author": author,
        "github_id": github_id,
        "name": target_name,
        "blurb": blurb
        or (
            f"{variant} snapshot from the 2026-03-20 SOTA workbench with vendored helpers; "
            f"score is the final exact post-roundtrip metric recorded in train.log."
        ),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_loss": metrics.final_val_loss,
        "val_bpb": metrics.final_val_bpb,
        "bytes_total": metrics.bytes_total,
        "bytes_code": metrics.bytes_code,
    }


def build_readme(
    *,
    submission_name: str,
    source_name: str,
    metrics: ParsedMetrics,
    has_log: bool,
) -> str:
    variant = metrics.variant or "unknown"
    run_script = "./run_mainline.sh" if variant == "mainline" else "./run_challenger.sh"
    lines = [
        f"This record captures `{submission_name}`.",
        "",
        f"It was packaged from `{source_name}` as a portable records candidate with vendored helper files, so it can compile and run from inside this records folder without reaching back into the repo root.",
        "",
        "Configuration:",
    ]
    for label, value in (
        ("Layout", metrics.layout),
        ("Training", metrics.train_cfg),
        ("Optimizer", metrics.optimizer_cfg),
        ("Quantization", metrics.quant_cfg),
    ):
        if value:
            lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "Command (track-relevant entry point):",
            "```bash",
            run_script,
            "```",
            "",
        ]
    )
    if has_log:
        lines.append("Key metrics (from `train.log`):")
        if metrics.final_val_loss is not None and metrics.final_val_bpb is not None:
            lines.append(f"- Final exact roundtrip eval: `val_loss:{metrics.final_val_loss:.8f}`, `val_bpb:{metrics.final_val_bpb:.8f}`")
        if metrics.sliding_val_loss is not None and metrics.sliding_val_bpb is not None and metrics.sliding_stride is not None:
            lines.append(
                f"- Final sliding-window exact eval: `val_loss:{metrics.sliding_val_loss:.8f}`, `val_bpb:{metrics.sliding_val_bpb:.8f}`, `stride:{metrics.sliding_stride}`"
            )
        if metrics.bytes_quantized is not None and metrics.bytes_payload is not None:
            lines.append(
                f"- Serialized quantized model: `{metrics.bytes_quantized}` bytes, payload `{metrics.bytes_payload}` bytes"
            )
        if metrics.bytes_code is not None:
            lines.append(f"- Code size: `{metrics.bytes_code}` bytes")
        if metrics.bytes_total is not None:
            lines.append(f"- Total submission size: `{metrics.bytes_total}` bytes")
        lines.append("")
    lines.extend(
        [
            "Included files:",
            "- `train_gpt.py` (records entry point used for the run family)",
            "- `root_train_gpt_vendor.py` plus local helper modules vendored for portability",
            "- `train.log` (exact training log used for packaging)" if has_log else "- `train.log` is not included yet; rerun packaging after a measured run",
            "- `submission.json` (leaderboard metadata)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the SOTA workbench into a portable records submission folder and fill metadata from train.log."
    )
    parser.add_argument("name", help="Target folder name under records/track_10min_16mb")
    parser.add_argument("--source", default=str(WORKBENCH), help="Source records workbench folder")
    parser.add_argument("--log-path", default="", help="Optional explicit train.log path; defaults to <source>/train.log")
    parser.add_argument("--author", default="TBD")
    parser.add_argument("--github-id", default="TBD")
    parser.add_argument("--submission-name", default="")
    parser.add_argument("--blurb", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    target = ROOT / "records" / "track_10min_16mb" / args.name
    snapshot_candidate(target, source=source, overwrite=args.overwrite, include_log=False)

    log_path = Path(args.log_path) if args.log_path else source / "train.log"
    has_log = log_path.exists()
    if has_log:
        shutil.copy2(log_path, target / "train.log")
        metrics = parse_train_log_metrics(log_path)
    else:
        metrics = ParsedMetrics(
            variant=None,
            layout=None,
            train_cfg=None,
            optimizer_cfg=None,
            quant_cfg=None,
            final_val_loss=None,
            final_val_bpb=None,
            sliding_val_loss=None,
            sliding_val_bpb=None,
            sliding_stride=None,
            bytes_total=None,
            bytes_code=None,
            bytes_quantized=None,
            bytes_payload=None,
        )

    submission_name = args.submission_name or args.name
    submission = build_submission_json(
        target_name=submission_name,
        author=args.author,
        github_id=args.github_id,
        blurb=args.blurb or None,
        metrics=metrics,
    )
    (target / "submission.json").write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        build_readme(submission_name=submission_name, source_name=source.name, metrics=metrics, has_log=has_log),
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
