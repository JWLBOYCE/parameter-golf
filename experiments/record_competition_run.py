from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER_JSONL = ROOT / "experiments" / "competition_run_history.jsonl"
LEDGER_MD = ROOT / "experiments" / "competition_run_log.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def dump_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def _fmt_float(value: float | None, digits: int = 6) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else f"{value:,}"


def write_markdown(path: Path, entries: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [e for e in entries if e.get("final_roundtrip_val_bpb") is not None],
        key=lambda e: (
            float(e["final_roundtrip_val_bpb"]),
            int(e.get("total_submission_bytes") or 1 << 62),
            e.get("timestamp") or "",
        ),
    )
    top3 = ranked[:3]
    lines = [
        "# Competition Run Log",
        "",
        "This file tracks local and remote competition runs, including what changed, the measured result, and how that result compared with the best public run known at the time.",
        "",
        "## Top 3 Runs",
        "",
        "| Rank | Run | Final BPB | Gap To Public Best | Bytes | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, entry in enumerate(top3, start=1):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(idx),
                    entry.get("run_name", "-"),
                    _fmt_float(entry.get("final_roundtrip_val_bpb")),
                    _fmt_float(entry.get("gap_to_public_best_bpb")),
                    _fmt_int(entry.get("total_submission_bytes")),
                    entry.get("summary_note", "-").replace("\n", " "),
                )
            )
            + " |"
        )
    if not top3:
        lines.append("| - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| Timestamp | Run | Variant | GPU | Train Time (s) | Shards | Pre BPB | Final BPB | Public Best | Gap | Bytes | Changes | Notes |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for entry in sorted(entries, key=lambda e: e.get("timestamp", ""), reverse=True):
        lines.append(
            "| "
            + " | ".join(
                (
                    entry.get("timestamp", "-"),
                    entry.get("run_name", "-"),
                    entry.get("variant", "-"),
                    entry.get("gpu", "-"),
                    _fmt_float(entry.get("train_wallclock_seconds"), digits=1),
                    _fmt_int(entry.get("train_shards")),
                    _fmt_float(entry.get("pre_roundtrip_val_bpb")),
                    _fmt_float(entry.get("final_roundtrip_val_bpb")),
                    _fmt_float(entry.get("public_best_bpb")),
                    _fmt_float(entry.get("gap_to_public_best_bpb")),
                    _fmt_int(entry.get("total_submission_bytes")),
                    entry.get("changes", "-").replace("\n", " "),
                    entry.get("summary_note", "-").replace("\n", " "),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a competition run to the local run ledger.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--train-wallclock-seconds", type=float, required=True)
    parser.add_argument("--train-shards", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--steps-completed", type=int, default=0)
    parser.add_argument("--pre-roundtrip-val-loss", type=float)
    parser.add_argument("--pre-roundtrip-val-bpb", type=float)
    parser.add_argument("--final-roundtrip-val-loss", type=float)
    parser.add_argument("--final-roundtrip-val-bpb", type=float)
    parser.add_argument("--raw-model-bytes", type=int)
    parser.add_argument("--quantized-model-bytes", type=int)
    parser.add_argument("--code-bytes", type=int)
    parser.add_argument("--total-submission-bytes", type=int)
    parser.add_argument("--public-best-bpb", type=float, required=True)
    parser.add_argument("--public-best-ref", required=True)
    parser.add_argument("--changes", required=True)
    parser.add_argument("--summary-note", required=True)
    parser.add_argument("--timestamp", default=now_iso())
    args = parser.parse_args()

    entry = {
        "timestamp": args.timestamp,
        "run_name": args.run_name,
        "variant": args.variant,
        "gpu": args.gpu,
        "gpu_count": args.gpu_count,
        "pod_id": args.pod_id or None,
        "train_wallclock_seconds": args.train_wallclock_seconds,
        "train_shards": args.train_shards,
        "iterations": args.iterations or None,
        "steps_completed": args.steps_completed or None,
        "pre_roundtrip_val_loss": args.pre_roundtrip_val_loss,
        "pre_roundtrip_val_bpb": args.pre_roundtrip_val_bpb,
        "final_roundtrip_val_loss": args.final_roundtrip_val_loss,
        "final_roundtrip_val_bpb": args.final_roundtrip_val_bpb,
        "raw_model_bytes": args.raw_model_bytes,
        "quantized_model_bytes": args.quantized_model_bytes,
        "code_bytes": args.code_bytes,
        "total_submission_bytes": args.total_submission_bytes,
        "public_best_bpb": args.public_best_bpb,
        "public_best_ref": args.public_best_ref,
        "gap_to_public_best_bpb": (
            None
            if args.final_roundtrip_val_bpb is None
            else args.final_roundtrip_val_bpb - args.public_best_bpb
        ),
        "changes": args.changes,
        "summary_note": args.summary_note,
    }

    entries = load_entries(LEDGER_JSONL)
    entries.append(entry)
    dump_entries(LEDGER_JSONL, entries)
    write_markdown(LEDGER_MD, entries)
    print(LEDGER_JSONL)
    print(LEDGER_MD)


if __name__ == "__main__":
    main()
