from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

try:
    from experiments.record_competition_run import LEDGER_JSONL, LEDGER_MD, dump_entries, load_entries, now_iso, write_markdown
except ImportError:
    from record_competition_run import LEDGER_JSONL, LEDGER_MD, dump_entries, load_entries, now_iso, write_markdown


GPU_RE = re.compile(r"NVIDIA H100[^\n|]*")
STEP_RE = re.compile(r"step:(\d+)/(\d+)")
TRAIN_TIME_RE = re.compile(r"train_time:(\d+(?:\.\d+)?)ms")
VAL_RE = re.compile(r"step:\d+/\d+ val_loss:(\d+(?:\.\d+)?) val_bpb:(\d+(?:\.\d+)?) train_time:(\d+(?:\.\d+)?)ms")
FINAL_RE = re.compile(r"final_roundtrip_exact val_loss:(\d+(?:\.\d+)?) val_bpb:(\d+(?:\.\d+)?)")
RAW_BYTES_RE = re.compile(r"Serialized model: (\d+) bytes")
QUANT_BYTES_RE = re.compile(r"Serialized quantized model: (\d+) bytes")
CODE_BYTES_RE = re.compile(r"Code size: (\d+) bytes")
TOTAL_BYTES_RE = re.compile(r"Total submission size: (\d+) bytes")
VARIANT_RE = re.compile(r"variant:([a-z0-9_-]+)")


def parse_workbench_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    variant_match = VARIANT_RE.search(text)
    final_match = FINAL_RE.search(text)
    if final_match is None:
        raise ValueError(f"Could not find final_roundtrip_exact metrics in {path}")
    val_matches = list(VAL_RE.finditer(text))
    step_matches = list(STEP_RE.finditer(text))
    train_time_matches = [float(match.group(1)) for match in TRAIN_TIME_RE.finditer(text)]
    gpu_match = GPU_RE.search(text)
    raw_bytes_match = RAW_BYTES_RE.search(text)
    quant_bytes_match = QUANT_BYTES_RE.search(text)
    code_bytes_match = CODE_BYTES_RE.search(text)
    total_bytes_match = TOTAL_BYTES_RE.search(text)

    parsed = {
        "variant": variant_match.group(1) if variant_match else "unknown",
        "gpu": gpu_match.group(0).strip() if gpu_match else "unknown",
        "steps_completed": int(step_matches[-1].group(1)) if step_matches else None,
        "iterations": int(step_matches[-1].group(2)) if step_matches else None,
        "train_wallclock_seconds": (max(train_time_matches) / 1000.0) if train_time_matches else None,
        "pre_roundtrip_val_loss": float(val_matches[-1].group(1)) if val_matches else None,
        "pre_roundtrip_val_bpb": float(val_matches[-1].group(2)) if val_matches else None,
        "final_roundtrip_val_loss": float(final_match.group(1)),
        "final_roundtrip_val_bpb": float(final_match.group(2)),
        "raw_model_bytes": int(raw_bytes_match.group(1)) if raw_bytes_match else None,
        "quantized_model_bytes": int(quant_bytes_match.group(1)) if quant_bytes_match else None,
        "code_bytes": int(code_bytes_match.group(1)) if code_bytes_match else None,
        "total_submission_bytes": int(total_bytes_match.group(1)) if total_bytes_match else None,
    }
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a records workbench train.log and append it to the competition ledger.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-shards", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--public-best-bpb", type=float, required=True)
    parser.add_argument("--public-best-ref", required=True)
    parser.add_argument("--changes", required=True)
    parser.add_argument("--summary-note", required=True)
    parser.add_argument("--timestamp", default=now_iso())
    args = parser.parse_args()

    parsed = parse_workbench_log(args.log_path)
    entry = {
        "timestamp": args.timestamp,
        "run_name": args.run_name,
        "variant": parsed["variant"],
        "gpu": parsed["gpu"],
        "gpu_count": args.gpu_count,
        "pod_id": args.pod_id or None,
        "train_wallclock_seconds": parsed["train_wallclock_seconds"],
        "train_shards": args.train_shards,
        "iterations": parsed["iterations"],
        "steps_completed": parsed["steps_completed"],
        "pre_roundtrip_val_loss": parsed["pre_roundtrip_val_loss"],
        "pre_roundtrip_val_bpb": parsed["pre_roundtrip_val_bpb"],
        "final_roundtrip_val_loss": parsed["final_roundtrip_val_loss"],
        "final_roundtrip_val_bpb": parsed["final_roundtrip_val_bpb"],
        "raw_model_bytes": parsed["raw_model_bytes"],
        "quantized_model_bytes": parsed["quantized_model_bytes"],
        "code_bytes": parsed["code_bytes"],
        "total_submission_bytes": parsed["total_submission_bytes"],
        "public_best_bpb": args.public_best_bpb,
        "public_best_ref": args.public_best_ref,
        "gap_to_public_best_bpb": parsed["final_roundtrip_val_bpb"] - args.public_best_bpb,
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
