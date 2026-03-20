from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path


FINAL_SEEDS = (1337, 2026, 4242)
LOWBIT_PATTERNS = ".mlp.,.attn.c_q.,.attn.c_v.,.attn.proj."


def metric_tuple(run: dict[str, object]) -> tuple[float, int, int, float]:
    summary = run["summary"]
    return (
        float(summary.get("final_exact_val_bpb", summary.get("final_int8_zlib_roundtrip_exact_val_bpb", math.inf))),
        int(summary.get("bytes_total", 1 << 62)),
        int(summary.get("roundtrip_eval_time_ms", 1 << 30)),
        float(summary.get("final_exact_val_loss", summary.get("final_int8_zlib_roundtrip_exact_val_loss", math.inf))),
    )


def load_runs(runs_dir: Path) -> list[dict[str, object]]:
    runs = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        config_path = summary_path.with_name("config.json")
        if not config_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8")).get("config", {})
        if "final_int8_zlib_roundtrip_exact_val_bpb" not in summary and "final_exact_val_bpb" not in summary:
            continue
        runs.append({"run_id": summary.get("run_id", summary_path.parent.name), "summary": summary, "config": config})
    return runs


def select_top(runs: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    return sorted(runs, key=metric_tuple)[:top_k]


def run_id(stage: str, cfg: dict[str, object], idx: int, seed: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [stage, f"l{cfg.get('NUM_LAYERS', 9)}", f"d{cfg.get('MODEL_DIM', 512)}", f"m{cfg.get('MLP_HIDDEN', cfg.get('MLP_MULT', 2))}"]
    if cfg.get("LOWBIT_STE"):
        parts.append("ste")
    if cfg.get("MTP_NUM_HEADS", 0):
        parts.append(f"mtp{cfg['MTP_NUM_HEADS']}")
    if cfg.get("OPTIMIZER_VARIANT") == "normuon":
        parts.append("normuon")
    if cfg.get("SWA_ENABLED"):
        parts.append("swa")
    if seed is not None:
        parts.append(f"s{seed}")
    parts.append(f"{idx:02d}")
    return "_".join([stamp, *parts])


def base_env(args: argparse.Namespace) -> dict[str, object]:
    env = {
        "DATA_PATH": args.data_path,
        "TOKENIZER_PATH": args.tokenizer_path,
        "VOCAB_SIZE": 1024,
        "NUM_LAYERS": 9,
        "MODEL_DIM": 512,
        "NUM_HEADS": 8,
        "NUM_KV_HEADS": 4,
        "MLP_MULT": 2,
        "MLP_HIDDEN": 1024,
        "MLP_KIND": "relu2",
        "NUM_UNIQUE_BLOCKS": 0,
        "TIE_EMBEDDINGS": 1,
        "TRAIN_BATCH_TOKENS": 524288,
        "TRAIN_SEQ_LEN": 1024,
        "EVAL_SEQ_LEN": 1024,
        "EVAL_STRIDE": 0,
        "VAL_LOSS_EVERY": 2000,
        "ROUNDTRIP_VAL_EVERY": 0,
        "SAVE_BEST_BY": "val_bpb",
        "MAX_WALLCLOCK_SECONDS": 600,
        "RUNS_DIR": args.runs_dir,
        "BEST_DIR": args.best_dir,
        "PROMOTE_METRIC": "final_exact_val_bpb",
        "PROMOTE_MAX_BYTES": 16000000,
        "PYTHONUNBUFFERED": 1,
        "SERIAL_COMPRESSOR": "zlib",
        "WEIGHT_QUANT_BITS": 8,
        "EMBED_QUANT_BITS": 8,
        "LOWBIT_NAME_PATTERNS": "",
        "KEEP_FLOAT_NAME_PATTERNS": "",
        "GROUPED_INT8_NAME_PATTERNS": "",
        "FP16_EMBED_EXPORT": 0,
        "LOWBIT_STE": 0,
        "MTP_NUM_HEADS": 0,
        "MTP_LOSS_WEIGHT": 0.01,
        "OPTIMIZER_VARIANT": "muon",
        "SWA_ENABLED": 0,
    }
    return {k: v for k, v in env.items() if v not in (None, "")}


def format_command(args: argparse.Namespace, env: dict[str, object]) -> str:
    env_blob = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    return f"env {env_blob} torchrun --standalone --nproc_per_node={args.nproc_per_node} {shlex.quote(args.trainer)}"


def emit(stage: str, cfgs: list[dict[str, object]], args: argparse.Namespace, seed_override: int | None = None) -> None:
    shared = base_env(args)
    for idx, cfg in enumerate(cfgs, start=1):
        env = dict(shared)
        env.update(cfg)
        if seed_override is not None:
            env["SEED"] = seed_override
        env["RUN_ID"] = run_id(stage, env, idx, seed_override)
        print(format_command(args, env))


def baseline_cfgs() -> list[dict[str, object]]:
    return [{"VAL_LOSS_EVERY": 0, "SAVE_BEST_BY": "val_bpb"}]


def eval_export_cfgs() -> list[dict[str, object]]:
    return [{
        "EVAL_STRIDE": 64,
        "ROUNDTRIP_VAL_EVERY": 2000,
        "SAVE_BEST_BY": "roundtrip_val_bpb",
        "WEIGHT_QUANT_BITS": 6,
        "EMBED_QUANT_BITS": 16,
        "SERIAL_COMPRESSOR": "zstd",
        "LOWBIT_NAME_PATTERNS": LOWBIT_PATTERNS,
        "KEEP_FLOAT_NAME_PATTERNS": "tok_emb.weight",
        "FP16_EMBED_EXPORT": 1,
    }]


def best_run(runs: list[dict[str, object]], predicate, message: str) -> dict[str, object]:
    matches = [run for run in runs if predicate(run)]
    if not matches:
        raise SystemExit(message)
    return select_top(matches, 1)[0]


def config_subset(config: dict[str, object], *keys: str) -> dict[str, object]:
    return {key: config[key] for key in keys if key in config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit leaderboard search commands from tracked run results.")
    parser.add_argument("stage", choices=("baseline", "eval_export", "mlp3x", "ste", "context", "mtp", "optimizer", "final"))
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--best-dir", default="artifacts/best")
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer-path", default=os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
    parser.add_argument("--trainer", default="./train_gpt.py")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    args = parser.parse_args()

    if args.stage == "baseline":
        emit("baseline", baseline_cfgs(), args)
        return
    if args.stage == "eval_export":
        emit("eval_export", eval_export_cfgs(), args)
        return

    runs = load_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"No completed runs found in {args.runs_dir}")

    if args.stage == "mlp3x":
        run = best_run(
            runs,
            lambda r: int(r["config"].get("WEIGHT_QUANT_BITS", 8)) == 6 and int(r["config"].get("EVAL_STRIDE", 0)) > 0,
            "Need one completed eval_export-style run to emit mlp3x.",
        )
        cfg = config_subset(run["config"], "WEIGHT_QUANT_BITS", "EMBED_QUANT_BITS", "SERIAL_COMPRESSOR", "LOWBIT_NAME_PATTERNS", "KEEP_FLOAT_NAME_PATTERNS", "FP16_EMBED_EXPORT", "EVAL_STRIDE", "ROUNDTRIP_VAL_EVERY", "SAVE_BEST_BY")
        cfg.update({"MLP_HIDDEN": 1536})
        emit("mlp3x", [cfg], args)
        return

    if args.stage == "ste":
        run = best_run(
            runs,
            lambda r: int(r["config"].get("MLP_HIDDEN", 0)) == 1536 and int(r["config"].get("WEIGHT_QUANT_BITS", 8)) == 6,
            "Need one completed mlp3x run to emit ste.",
        )
        cfg = config_subset(run["config"], "MLP_HIDDEN", "WEIGHT_QUANT_BITS", "EMBED_QUANT_BITS", "SERIAL_COMPRESSOR", "LOWBIT_NAME_PATTERNS", "KEEP_FLOAT_NAME_PATTERNS", "FP16_EMBED_EXPORT", "EVAL_STRIDE", "ROUNDTRIP_VAL_EVERY", "SAVE_BEST_BY")
        cfg.update({
            "LOWBIT_STE": 1,
            "LOWBIT_STE_START_FRAC": 0.80,
            "LOWBIT_STE_LR_SCALE": 0.20,
            "LOWBIT_STE_NAME_PATTERNS": LOWBIT_PATTERNS,
        })
        emit("ste", [cfg], args)
        return

    if args.stage == "context":
        run = best_run(
            runs,
            lambda r: int(r["config"].get("LOWBIT_STE", 0)) == 1,
            "Need one completed ste run to emit context.",
        )
        cfgs = []
        for seq_len in (1024, 2048, 4096):
            cfg = config_subset(
                run["config"],
                "MLP_HIDDEN", "WEIGHT_QUANT_BITS", "EMBED_QUANT_BITS", "SERIAL_COMPRESSOR", "LOWBIT_NAME_PATTERNS",
                "KEEP_FLOAT_NAME_PATTERNS", "FP16_EMBED_EXPORT", "LOWBIT_STE", "LOWBIT_STE_START_FRAC",
                "LOWBIT_STE_LR_SCALE", "LOWBIT_STE_NAME_PATTERNS", "ROUNDTRIP_VAL_EVERY", "SAVE_BEST_BY",
            )
            cfg.update({"TRAIN_SEQ_LEN": seq_len, "EVAL_SEQ_LEN": seq_len, "EVAL_STRIDE": 64 if seq_len <= 2048 else 512})
            cfgs.append(cfg)
        emit("context", cfgs, args)
        return

    if args.stage == "mtp":
        run = best_run(
            runs,
            lambda r: int(r["config"].get("LOWBIT_STE", 0)) == 1 and int(r["config"].get("TRAIN_SEQ_LEN", 0)) in {1024, 2048, 4096},
            "Need one completed context run to emit mtp.",
        )
        cfg = config_subset(
            run["config"],
            "MLP_HIDDEN", "WEIGHT_QUANT_BITS", "EMBED_QUANT_BITS", "SERIAL_COMPRESSOR", "LOWBIT_NAME_PATTERNS",
            "KEEP_FLOAT_NAME_PATTERNS", "FP16_EMBED_EXPORT", "LOWBIT_STE", "LOWBIT_STE_START_FRAC",
            "LOWBIT_STE_LR_SCALE", "LOWBIT_STE_NAME_PATTERNS", "TRAIN_SEQ_LEN", "EVAL_SEQ_LEN", "EVAL_STRIDE",
            "ROUNDTRIP_VAL_EVERY", "SAVE_BEST_BY",
        )
        cfg.update({"MTP_NUM_HEADS": 1, "MTP_LOSS_WEIGHT": 0.01})
        emit("mtp", [cfg], args)
        return

    if args.stage == "optimizer":
        run = best_run(
            runs,
            lambda r: int(r["config"].get("MTP_NUM_HEADS", 0)) == 1,
            "Need one completed mtp run to emit optimizer.",
        )
        base = config_subset(
            run["config"],
            "MLP_HIDDEN", "WEIGHT_QUANT_BITS", "EMBED_QUANT_BITS", "SERIAL_COMPRESSOR", "LOWBIT_NAME_PATTERNS",
            "KEEP_FLOAT_NAME_PATTERNS", "FP16_EMBED_EXPORT", "LOWBIT_STE", "LOWBIT_STE_START_FRAC",
            "LOWBIT_STE_LR_SCALE", "LOWBIT_STE_NAME_PATTERNS", "TRAIN_SEQ_LEN", "EVAL_SEQ_LEN", "EVAL_STRIDE",
            "MTP_NUM_HEADS", "MTP_LOSS_WEIGHT", "ROUNDTRIP_VAL_EVERY", "SAVE_BEST_BY",
        )
        emit("optimizer", [
            {**base, "OPTIMIZER_VARIANT": "normuon"},
            {**base, "OPTIMIZER_VARIANT": "normuon", "SWA_ENABLED": 1, "SWA_START_FRAC": 0.85, "SWA_EVERY_STEPS": 200},
        ], args)
        return

    finalists = select_top(
        [run for run in runs if int(run["summary"].get("bytes_total", 1 << 62)) <= 16_000_000],
        2,
    )
    if len(finalists) < 2:
        raise SystemExit("Need two completed runs under the artifact cap to emit final.")
    for candidate_idx, run in enumerate(finalists, start=1):
        cfg = config_subset(
            run["config"],
            "NUM_LAYERS", "MODEL_DIM", "NUM_HEADS", "NUM_KV_HEADS", "MLP_MULT", "MLP_HIDDEN", "MLP_KIND",
            "NUM_UNIQUE_BLOCKS", "TIE_EMBEDDINGS", "TRAIN_BATCH_TOKENS", "TRAIN_SEQ_LEN", "EVAL_SEQ_LEN",
            "EVAL_STRIDE", "VAL_LOSS_EVERY", "ROUNDTRIP_VAL_EVERY", "SAVE_BEST_BY", "SERIAL_COMPRESSOR",
            "WEIGHT_QUANT_BITS", "EMBED_QUANT_BITS", "LOWBIT_NAME_PATTERNS", "KEEP_FLOAT_NAME_PATTERNS",
            "GROUPED_INT8_NAME_PATTERNS", "FP16_EMBED_EXPORT", "LOWBIT_STE", "LOWBIT_STE_START_FRAC",
            "LOWBIT_STE_LR_SCALE", "LOWBIT_STE_NAME_PATTERNS", "MTP_NUM_HEADS", "MTP_LOSS_WEIGHT",
            "OPTIMIZER_VARIANT", "SWA_ENABLED", "SWA_START_FRAC", "SWA_EVERY_STEPS",
        )
        for seed in FINAL_SEEDS:
            emit(f"final{candidate_idx}", [cfg], args, seed_override=seed)


if __name__ == "__main__":
    main()
