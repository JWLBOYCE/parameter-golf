from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch

from lowbit_utils import compress_quantized, dequantize_state_dict, decompress_quantized, load_export_state_dict, quantize_state_dict
from validation_utils import build_sentencepiece_luts, clip_doc_offsets_to_total_tokens, load_doc_offsets


def parse_patterns(raw: str) -> tuple[str, ...]:
    return tuple(part for part in raw.split(",") if part)


def parse_bit_overrides(raw: str) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for part in parse_patterns(raw):
        pattern, _, bits_raw = part.partition(":")
        if not pattern or not bits_raw:
            raise ValueError(f"BIT_OVERRIDES entries must look like pattern:bits, got {part!r}")
        out.append((pattern, int(bits_raw)))
    return tuple(out)


def format_bit_overrides(overrides: tuple[tuple[str, int], ...]) -> str:
    return ",".join(f"{pattern}:{bits}" for pattern, bits in overrides)


def load_workbench_module(path: Path):
    spec = importlib.util.spec_from_file_location("pg_frontier_workbench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load workbench module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_sweep_space(config: dict[str, object], defaults: dict[str, object]) -> dict[str, list[object]]:
    if str(config.get("variant")) == "leader_parity":
        late_keep = ",".join(
            [
                "tok_emb.weight",
                f"blocks.{max(int(config['num_layers']) - 2, 0)}.attn.c_k.weight",
                f"blocks.{max(int(config['num_layers']) - 1, 0)}.attn.c_k.weight",
            ]
        )
        return {
            "bit_overrides": ["", ".mlp.:5"],
            "keep_float": ["tok_emb.weight", late_keep],
            "grouped_int8": [""],
            "zstd_levels": [19, 22],
            "eval_modes": ["sliding"],
            "strides": [64],
        }
    return {
        "bit_overrides": [format_bit_overrides(tuple(tuple(x) for x in defaults["bit_overrides"]))],
        "keep_float": [",".join(defaults["keep_float_name_patterns"])],
        "grouped_int8": [",".join(defaults["grouped_int8_name_patterns"])],
        "zstd_levels": [int(defaults["zstd_level"])],
        "eval_modes": [str(defaults["eval_mode"])],
        "strides": [int(defaults["eval_stride"])],
    }


def normalize_candidate_lists(defaults: dict[str, object], args) -> tuple[list[str], list[str], list[str], list[int], list[str], list[int]]:
    sweep_defaults = default_sweep_space(args.manifest_config, defaults)
    bit_override_values = args.bit_overrides if args.bit_overrides is not None else [str(x) for x in sweep_defaults["bit_overrides"]]
    keep_float_values = args.keep_float if args.keep_float is not None else [str(x) for x in sweep_defaults["keep_float"]]
    grouped_values = args.grouped_int8 if args.grouped_int8 is not None else [str(x) for x in sweep_defaults["grouped_int8"]]
    zstd_levels = [int(x) for x in (args.zstd_levels.split(",") if args.zstd_levels else [str(x) for x in sweep_defaults["zstd_levels"]]) if x]
    eval_modes = [x for x in (args.eval_modes.split(",") if args.eval_modes else [str(x) for x in sweep_defaults["eval_modes"]]) if x]
    strides = [int(x) for x in (args.strides.split(",") if args.strides else [str(x) for x in sweep_defaults["strides"]]) if x]
    return bit_override_values, keep_float_values, grouped_values, zstd_levels, eval_modes, strides


def build_export_candidates(defaults: dict[str, object], args) -> list[dict[str, object]]:
    bit_override_values, keep_float_values, grouped_values, zstd_levels, _, _ = normalize_candidate_lists(defaults, args)
    candidates: list[dict[str, object]] = []
    for bit_raw, keep_raw, grouped_raw, zstd_level in itertools.product(bit_override_values, keep_float_values, grouped_values, zstd_levels):
        candidates.append(
            {
                "bit_overrides": parse_bit_overrides(bit_raw),
                "keep_float_name_patterns": parse_patterns(keep_raw),
                "grouped_int8_name_patterns": parse_patterns(grouped_raw),
                "zstd_level": zstd_level,
            }
        )
    return candidates


def build_eval_settings(defaults: dict[str, object], args) -> list[tuple[str, int]]:
    _, _, _, _, eval_modes, strides = normalize_candidate_lists(defaults, args)
    settings: list[tuple[str, int]] = []
    for eval_mode, stride in itertools.product(eval_modes, strides):
        actual_stride = 0 if eval_mode == "contiguous" else stride
        settings.append((eval_mode, actual_stride))
    deduped: list[tuple[str, int]] = []
    for setting in settings:
        if setting not in deduped:
            deduped.append(setting)
    return deduped


def size_prescreen(
    *,
    raw_state: dict[str, torch.Tensor],
    defaults: dict[str, object],
    candidates: list[dict[str, object]],
    byte_budget: int,
    code_bytes: int,
) -> list[dict[str, object]]:
    prescreened: list[dict[str, object]] = []
    for candidate in candidates:
        quant_obj, quant_stats = quantize_state_dict(
            raw_state,
            weight_quant_bits=int(defaults["weight_quant_bits"]),
            embed_quant_bits=int(defaults["embed_quant_bits"]),
            lowbit_name_patterns=tuple(defaults["lowbit_name_patterns"]),
            keep_float_name_patterns=tuple(candidate["keep_float_name_patterns"]),
            grouped_int8_name_patterns=tuple(candidate["grouped_int8_name_patterns"]),
            group_size=int(defaults["group_size"]),
            keep_float_max_numel=int(defaults["keep_float_max_numel"]),
            keep_float_fp32_name_patterns=tuple(defaults["keep_float_fp32_name_patterns"]),
            keep_float_store_dtype=getattr(torch, str(defaults["keep_float_store_dtype"])),
            per_row_scale_dtype=getattr(torch, str(defaults["per_row_scale_dtype"])),
            clip_q=float(defaults["clip_q"]),
            fp16_embed_export=bool(defaults["fp16_embed_export"]),
            bit_overrides=tuple(candidate["bit_overrides"]),
        )
        blob, raw_bytes = compress_quantized(quant_obj, str(defaults["serial_compressor"]), zstd_level=int(candidate["zstd_level"]))
        total_bytes = len(blob) + code_bytes
        prescreened.append(
            {
                **candidate,
                "compressed_bytes": len(blob),
                "raw_bytes": raw_bytes,
                "payload_bytes": quant_stats["int8_payload_bytes"],
                "total_bytes": total_bytes,
                "legal": total_bytes <= byte_budget,
                "blob": blob,
            }
        )
    return sorted(prescreened, key=lambda item: (not item["legal"], item["total_bytes"], item["compressed_bytes"]))


def choose_exact_candidates(prescreened: list[dict[str, object]], max_exact_candidates: int) -> list[dict[str, object]]:
    legal = [item for item in prescreened if item["legal"]]
    if len(legal) <= max_exact_candidates:
        return legal
    return legal[:max_exact_candidates]


def markdown_summary(results: list[dict[str, object]], best: dict[str, object] | None) -> str:
    lines = ["# Frontier Sweep Results", ""]
    if best is not None:
        lines.append(f"- Best candidate: `val_bpb={best['ranking_bpb']:.8f}`, `bytes_total={best['total_bytes']}`")
        lines.append(
            f"- Export: `bit_overrides={best['bit_overrides_raw'] or '<default>'}`, `keep_float={best['keep_float_raw'] or '<default>'}`, `zstd={best['zstd_level']}`"
        )
        lines.append(f"- Eval: `mode={best['eval_mode']}`, `stride={best['eval_stride']}`")
        lines.append("")
    lines.append("| Rank | val_bpb | bytes_total | eval_mode | stride | bit_overrides | keep_float | zstd |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for rank, item in enumerate(results, start=1):
        lines.append(
            f"| {rank} | {item['ranking_bpb']:.8f} | {item['total_bytes']} | {item['eval_mode']} | {item['eval_stride']} | `{item['bit_overrides_raw']}` | `{item['keep_float_raw']}` | {item['zstd_level']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def rank_exact_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(results, key=lambda item: (float(item["ranking_bpb"]), int(item["total_bytes"]), float(item["eval_time_ms"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep post-train export candidates and exact-evaluate the best legal ones.")
    parser.add_argument("checkpoint", type=Path, help="Path to a raw frontier checkpoint, usually frontier_candidate.pt")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--byte-budget", type=int, default=16_000_000)
    parser.add_argument("--max-exact-candidates", type=int, default=8)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--bit-overrides", action="append", default=None)
    parser.add_argument("--keep-float", action="append", default=None)
    parser.add_argument("--grouped-int8", action="append", default=None)
    parser.add_argument("--zstd-levels", default="")
    parser.add_argument("--eval-modes", default="")
    parser.add_argument("--strides", default="")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.manifest_config = manifest["config"]
    defaults = manifest["export_defaults"]
    defaults.setdefault("lowbit_name_patterns", manifest["config"]["lowbit_name_patterns"])
    defaults.setdefault("group_size", manifest["config"]["group_size"])
    defaults.setdefault("keep_float_max_numel", 65_536)
    defaults.setdefault("keep_float_fp32_name_patterns", ["attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights"])
    defaults.setdefault("keep_float_store_dtype", "float16")
    defaults.setdefault("per_row_scale_dtype", "float16")
    defaults.setdefault("clip_q", 0.9999984)
    configured_artifact_dir = Path(str(manifest["config"].get("artifact_dir", "artifacts")))
    if not configured_artifact_dir.is_absolute():
        configured_artifact_dir = args.manifest.parent / configured_artifact_dir
    output_dir = args.output_dir or (configured_artifact_dir / "frontier_exports")
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    workbench_module = load_workbench_module(Path(manifest["workbench_train_gpt_path"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    raw_state = torch.load(args.checkpoint, map_location="cpu")
    export_candidates = build_export_candidates(defaults, args)
    prescreened = size_prescreen(
        raw_state=raw_state,
        defaults=defaults,
        candidates=export_candidates,
        byte_budget=args.byte_budget,
        code_bytes=int(manifest["code_bytes"]),
    )
    exact_candidates = choose_exact_candidates(prescreened, args.max_exact_candidates)

    workbench_args = SimpleNamespace(**manifest["config"])
    sp = spm.SentencePieceProcessor(model_file=manifest["tokenizer_path"])
    val_tokens = workbench_module.load_validation_tokens(workbench_args.val_files, max(int(workbench_args.train_seq_len), int(workbench_args.eval_seq_len)))
    doc_offsets = None
    if str(defaults["eval_mode"]) == "doc_sliding" or "doc_sliding" in (args.eval_modes.split(",") if args.eval_modes else []):
        doc_offsets = clip_doc_offsets_to_total_tokens(load_doc_offsets(manifest["val_doc_offsets_path"]), int(val_tokens.numel()))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, int(workbench_args.vocab_size), device)

    model = workbench_module.RecordGPT(workbench_args)
    if device.type == "cuda":
        model = model.to(device).bfloat16()
        for module in model.modules():
            if isinstance(module, workbench_module.CastedLinear):
                module.float()
    else:
        model = model.to(device)
    workbench_module.restore_low_dim_params_to_fp32(model)

    eval_settings = build_eval_settings(defaults, args)
    exact_results: list[dict[str, object]] = []
    for candidate_idx, candidate in enumerate(exact_candidates):
        blob = candidate["blob"]
        load_export_state_dict(model, dequantize_state_dict(decompress_quantized(blob, str(defaults["serial_compressor"]))))
        blob_path = candidates_dir / f"candidate_{candidate_idx:03d}.ptz"
        blob_path.write_bytes(blob)
        for eval_mode, eval_stride in eval_settings:
            workbench_args.eval_mode = eval_mode
            workbench_args.eval_stride = eval_stride
            eval_t0 = time.perf_counter()
            metrics = workbench_module.evaluate_model(
                args=workbench_args,
                model=model,
                rank=0,
                world_size=1,
                device=device,
                grad_accum_steps=1,
                val_tokens=val_tokens,
                doc_offsets=doc_offsets,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            eval_time_ms = 1000.0 * (time.perf_counter() - eval_t0)
            ranking_bpb = metrics["mode_bpb"] if eval_mode != "contiguous" and metrics["mode_bpb"] is not None else metrics["val_bpb"]
            exact_results.append(
                {
                    "candidate_id": candidate_idx,
                    "blob_path": str(blob_path),
                    "compressed_bytes": candidate["compressed_bytes"],
                    "payload_bytes": candidate["payload_bytes"],
                    "total_bytes": candidate["total_bytes"],
                    "bit_overrides_raw": format_bit_overrides(tuple(candidate["bit_overrides"])),
                    "keep_float_raw": ",".join(candidate["keep_float_name_patterns"]),
                    "grouped_int8_raw": ",".join(candidate["grouped_int8_name_patterns"]),
                    "zstd_level": candidate["zstd_level"],
                    "eval_mode": eval_mode,
                    "eval_stride": eval_stride,
                    "val_loss": metrics["val_loss"],
                    "val_bpb": metrics["val_bpb"],
                    "mode_loss": metrics["mode_loss"],
                    "mode_bpb": metrics["mode_bpb"],
                    "ranking_bpb": ranking_bpb,
                    "eval_time_ms": eval_time_ms,
                }
            )

    exact_results = rank_exact_results(exact_results)
    keep_ids = {item["candidate_id"] for item in exact_results[: max(1, args.keep_top_k)]}
    for item in exact_results:
        if item["candidate_id"] in keep_ids:
            continue
        path = Path(item["blob_path"])
        if path.exists():
            path.unlink()
    best = exact_results[0] if exact_results else None
    payload = {
        "manifest_path": str(args.manifest),
        "checkpoint_path": str(args.checkpoint),
        "byte_budget": args.byte_budget,
        "prescreened": [
            {k: v for k, v in item.items() if k != "blob"}
            for item in prescreened
        ],
        "results": exact_results,
        "best": best,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frontier_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "frontier_results.md").write_text(markdown_summary(exact_results, best), encoding="utf-8")
    if best is not None:
        print(
            "chosen_export_candidate "
            f"compressor:{defaults['serial_compressor']} zstd_level:{best['zstd_level']} "
            f"bit_overrides:{best['bit_overrides_raw']} keep_float:{best['keep_float_raw']} "
            f"eval_mode:{best['eval_mode']} stride:{best['eval_stride']} bytes_total:{best['total_bytes']} "
            f"val_bpb:{best['ranking_bpb']:.8f}"
        )
    print(output_dir / "frontier_results.json")
    print(output_dir / "frontier_results.md")


if __name__ == "__main__":
    main()
