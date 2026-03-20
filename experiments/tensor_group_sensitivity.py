from __future__ import annotations

import argparse
import json

import torch

from lowbit_utils import compress_quantized, quantize_state_dict


GROUPS = {
    "embeddings": ("tok_emb.weight",),
    "early_k": tuple(f"blocks.{i}.attn.c_k.weight" for i in range(0, 4)),
    "late_k": tuple(f"blocks.{i}.attn.c_k.weight" for i in range(8, 12)),
    "qvo": (".attn.c_q.", ".attn.c_v.", ".attn.proj."),
    "mlp_up": (".mlp.fc.", ".mlp.up.", ".mlp.gate."),
    "mlp_down": (".mlp.proj.",),
    "control": ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights"),
    "bigram": ("bigram.table.weight", "bigram.proj.weight"),
}


def parse_overrides(raw: str) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for part in [piece for piece in raw.split(",") if piece]:
        pattern, _, bits_raw = part.partition(":")
        if not pattern or not bits_raw:
            raise ValueError(f"override entries must look like pattern:bits, got {part!r}")
        out.append((pattern, int(bits_raw)))
    return tuple(out)


def build_overrides(group: str, bits: int) -> tuple[tuple[str, int], ...]:
    return tuple((pattern, bits) for pattern in GROUPS[group])


def merge_overrides(base: tuple[tuple[str, int], ...], extra: tuple[tuple[str, int], ...]) -> tuple[tuple[str, int], ...]:
    merged: dict[str, int] = {}
    for pattern, bits in (*base, *extra):
        merged[pattern] = bits
    return tuple(merged.items())


def format_overrides(overrides: tuple[tuple[str, int], ...]) -> str:
    return ",".join(f"{pattern}:{bits}" for pattern, bits in overrides)


def quantized_size(
    state_dict: dict[str, torch.Tensor],
    *,
    bit_overrides: tuple[tuple[str, int], ...],
    keep_float: tuple[str, ...],
    compressor: str,
) -> tuple[int, int]:
    quant_obj, quant_stats = quantize_state_dict(
        state_dict,
        weight_quant_bits=6,
        embed_quant_bits=16,
        lowbit_name_patterns=(".attn.", ".mlp."),
        keep_float_name_patterns=keep_float,
        grouped_int8_name_patterns=(),
        group_size=64,
        keep_float_max_numel=65_536,
        keep_float_fp32_name_patterns=("attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights"),
        keep_float_store_dtype=torch.float16,
        per_row_scale_dtype=torch.float16,
        clip_q=0.9999984,
        fp16_embed_export=True,
        bit_overrides=bit_overrides,
    )
    blob, _ = compress_quantized(quant_obj, compressor)
    return len(blob), quant_stats["int8_payload_bytes"]


def candidate_allocation(
    *,
    group: str,
    bits: int,
    base_overrides: tuple[tuple[str, int], ...],
    base_keep_float: tuple[str, ...],
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    if group == "embeddings" and bits == 16:
        keep = base_keep_float if "tok_emb.weight" in base_keep_float else (*base_keep_float, "tok_emb.weight")
        return base_overrides, keep
    keep = tuple(pattern for pattern in base_keep_float if not (group == "embeddings" and pattern == "tok_emb.weight"))
    return merge_overrides(base_overrides, build_overrides(group, bits)), keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure byte savings for tensor-group precision changes.")
    parser.add_argument("checkpoint", help="Path to a raw model checkpoint, usually final_model.pt")
    parser.add_argument("--compressor", default="zstd", choices=("zlib", "zstd"))
    parser.add_argument("--keep-float", default="tok_emb.weight")
    parser.add_argument("--base-bit-overrides", default="")
    parser.add_argument("--groups", default="mlp_up,mlp_down,qvo,early_k,late_k,bigram,embeddings")
    parser.add_argument("--candidate-bits", default="5,6,8,16")
    parser.add_argument("--emit-eval-cmds", action="store_true")
    parser.add_argument("--target-total-bytes", type=int, default=0)
    parser.add_argument("--code-bytes", type=int, default=0)
    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    keep_float = tuple(part for part in args.keep_float.split(",") if part)
    base_overrides = parse_overrides(args.base_bit_overrides)
    baseline_size, baseline_payload = quantized_size(
        state_dict,
        bit_overrides=base_overrides,
        keep_float=keep_float,
        compressor=args.compressor,
    )

    groups = [g for g in args.groups.split(",") if g]
    candidate_bits = [int(x) for x in args.candidate_bits.split(",") if x]
    report: list[dict[str, object]] = []
    for group in groups:
        for bits in candidate_bits:
            if group == "embeddings" and bits not in {8, 16}:
                continue
            overrides, extra_keep = candidate_allocation(
                group=group,
                bits=bits,
                base_overrides=base_overrides,
                base_keep_float=keep_float,
            )
            size, payload = quantized_size(
                state_dict,
                bit_overrides=overrides,
                keep_float=extra_keep,
                compressor=args.compressor,
            )
            item = {
                "group": group,
                "bits": bits,
                "compressed_bytes": size,
                "payload_bytes": payload,
                "delta_compressed_bytes": size - baseline_size,
                "delta_payload_bytes": payload - baseline_payload,
            }
            if args.emit_eval_cmds:
                env_parts = []
                if extra_keep:
                    env_parts.append(f"KEEP_FLOAT_NAME_PATTERNS={','.join(extra_keep)}")
                if overrides:
                    env_parts.append(f"BIT_OVERRIDES={format_overrides(overrides)}")
                item["eval_env"] = " ".join(env_parts)
            report.append(item)

    greedy_plan: list[dict[str, object]] = []
    if args.target_total_bytes > 0:
        current_overrides = base_overrides
        current_keep = keep_float
        current_size = baseline_size
        remaining_groups = list(groups)
        while current_size + args.code_bytes > args.target_total_bytes and remaining_groups:
            best: tuple[str, int, int, int, tuple[tuple[str, int], ...], tuple[str, ...]] | None = None
            for group in remaining_groups:
                for bits in candidate_bits:
                    if group == "embeddings" and bits not in {8, 16}:
                        continue
                    overrides, extra_keep = candidate_allocation(
                        group=group,
                        bits=bits,
                        base_overrides=current_overrides,
                        base_keep_float=current_keep,
                    )
                    size, payload = quantized_size(
                        state_dict,
                        bit_overrides=overrides,
                        keep_float=extra_keep,
                        compressor=args.compressor,
                    )
                    if best is None or size < best[2]:
                        best = (group, bits, size, payload, overrides, extra_keep)
            if best is None or best[2] >= current_size:
                break
            group, bits, current_size, payload, current_overrides, current_keep = best
            remaining_groups.remove(group)
            greedy_plan.append(
                {
                    "group": group,
                    "bits": bits,
                    "compressed_bytes": current_size,
                    "payload_bytes": payload,
                    "total_bytes_with_code": current_size + args.code_bytes,
                    "keep_float": list(current_keep),
                    "bit_overrides": format_overrides(current_overrides),
                }
            )

    print(
        json.dumps(
            {
                "baseline_compressed_bytes": baseline_size,
                "baseline_payload_bytes": baseline_payload,
                "base_bit_overrides": format_overrides(base_overrides),
                "base_keep_float": list(keep_float),
                "candidates": report,
                "greedy_plan": greedy_plan,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
