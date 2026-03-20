from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local env may not have torch
    torch = None

if torch is not None:
    from experiments.export_frontier_sweep import choose_exact_candidates, default_sweep_space, rank_exact_results, size_prescreen


@unittest.skipUnless(torch is not None, "torch is required for export frontier sweep tests")
class ExportFrontierSweepTest(unittest.TestCase):
    def test_leader_parity_default_sweep_excludes_doc_sliding(self) -> None:
        defaults = {
            "bit_overrides": [],
            "keep_float_name_patterns": ["tok_emb.weight"],
            "grouped_int8_name_patterns": [],
            "zstd_level": 22,
            "eval_mode": "doc_sliding",
            "eval_stride": 256,
        }
        config = {"variant": "leader_parity", "num_layers": 11}
        sweep = default_sweep_space(config, defaults)
        self.assertEqual(sweep["eval_modes"], ["sliding"])
        self.assertEqual(sweep["strides"], [64])
        self.assertEqual(sweep["bit_overrides"], ["", ".mlp.:5"])
        self.assertIn("tok_emb.weight", sweep["keep_float"][0])

    def test_size_prescreen_marks_legal_candidates(self) -> None:
        raw_state = {
            "tok_emb.weight": torch.randn(32, 16),
            "blocks.0.attn.c_q.weight": torch.randn(16, 16),
            "blocks.0.mlp.fc.weight": torch.randn(32, 16),
        }
        defaults = {
            "weight_quant_bits": 6,
            "embed_quant_bits": 16,
            "lowbit_name_patterns": [".attn.", ".mlp."],
            "group_size": 64,
            "keep_float_max_numel": 0,
            "keep_float_fp32_name_patterns": ["attn_scale"],
            "keep_float_store_dtype": "float16",
            "per_row_scale_dtype": "float16",
            "clip_q": 0.9999984,
            "fp16_embed_export": True,
            "serial_compressor": "zlib",
        }
        candidates = [
            {
                "bit_overrides": tuple(),
                "keep_float_name_patterns": ("tok_emb.weight",),
                "grouped_int8_name_patterns": tuple(),
                "zstd_level": 22,
            }
        ]
        prescreened = size_prescreen(raw_state=raw_state, defaults=defaults, candidates=candidates, byte_budget=1_000_000, code_bytes=100)
        self.assertEqual(len(prescreened), 1)
        self.assertTrue(prescreened[0]["legal"])
        self.assertGreater(prescreened[0]["compressed_bytes"], 0)

    def test_choose_exact_candidates_respects_limit(self) -> None:
        prescreened = [
            {"legal": True, "total_bytes": 10},
            {"legal": True, "total_bytes": 20},
            {"legal": True, "total_bytes": 30},
        ]
        chosen = choose_exact_candidates(prescreened, 2)
        self.assertEqual([item["total_bytes"] for item in chosen], [10, 20])

    def test_rank_exact_results_prefers_bpb_then_bytes_then_time(self) -> None:
        results = [
            {"ranking_bpb": 1.2, "total_bytes": 100, "eval_time_ms": 20.0},
            {"ranking_bpb": 1.1, "total_bytes": 120, "eval_time_ms": 50.0},
            {"ranking_bpb": 1.1, "total_bytes": 110, "eval_time_ms": 70.0},
            {"ranking_bpb": 1.1, "total_bytes": 110, "eval_time_ms": 40.0},
        ]
        ranked = rank_exact_results(results)
        self.assertEqual(ranked[0]["eval_time_ms"], 40.0)
        self.assertEqual(ranked[1]["total_bytes"], 110)
        self.assertEqual(ranked[2]["total_bytes"], 120)


if __name__ == "__main__":
    unittest.main()
