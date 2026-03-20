from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.log_workbench_run import parse_workbench_log


class RecordCompetitionRunTests(unittest.TestCase):
    def test_parse_workbench_log_extracts_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "train.log"
            log_path.write_text(
                "\n".join(
                    [
                        "variant:challenger",
                        "| NVIDIA H100 80GB HBM3 |",
                        "step:50/2000 train_loss:4.5000 train_time:1000ms step_avg:20.00ms",
                        "step:100/2000 val_loss:4.1000 val_bpb:2.4000 train_time:2000ms",
                        "Serialized model: 12345 bytes",
                        "Serialized quantized model: 6789 bytes (payload:6000)",
                        "Code size: 4321 bytes",
                        "Total submission size: 11110 bytes",
                        "chosen_export_candidate compressor:zstd zstd_level:22 bit_overrides:.mlp.:5 keep_float:tok_emb.weight eval_mode:doc_sliding stride:64 bytes_total:11110",
                        "stage_timing:train ms:2000.000",
                        "stage_timing:quantize ms:12.500",
                        "final_roundtrip_exact val_loss:5.02976709 val_bpb:2.97891138",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = parse_workbench_log(log_path)

        self.assertEqual(parsed["variant"], "challenger")
        self.assertEqual(parsed["gpu"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(parsed["steps_completed"], 100)
        self.assertEqual(parsed["iterations"], 2000)
        self.assertAlmostEqual(parsed["train_wallclock_seconds"], 2.0)
        self.assertAlmostEqual(parsed["pre_roundtrip_val_bpb"], 2.4)
        self.assertAlmostEqual(parsed["final_roundtrip_val_bpb"], 2.97891138)
        self.assertEqual(parsed["raw_model_bytes"], 12345)
        self.assertEqual(parsed["quantized_model_bytes"], 6789)
        self.assertEqual(parsed["code_bytes"], 4321)
        self.assertEqual(parsed["total_submission_bytes"], 11110)
        self.assertIn("compressor:zstd", parsed["chosen_export_candidate"])
        self.assertAlmostEqual(parsed["stage_timings_ms"]["quantize"], 12.5)

    def test_parse_workbench_log_allows_missing_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "train.log"
            log_path.write_text(
                "\n".join(
                    [
                        "variant:challenger",
                        "| NVIDIA H100 80GB HBM3 |",
                        "step:376/2000 val_loss:3.2289 val_bpb:1.9123 train_time:600645ms",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = parse_workbench_log(log_path)

        self.assertAlmostEqual(parsed["pre_roundtrip_val_bpb"], 1.9123)
        self.assertIsNone(parsed["final_roundtrip_val_bpb"])


if __name__ == "__main__":
    unittest.main()
