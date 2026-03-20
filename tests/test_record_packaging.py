from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.package_record_submission import parse_train_log_metrics
from experiments.snapshot_record_candidate import ROOT, WORKBENCH, snapshot_candidate


class RecordPackagingTest(unittest.TestCase):
    def test_snapshot_candidate_keeps_records_entrypoint_and_vendors_root_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "candidate"
            snapshot_candidate(target, source=WORKBENCH, include_log=False)
            self.assertTrue((target / "train_gpt.py").exists())
            self.assertTrue((target / "root_train_gpt_vendor.py").exists())
            self.assertFalse((target / "train.log").exists())
            self.assertEqual(
                (target / "train_gpt.py").read_text(encoding="utf-8"),
                (WORKBENCH / "train_gpt.py").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (target / "root_train_gpt_vendor.py").read_text(encoding="utf-8"),
                (ROOT / "train_gpt.py").read_text(encoding="utf-8"),
            )

    def test_parse_train_log_metrics_extracts_roundtrip_and_size_fields(self) -> None:
        log_text = "\n".join(
            (
                "variant:mainline",
                "layout:vocab=1024 layers=10 dim=512 heads=8 kv=4 mlp_hidden=1536",
                "train:batch_tokens=786432 train_seq_len=2048 eval_seq_len=2048 eval_stride=64",
                "optimizer:muon matrix_lr=0.02 scalar_lr=0.02 tied_embed_lr=0.03 muon_wd=0.04",
                "quant:compressor=zstd weight_bits=6 embed_bits=16 bit_overrides=(('.mlp.', 5),) keep_float=('tok_emb.weight',)",
                "Serialized quantized model: 15500000 bytes (payload:17000000)",
                "Code size: 65000 bytes",
                "Total submission size: 15565000 bytes",
                "final_roundtrip_exact val_loss:1.94000000 val_bpb:1.15000000",
                "final_sliding_window_exact val_loss:1.93000000 val_bpb:1.14500000 stride:64",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "train.log"
            log_path.write_text(log_text, encoding="utf-8")
            metrics = parse_train_log_metrics(log_path)
        self.assertEqual(metrics.variant, "mainline")
        self.assertEqual(metrics.bytes_total, 15_565_000)
        self.assertEqual(metrics.bytes_code, 65_000)
        self.assertEqual(metrics.bytes_quantized, 15_500_000)
        self.assertEqual(metrics.bytes_payload, 17_000_000)
        self.assertAlmostEqual(metrics.final_val_loss or 0.0, 1.94)
        self.assertAlmostEqual(metrics.final_val_bpb or 0.0, 1.15)
        self.assertAlmostEqual(metrics.sliding_val_bpb or 0.0, 1.145)
        self.assertEqual(metrics.sliding_stride, 64)


if __name__ == "__main__":
    unittest.main()
