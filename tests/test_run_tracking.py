from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_tracking import RunTracker


class RunTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs_dir = self.root / "artifacts" / "runs"
        self.best_dir = self.root / "artifacts" / "best"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _tracker(self, run_id: str, *, retain_top_k: int = 1) -> RunTracker:
        return RunTracker(
            run_id=run_id,
            trainer_name="test_trainer.py",
            backend="test",
            config={"run_id": run_id, "seed": 1337},
            runs_dir=str(self.runs_dir),
            best_dir=str(self.best_dir),
            promote_max_bytes=16_000_000,
            promote_metric="final_int8_zlib_roundtrip_exact_val_bpb",
            retain_top_k=retain_top_k,
            keep_nonbest_artifacts=False,
        )

    def _artifact_paths(self, tracker: RunTracker) -> dict[str, Path]:
        raw = tracker.run_dir / "model.raw.pt"
        quant = tracker.run_dir / "model.int8.ptz"
        raw.write_bytes(b"raw-model")
        quant.write_bytes(b"quant-model")
        tracker.log("test-log", event_type="log")
        return {"raw_model": raw, "quantized_model": quant}

    def _summary(self, tracker: RunTracker, *, bpb: float, bytes_total: int = 15_000_000) -> dict:
        return {
            "git_sha": "deadbeef",
            "final_prequant_val_loss": 2.0,
            "final_prequant_val_bpb": 1.2,
            "final_int8_zlib_roundtrip_exact_val_loss": 2.1,
            "final_int8_zlib_roundtrip_exact_val_bpb": bpb,
            "raw_model_bytes": 10,
            "quantized_model_bytes": 11,
            "bytes_code": 12,
            "bytes_total": bytes_total,
            "train_time_ms": 100,
            "roundtrip_eval_time_ms": 10,
            "selection_metric": "roundtrip_val_bpb",
            "selection_metric_value": bpb,
            "raw_model_path": str(tracker.run_dir / "model.raw.pt"),
            "quantized_model_path": str(tracker.run_dir / "model.int8.ptz"),
        }

    def test_promotes_first_best_and_writes_best_dir(self) -> None:
        tracker = self._tracker("run-1")
        tracker.finalize(summary=self._summary(tracker, bpb=1.20), artifact_paths=self._artifact_paths(tracker))

        best_json = json.loads((self.best_dir / "best.json").read_text(encoding="utf-8"))
        self.assertEqual(best_json["run_id"], "run-1")
        self.assertEqual(best_json["best_paths"]["quantized_model"], str(self.best_dir / "model.int8.ptz"))
        self.assertTrue((self.best_dir / "model.int8.ptz").is_file())
        self.assertTrue((self.best_dir / "train.log").is_file())
        self.assertTrue((tracker.run_dir / "model.raw.pt").is_file())
        self.assertTrue((tracker.run_dir / "summary.json").is_file())

    def test_nonbest_run_prunes_its_artifacts(self) -> None:
        tracker1 = self._tracker("run-1")
        tracker1.finalize(summary=self._summary(tracker1, bpb=1.20), artifact_paths=self._artifact_paths(tracker1))

        tracker2 = self._tracker("run-2")
        tracker2.finalize(summary=self._summary(tracker2, bpb=1.25), artifact_paths=self._artifact_paths(tracker2))

        self.assertFalse((tracker2.run_dir / "model.raw.pt").exists())
        self.assertFalse((tracker2.run_dir / "model.int8.ptz").exists())
        self.assertTrue((tracker2.run_dir / "train.log").is_file())
        self.assertTrue((tracker2.run_dir / "summary.json").is_file())

    def test_new_best_prunes_old_best_run_artifacts_but_keeps_metadata(self) -> None:
        tracker1 = self._tracker("run-1")
        tracker1.finalize(summary=self._summary(tracker1, bpb=1.20), artifact_paths=self._artifact_paths(tracker1))

        tracker2 = self._tracker("run-2")
        tracker2.finalize(summary=self._summary(tracker2, bpb=1.10), artifact_paths=self._artifact_paths(tracker2))

        best_json = json.loads((self.best_dir / "best.json").read_text(encoding="utf-8"))
        self.assertEqual(best_json["run_id"], "run-2")
        self.assertFalse((tracker1.run_dir / "model.raw.pt").exists())
        self.assertFalse((tracker1.run_dir / "model.int8.ptz").exists())
        self.assertTrue((tracker1.run_dir / "train.log").is_file())
        self.assertTrue((tracker1.run_dir / "summary.json").is_file())

    def test_over_limit_run_never_promotes(self) -> None:
        tracker = self._tracker("run-over")
        enriched = tracker.finalize(
            summary=self._summary(tracker, bpb=1.00, bytes_total=16_500_000),
            artifact_paths=self._artifact_paths(tracker),
        )
        self.assertFalse(enriched["promoted"])
        self.assertFalse((self.best_dir / "best.json").exists())

    def test_retains_top_three_run_artifacts_when_requested(self) -> None:
        trackers = [self._tracker(f"run-{idx}", retain_top_k=3) for idx in range(1, 5)]
        bpbs = [1.40, 1.30, 1.20, 1.10]
        for tracker, bpb in zip(trackers, bpbs, strict=True):
            tracker.finalize(summary=self._summary(tracker, bpb=bpb), artifact_paths=self._artifact_paths(tracker))

        self.assertFalse((trackers[0].run_dir / "model.raw.pt").exists())
        self.assertFalse((trackers[0].run_dir / "model.int8.ptz").exists())
        for tracker in trackers[1:]:
            self.assertTrue((tracker.run_dir / "model.raw.pt").is_file())
            self.assertTrue((tracker.run_dir / "model.int8.ptz").is_file())


if __name__ == "__main__":
    unittest.main()
