from __future__ import annotations

import json
import unittest
from pathlib import Path


class TokenizerSpecTests(unittest.TestCase):
    def test_expected_sweep_candidates_exist(self) -> None:
        payload = json.loads(Path("data/tokenizer_specs.json").read_text(encoding="utf-8"))
        specs = {entry["dataset_suffix"]: entry["vocab_size"] for entry in payload["tokenizers"]}
        self.assertEqual(
            specs,
            {
                "byte260": 260,
                "sp512": 512,
                "sp768": 768,
                "sp1024": 1024,
                "sp1536": 1536,
                "sp2048": 2048,
                "sp4096": 4096,
            },
        )


if __name__ == "__main__":
    unittest.main()
