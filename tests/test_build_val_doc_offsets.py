from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.build_val_doc_offsets import build_doc_offsets, validate_offsets_against_validation_shards
from data.download_hf_docs_and_tokenize import PureByteTokenizer, write_datafile


class BuildValDocOffsetsTest(unittest.TestCase):
    def test_build_doc_offsets_matches_expected_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_path = root / "docs_selected.jsonl"
            docs_path.write_text('{"text":"ab"}\n{"text":"c"}\n', encoding="utf-8")
            tokenizer_path = root / "pure_byte.json"
            PureByteTokenizer().save_json(tokenizer_path)
            val_path = root / "fineweb_val_000000.bin"
            write_datafile(val_path, np.asarray([1, 10, 11, 1, 12], dtype=np.uint16))
            offsets = build_doc_offsets(
                docs_jsonl=docs_path,
                tokenizer_path=tokenizer_path,
                num_val_docs=2,
                batch_size=2,
            )
            self.assertTrue(np.array_equal(offsets, np.asarray([0, 3, 5], dtype=np.int64)))
            total = validate_offsets_against_validation_shards(offsets, str(root / "fineweb_val_*.bin"))
            self.assertEqual(total, 5)

    def test_validate_offsets_raises_on_shard_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            val_path = root / "fineweb_val_000000.bin"
            write_datafile(val_path, np.asarray([1, 2, 3], dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_offsets_against_validation_shards(np.asarray([0, 5], dtype=np.int64), str(root / "fineweb_val_*.bin"))


if __name__ == "__main__":
    unittest.main()
