from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - local env may not have torch
    torch = None
    nn = object

if torch is not None:
    from validation_utils import build_doc_sliding_window_specs, clip_doc_offsets_to_total_tokens, eval_val_doc_sliding, eval_val_sliding


if torch is not None:
    class ToyModel(nn.Module):
        def forward_per_position(self, input_ids, target_ids):
            return (input_ids.float() * 10.0 + target_ids.float()) / 100.0

        def forward(self, input_ids, target_ids):
            return self.forward_per_position(input_ids, target_ids).mean()


@unittest.skipUnless(torch is not None, "torch is required for validation_utils tests")
class ValidationUtilsTest(unittest.TestCase):
    def test_clip_doc_offsets_truncates_partial_tail_doc(self) -> None:
        offsets = torch.tensor([0, 4, 9], dtype=torch.int64)
        clipped = clip_doc_offsets_to_total_tokens(offsets, 7)
        self.assertTrue(torch.equal(clipped, torch.tensor([0, 4, 7], dtype=torch.int64)))

    def test_build_doc_sliding_window_specs_covers_each_valid_target_once(self) -> None:
        offsets = torch.tensor([0, 5, 10], dtype=torch.int64)
        specs = build_doc_sliding_window_specs(doc_offsets=offsets, eval_seq_len=4, eval_stride=2, total_tokens=10)
        covered: list[int] = []
        for window_start, _, score_from, score_count in specs:
            covered.extend(range(window_start + 1 + score_from, window_start + 1 + score_from + score_count))
        self.assertEqual(covered, [1, 2, 3, 4, 6, 7, 8, 9])

    def test_doc_sliding_equals_raw_sliding_for_single_doc(self) -> None:
        model = ToyModel()
        val_tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        offsets = torch.tensor([0, 7], dtype=torch.int64)
        lut = torch.ones((16,), dtype=torch.int16)
        flags = torch.zeros((16,), dtype=torch.bool)
        raw_loss, raw_bpb = eval_val_sliding(
            eval_seq_len=4,
            eval_stride=2,
            model=model,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            val_tokens=val_tokens,
            base_bytes_lut=lut,
            has_leading_space_lut=flags,
            is_boundary_token_lut=flags,
        )
        doc_loss, doc_bpb = eval_val_doc_sliding(
            eval_seq_len=4,
            eval_stride=2,
            model=model,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            val_tokens=val_tokens,
            doc_offsets=offsets,
            base_bytes_lut=lut,
            has_leading_space_lut=flags,
            is_boundary_token_lut=flags,
        )
        self.assertAlmostEqual(raw_loss, doc_loss, places=6)
        self.assertAlmostEqual(raw_bpb, doc_bpb, places=6)

    def test_doc_sliding_differs_from_raw_sliding_on_multi_doc_stream(self) -> None:
        model = ToyModel()
        val_tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
        offsets = torch.tensor([0, 3, 8], dtype=torch.int64)
        lut = torch.ones((16,), dtype=torch.int16)
        flags = torch.zeros((16,), dtype=torch.bool)
        raw_loss, _ = eval_val_sliding(
            eval_seq_len=4,
            eval_stride=2,
            model=model,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            val_tokens=val_tokens,
            base_bytes_lut=lut,
            has_leading_space_lut=flags,
            is_boundary_token_lut=flags,
        )
        doc_loss, _ = eval_val_doc_sliding(
            eval_seq_len=4,
            eval_stride=2,
            model=model,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            val_tokens=val_tokens,
            doc_offsets=offsets,
            base_bytes_lut=lut,
            has_leading_space_lut=flags,
            is_boundary_token_lut=flags,
        )
        self.assertNotEqual(raw_loss, doc_loss)


if __name__ == "__main__":
    unittest.main()
