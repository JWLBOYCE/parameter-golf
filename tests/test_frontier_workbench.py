from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - local env may not have torch
    torch = None
    F = None


def load_workbench():
    path = Path("records/track_10min_16mb/2026-03-20_SOTA_Workbench/train_gpt.py").resolve()
    spec = importlib.util.spec_from_file_location("pg_test_frontier_workbench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load workbench from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None, "torch is required for frontier workbench tests")
class FrontierWorkbenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_workbench()

    def make_args(self, **overrides):
        config = {
            "variant": "frontier",
            "vocab_size": 1024,
            "num_layers": 6,
            "model_dim": 96,
            "num_heads": 6,
            "num_kv_heads": 3,
            "mlp_mult": 2,
            "mlp_hidden": 192,
            "mlp_kind": "relu2",
            "swiglu_hidden_mult": 1.333333,
            "tie_embeddings": True,
            "tied_embed_init_std": 0.005,
            "logit_softcap": 30.0,
            "rope_base": 10000.0,
            "qk_gain_init": 1.5,
            "mtp_num_heads": 0,
            "mtp_loss_weight": 0.01,
            "use_bigram_hash": False,
            "use_smeargate": False,
            "bigram_buckets": 4096,
            "bigram_dim": 128,
            "smeargate_init": 3.0,
            "attn_backend": "auto",
            "weight_quant_bits": 6,
        }
        config.update(overrides)
        return SimpleNamespace(**config)

    def test_leader_parity_defaults_match_expected_stack(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "MODEL_VARIANT": "leader_parity",
            },
            clear=True,
        ):
            args = self.module.Hyperparameters()
        self.assertEqual(args.variant, "leader_parity")
        self.assertEqual(args.num_layers, 11)
        self.assertEqual(args.mlp_mult, 3)
        self.assertEqual(args.eval_mode, "sliding")
        self.assertEqual(args.eval_stride, 64)
        self.assertEqual(args.bigram_buckets, 2048)
        self.assertEqual(args.tied_embed_lr, 0.035)
        self.assertEqual(args.muon_weight_decay, 0.04)
        self.assertEqual(args.token_weight_decay, 0.04)
        self.assertTrue(args.swa_enabled)
        self.assertEqual(args.swa_start_frac, 0.50)
        self.assertEqual(args.swa_every_steps, 50)
        self.assertFalse(args.lowbit_ste)
        self.assertEqual(args.attn_backend, "auto")

    def test_swa_and_lawa_cannot_both_be_enabled(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "MODEL_VARIANT": "leader_parity",
                "SWA_ENABLED": "1",
                "LAWA_ENABLED": "1",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                self.module.Hyperparameters()

    def test_attn_backend_auto_falls_back_to_sdpa(self) -> None:
        backend, reason = self.module.resolve_attention_backend("auto", device=torch.device("cpu"), dtype=torch.float32, head_dim=64)
        self.assertEqual(backend, "sdpa")
        self.assertTrue(reason)

    def test_ste_mirror_export_only_marks_sub_int8_modules(self) -> None:
        model = self.module.RecordGPT(self.make_args())
        model.set_lowbit_ste(
            True,
            (".attn.", ".mlp."),
            6,
            ((".mlp.", 5),),
            mirror_export=True,
            lowbit_name_patterns=(".attn.", ".mlp."),
            keep_float_name_patterns=("tok_emb.weight", "blocks.4.attn.c_k.weight", "blocks.5.attn.c_k.weight"),
        )
        statuses = {
            name: (module.fake_quant, module.fake_quant_bits)
            for name, module in model.named_modules()
            if isinstance(module, self.module.CastedLinear)
        }
        self.assertTrue(statuses["blocks.0.attn.c_q"][0])
        self.assertEqual(statuses["blocks.0.attn.c_q"][1], 6)
        self.assertTrue(statuses["blocks.0.mlp.fc"][0])
        self.assertEqual(statuses["blocks.0.mlp.fc"][1], 5)

    def test_lawa_shadow_updates_as_ema(self) -> None:
        shadow = self.module.build_lawa_shadow({"w": torch.tensor([1.0, 3.0])})
        self.module.update_lawa_shadow(shadow, {"w": torch.tensor([5.0, 7.0])}, 0.75)
        self.assertTrue(torch.allclose(shadow["w"], torch.tensor([2.0, 4.0])))

    def test_flash_attention_forward_matches_sdpa_with_fake_runner(self) -> None:
        q = torch.randn(2, 4, 5, 8)
        k = torch.randn(2, 2, 5, 8)
        v = torch.randn(2, 2, 5, 8)

        def fake_flash(q_f, k_f, v_f, causal=True, softmax_scale=None):
            q_sdpa = q_f.transpose(1, 2).contiguous()
            k_sdpa = k_f.transpose(1, 2).contiguous()
            v_sdpa = v_f.transpose(1, 2).contiguous()
            if k_sdpa.shape[1] != q_sdpa.shape[1]:
                repeats = q_sdpa.shape[1] // k_sdpa.shape[1]
                k_sdpa = k_sdpa.repeat_interleave(repeats, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(repeats, dim=1)
            out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
            return out.transpose(1, 2).contiguous()

        old_resolved = self.module._FLASH_ATTN_RESOLVED
        old_func = self.module._FLASH_ATTN_FUNC
        old_error = self.module._FLASH_ATTN_ERROR
        try:
            self.module._FLASH_ATTN_RESOLVED = True
            self.module._FLASH_ATTN_FUNC = fake_flash
            self.module._FLASH_ATTN_ERROR = None
            expected = fake_flash(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            actual = self.module.flash_attention_forward(q, k, v)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        finally:
            self.module._FLASH_ATTN_RESOLVED = old_resolved
            self.module._FLASH_ATTN_FUNC = old_func
            self.module._FLASH_ATTN_ERROR = old_error


if __name__ == "__main__":
    unittest.main()
