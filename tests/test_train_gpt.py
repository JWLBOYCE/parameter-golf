from __future__ import annotations

import unittest

import torch

from lowbit_utils import dequantize_state_dict, export_state_dict_without_mtp, load_export_state_dict, quant_qmax, quantize_state_dict
from train_gpt import CastedLinear, GPT, dequantize_state_dict_int8, quantize_state_dict_int8


class TrainGPTTest(unittest.TestCase):
    def make_model(self, **overrides) -> GPT:
        config = {
            "vocab_size": 1024,
            "num_layers": 6,
            "model_dim": 96,
            "num_heads": 6,
            "num_kv_heads": 3,
            "mlp_mult": 2,
            "mlp_hidden": None,
            "mlp_kind": "relu2",
            "swiglu_hidden_mult": 1.333333,
            "num_unique_blocks": 3,
            "tie_embeddings": True,
            "tied_embed_init_std": 0.005,
            "logit_softcap": 30.0,
            "rope_base": 10000.0,
            "qk_gain_init": 1.5,
            "mtp_num_heads": 0,
            "mtp_loss_weight": 0.01,
        }
        config.update(overrides)
        return GPT(**config)

    def test_num_unique_blocks_must_divide_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "must divide"):
            self.make_model(num_layers=10, num_unique_blocks=3)

    def test_invalid_mlp_kind_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported MLP_KIND"):
            self.make_model(mlp_kind="bad")

    def test_mlp_hidden_override_shapes(self) -> None:
        model = self.make_model(mlp_hidden=192)
        self.assertEqual(model.blocks[0].mlp.fc.weight.shape, (192, 96))

    def test_swiglu_forward_returns_scalar_loss(self) -> None:
        model = self.make_model(mlp_kind="swiglu")
        x = torch.randint(0, 1024, (2, 8), dtype=torch.int64)
        y = torch.randint(0, 1024, (2, 8), dtype=torch.int64)
        loss = model(x, y)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_forward_per_position_matches_mean_loss(self) -> None:
        model = self.make_model()
        x = torch.randint(0, 1024, (2, 8), dtype=torch.int64)
        y = torch.randint(0, 1024, (2, 8), dtype=torch.int64)
        loss = model(x, y)
        per_pos = model.forward_per_position(x, y)
        self.assertAlmostEqual(loss.item(), per_pos.mean().item(), places=5)

    def test_int8_quant_roundtrip_preserves_state_keys(self) -> None:
        model = self.make_model(mlp_kind="swiglu")
        quant_obj, _ = quantize_state_dict_int8(model.state_dict())
        restored = dequantize_state_dict_int8(quant_obj)
        self.assertEqual(set(restored), set(model.state_dict()))

    def test_selective_lowbit_roundtrip_preserves_shapes(self) -> None:
        model = self.make_model(mtp_num_heads=1)
        quant_obj, _ = quantize_state_dict(
            export_state_dict_without_mtp(model.state_dict()),
            weight_quant_bits=6,
            embed_quant_bits=16,
            lowbit_name_patterns=(".mlp.", ".attn.c_q.", ".attn.c_v.", ".attn.proj."),
            keep_float_name_patterns=("tok_emb.weight",),
            grouped_int8_name_patterns=(),
            group_size=64,
            keep_float_max_numel=65_536,
            keep_float_fp32_name_patterns=("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight"),
            keep_float_store_dtype=torch.float16,
            per_row_scale_dtype=torch.float16,
            clip_q=0.9999984,
            fp16_embed_export=True,
        )
        restored = dequantize_state_dict(quant_obj)
        self.assertNotIn("mtp_heads.0.weight", restored)
        for name, tensor in restored.items():
            self.assertEqual(tensor.shape, model.state_dict()[name].shape)

    def test_int5_quantization_path_is_supported(self) -> None:
        self.assertEqual(quant_qmax(5), 15)
        model = self.make_model()
        quant_obj, _ = quantize_state_dict(
            model.state_dict(),
            weight_quant_bits=6,
            embed_quant_bits=16,
            lowbit_name_patterns=(".attn.", ".mlp."),
            keep_float_name_patterns=("tok_emb.weight",),
            grouped_int8_name_patterns=(),
            group_size=64,
            keep_float_max_numel=0,
            keep_float_fp32_name_patterns=("attn_scale",),
            keep_float_store_dtype=torch.float16,
            per_row_scale_dtype=torch.float16,
            clip_q=0.9999984,
            fp16_embed_export=True,
            bit_overrides=((".mlp.", 5),),
        )
        mlp_keys = [name for name in quant_obj["qmeta"] if ".mlp." in name]
        self.assertTrue(mlp_keys)
        self.assertTrue(all(int(quant_obj["qmeta"][name]["bits"]) == 5 for name in mlp_keys))

    def test_embed_quant_bits_16_keeps_embedding_float(self) -> None:
        model = self.make_model()
        quant_obj, _ = quantize_state_dict(
            model.state_dict(),
            weight_quant_bits=6,
            embed_quant_bits=16,
            lowbit_name_patterns=(".mlp.",),
            keep_float_name_patterns=(),
            grouped_int8_name_patterns=(),
            group_size=64,
            keep_float_max_numel=65_536,
            keep_float_fp32_name_patterns=("attn_scale",),
            keep_float_store_dtype=torch.float16,
            per_row_scale_dtype=torch.float16,
            clip_q=0.9999984,
            fp16_embed_export=False,
        )
        self.assertIn("tok_emb.weight", quant_obj["passthrough"])

    def test_training_only_mtp_weights_are_excluded_from_export(self) -> None:
        model = self.make_model(mtp_num_heads=1)
        export_state = export_state_dict_without_mtp(model.state_dict())
        self.assertFalse(any("mtp_heads" in name for name in export_state))

    def test_load_export_state_dict_allows_missing_mtp(self) -> None:
        model = self.make_model(mtp_num_heads=1)
        export_state = export_state_dict_without_mtp(model.state_dict())
        load_export_state_dict(model, export_state)

    def test_selective_fake_quant_only_marks_matching_modules(self) -> None:
        model = self.make_model()
        model.set_lowbit_ste(True, (".mlp.",), 6)
        statuses = {name: module.fake_quant for name, module in model.named_modules() if isinstance(module, CastedLinear)}
        self.assertTrue(any(statuses.values()))
        self.assertTrue(all(value == (".mlp." in name) for name, value in statuses.items()))

    def test_qat_toggle_marks_all_casted_linear_modules(self) -> None:
        model = self.make_model(mlp_kind="swiglu")
        model.set_qat_active(True)
        self.assertTrue(all(m.fake_quant for m in model.modules() if isinstance(m, CastedLinear)))


if __name__ == "__main__":
    unittest.main()
