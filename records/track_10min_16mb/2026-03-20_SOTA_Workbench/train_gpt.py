"""
Competition workbench for Parameter Golf.

This script is intended to be run from inside this records folder while iterating on
standard-track submissions. It supports four variants:
- MODEL_VARIANT=mainline   -> 10L mixed int5/int6 with SmearGate + BigramHash
- MODEL_VARIANT=challenger -> 11L all-int6 with the same context features
- MODEL_VARIANT=leader_parity -> 11L all-int6 with the current public-leading stack defaults
- MODEL_VARIANT=frontier   -> 11L all-int6 with top-PR-style lexical context, SWA, and STE defaults

For local iteration it can fall back to repo-root helpers. When this folder is
snapshotted for a real record candidate, the vendored helper files in the same folder
take precedence so the candidate remains runnable from inside its own records directory.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lowbit_utils import (  # noqa: E402
    compress_quantized,
    decompress_quantized,
    dequantize_state_dict,
    export_state_dict_without_mtp,
    fake_quantize_tensor,
    load_export_state_dict,
    quantize_state_dict,
)
from optimizer_variants import Muon, NorMuon, zeropower_via_newtonschulz5  # noqa: E402
from validation_utils import (  # noqa: E402
    build_sentencepiece_luts,
    clip_doc_offsets_to_total_tokens,
    eval_val,
    eval_val_doc_sliding,
    eval_val_sliding,
    load_doc_offsets,
)

ROOT_TRAIN_GPT = HERE / "root_train_gpt_vendor.py"
if not ROOT_TRAIN_GPT.exists():
    ROOT_TRAIN_GPT = ROOT / "train_gpt.py"
_ROOT_SPEC = importlib.util.spec_from_file_location("pg_root_train_gpt", ROOT_TRAIN_GPT)
if _ROOT_SPEC is None or _ROOT_SPEC.loader is None:
    raise RuntimeError("failed to load root train_gpt.py")
pg_root = importlib.util.module_from_spec(_ROOT_SPEC)
sys.modules[_ROOT_SPEC.name] = pg_root
_ROOT_SPEC.loader.exec_module(pg_root)

CastedLinear = pg_root.CastedLinear
Block = pg_root.Block
DistributedTokenLoader = pg_root.DistributedTokenLoader
load_validation_tokens = pg_root.load_validation_tokens
restore_low_dim_params_to_fp32 = pg_root.restore_low_dim_params_to_fp32
Rotary = pg_root.Rotary
apply_rotary_emb = pg_root.apply_rotary_emb
CLIP_Q = pg_root.CLIP_Q
KEEP_FLOAT_FP32_NAME_PATTERNS = pg_root.KEEP_FLOAT_FP32_NAME_PATTERNS
KEEP_FLOAT_MAX_NUMEL = pg_root.KEEP_FLOAT_MAX_NUMEL
KEEP_FLOAT_STORE_DTYPE = pg_root.KEEP_FLOAT_STORE_DTYPE
PER_ROW_SCALE_DTYPE = pg_root.PER_ROW_SCALE_DTYPE
CONTROL_TENSOR_NAME_PATTERNS = pg_root.CONTROL_TENSOR_NAME_PATTERNS


def env_flag(name: str, default: str = "0") -> bool:
    return bool(int(os.environ.get(name, default)))


def parse_patterns(raw: str) -> tuple[str, ...]:
    return tuple(part for part in raw.split(",") if part)


def parse_bit_overrides(raw: str) -> tuple[tuple[str, int], ...]:
    out: list[tuple[str, int]] = []
    for part in parse_patterns(raw):
        pattern, _, bits_raw = part.partition(":")
        if not pattern or not bits_raw:
            raise ValueError(f"BIT_OVERRIDES entries must look like pattern:bits, got {part!r}")
        out.append((pattern, int(bits_raw)))
    return tuple(out)


def default_keep_float_patterns(num_layers: int) -> tuple[str, ...]:
    late = [f"blocks.{idx}.attn.c_k.weight" for idx in range(max(num_layers - 2, 0), num_layers)]
    return ("tok_emb.weight", *late)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def extract_args_config(args) -> dict[str, object]:
    config: dict[str, object] = {}
    for name in dir(args):
        if name.startswith("_"):
            continue
        value = getattr(args, name)
        if callable(value):
            continue
        config[name] = jsonable(value)
    return config


_FLASH_ATTN_RESOLVED = False
_FLASH_ATTN_FUNC = None
_FLASH_ATTN_ERROR = None


def get_flash_attn_func():
    global _FLASH_ATTN_RESOLVED, _FLASH_ATTN_FUNC, _FLASH_ATTN_ERROR
    if _FLASH_ATTN_RESOLVED:
        return _FLASH_ATTN_FUNC
    _FLASH_ATTN_RESOLVED = True
    for module_name in ("flash_attn_interface", "flash_attn"):
        try:
            module = __import__(module_name, fromlist=["flash_attn_func"])
        except ImportError as exc:
            _FLASH_ATTN_ERROR = exc
            continue
        fn = getattr(module, "flash_attn_func", None)
        if fn is not None:
            _FLASH_ATTN_FUNC = fn
            _FLASH_ATTN_ERROR = None
            return _FLASH_ATTN_FUNC
    return None


def resolve_attention_backend(attn_backend: str, *, device: torch.device, dtype: torch.dtype, head_dim: int) -> tuple[str, str]:
    if attn_backend == "sdpa":
        return "sdpa", "forced"
    flash_fn = get_flash_attn_func()
    reason = ""
    if flash_fn is None:
        reason = f"flash_attn_unavailable:{type(_FLASH_ATTN_ERROR).__name__ if _FLASH_ATTN_ERROR is not None else 'missing'}"
    elif device.type != "cuda":
        reason = "requires_cuda"
    elif dtype not in {torch.float16, torch.bfloat16}:
        reason = f"requires_half_precision:{dtype}"
    elif head_dim <= 0 or head_dim % 8 != 0 or head_dim > 256:
        reason = f"unsupported_head_dim:{head_dim}"
    else:
        return "fa3", "available"
    if attn_backend == "fa3":
        raise RuntimeError(f"ATTN_BACKEND=fa3 is not supported in this runtime ({reason})")
    return "sdpa", reason


def describe_attention_backend(attn_backend: str, *, device: torch.device, dtype: torch.dtype, head_dim: int) -> str:
    backend, reason = resolve_attention_backend(attn_backend, device=device, dtype=dtype, head_dim=head_dim)
    return f"requested={attn_backend} resolved={backend} reason={reason}"


def flash_attention_forward(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    flash_fn = get_flash_attn_func()
    if flash_fn is None:
        raise RuntimeError("flash attention function is unavailable")
    q_f = q.transpose(1, 2).contiguous()
    k_f = k.transpose(1, 2).contiguous()
    v_f = v.transpose(1, 2).contiguous()
    if k_f.shape[2] != q_f.shape[2]:
        repeats = q_f.shape[2] // k_f.shape[2]
        k_f = k_f.repeat_interleave(repeats, dim=2)
        v_f = v_f.repeat_interleave(repeats, dim=2)
    for kwargs in ({"causal": True}, {"softmax_scale": None, "causal": True}):
        try:
            return flash_fn(q_f, k_f, v_f, **kwargs).transpose(1, 2).contiguous()
        except TypeError:
            continue
    return flash_fn(q_f, k_f, v_f).transpose(1, 2).contiguous()


class WorkbenchCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        attn_backend: str,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.attn_backend = attn_backend

    def forward(self, x: Tensor, q_gain: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * q_gain.to(dtype=q.dtype)[None, :, None, None]
        backend, _ = resolve_attention_backend(self.attn_backend, device=x.device, dtype=q.dtype, head_dim=self.head_dim)
        if backend == "fa3":
            y = flash_attention_forward(q, k, v)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class WorkbenchBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int | None,
        mlp_kind: str,
        swiglu_hidden_mult: float,
        rope_base: float,
        attn_backend: str,
    ):
        super().__init__(dim, num_heads, num_kv_heads, mlp_mult, mlp_hidden, mlp_kind, swiglu_hidden_mult, rope_base)
        self.attn = WorkbenchCausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, attn_backend)


class Hyperparameters:
    def __init__(self) -> None:
        variant = os.environ.get("MODEL_VARIANT", "mainline").strip().lower()
        if variant not in {"mainline", "challenger", "leader_parity", "frontier"}:
            raise ValueError(f"MODEL_VARIANT must be 'mainline', 'challenger', 'leader_parity', or 'frontier', got {variant!r}")
        is_mainline = variant == "mainline"
        is_leader_parity = variant == "leader_parity"
        is_frontier = variant == "frontier"
        self.variant = variant
        self.data_path = os.environ.get("DATA_PATH", str(ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
        self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
        self.log_path = Path(os.environ.get("LOG_PATH", "train.log"))
        self.seed = int(os.environ.get("SEED", "1337"))
        self.vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))
        self.num_layers = int(os.environ.get("NUM_LAYERS", "10" if is_mainline else "11"))
        self.model_dim = int(os.environ.get("MODEL_DIM", "512"))
        self.num_heads = int(os.environ.get("NUM_HEADS", "8"))
        self.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", "4"))
        self.mlp_mult = int(os.environ.get("MLP_MULT", "3" if is_leader_parity else "2"))
        self.mlp_hidden = int(os.environ.get("MLP_HIDDEN", "1536"))
        self.mlp_kind = os.environ.get("MLP_KIND", "relu2")
        self.swiglu_hidden_mult = float(os.environ.get("SWIGLU_HIDDEN_MULT", "1.333333"))
        self.tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
        self.tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "0.005"))
        self.logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))
        self.rope_base = float(os.environ.get("ROPE_BASE", "50000.0" if is_frontier else "10000.0"))
        self.qk_gain_init = float(os.environ.get("QK_GAIN_INIT", "1.5"))
        self.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "786432"))
        self.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "2048"))
        self.eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", str(self.train_seq_len)))
        self.eval_stride = int(os.environ.get("EVAL_STRIDE", os.environ.get("DOC_SLIDE_STRIDE", "256" if is_frontier else "64")))
        default_eval_mode = "sliding" if (self.eval_stride > 0 or is_leader_parity) else "contiguous"
        self.eval_mode = os.environ.get("EVAL_MODE", default_eval_mode).strip().lower()
        if self.eval_mode not in {"contiguous", "sliding", "doc_sliding"}:
            raise ValueError(f"EVAL_MODE must be 'contiguous', 'sliding', or 'doc_sliding', got {self.eval_mode!r}")
        self.val_doc_offsets_path = Path(os.environ.get("VAL_DOC_OFFSETS_PATH", str(Path(self.data_path) / "fineweb_val_doc_offsets.npy")))
        self.val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", "524288"))
        self.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", "0"))
        self.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "200"))
        self.iterations = int(os.environ.get("ITERATIONS", "20000"))
        self.warmup_steps = int(os.environ.get("WARMUP_STEPS", "20"))
        self.warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", "3000"))
        self.max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))
        self.matrix_lr = float(os.environ.get("MATRIX_LR", "0.02" if is_mainline else "0.025"))
        self.scalar_lr = float(os.environ.get("SCALAR_LR", "0.02" if is_mainline else "0.025"))
        self.tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", "0.035" if is_leader_parity else "0.03"))
        self.muon_momentum = float(os.environ.get("MUON_MOMENTUM", "0.99"))
        self.muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.92"))
        self.muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "1500"))
        self.muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
        self.muon_weight_decay = float(
            os.environ.get("MUON_WEIGHT_DECAY", os.environ.get("MUON_WD", "0.04" if variant in {"mainline", "leader_parity", "frontier"} else "0.038"))
        )
        self.optimizer_variant = os.environ.get("OPTIMIZER_VARIANT", "muon")
        self.beta1 = float(os.environ.get("BETA1", "0.9"))
        self.beta2 = float(os.environ.get("BETA2", "0.95"))
        self.adam_eps = float(os.environ.get("ADAM_EPS", "1e-8"))
        adam_wd_default = os.environ.get("ADAM_WEIGHT_DECAY", os.environ.get("ADAM_WD", "0.04" if is_leader_parity else "0.01"))
        self.token_weight_decay = float(os.environ.get("TOKEN_WEIGHT_DECAY", adam_wd_default))
        self.scalar_weight_decay = float(os.environ.get("SCALAR_WEIGHT_DECAY", adam_wd_default))
        self.grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "0.3"))
        self.save_raw_model = env_flag("SAVE_RAW_MODEL", "0")
        self.export_mode = os.environ.get("EXPORT_MODE", "inline").strip().lower()
        if self.export_mode not in {"inline", "deferred", "both"}:
            raise ValueError(f"EXPORT_MODE must be 'inline', 'deferred', or 'both', got {self.export_mode!r}")
        self.serial_compressor = os.environ.get("SERIAL_COMPRESSOR", "zstd")
        self.zstd_level = int(os.environ.get("ZSTD_LEVEL", "22"))
        self.weight_quant_bits = int(os.environ.get("WEIGHT_QUANT_BITS", "6"))
        self.embed_quant_bits = int(os.environ.get("EMBED_QUANT_BITS", "16"))
        self.lowbit_name_patterns = parse_patterns(os.environ.get("LOWBIT_NAME_PATTERNS", ".attn.,.mlp."))
        self.keep_float_name_patterns = parse_patterns(
            os.environ.get("KEEP_FLOAT_NAME_PATTERNS", ",".join(default_keep_float_patterns(self.num_layers)))
        )
        self.grouped_int8_name_patterns = parse_patterns(os.environ.get("GROUPED_INT8_NAME_PATTERNS", ""))
        self.group_size = int(os.environ.get("GROUP_SIZE", "64"))
        default_overrides = ".mlp.:5" if is_mainline else ""
        self.bit_overrides = parse_bit_overrides(os.environ.get("BIT_OVERRIDES", default_overrides))
        self.fp16_embed_export = bool(int(os.environ.get("FP16_EMBED_EXPORT", "1")))
        self.lowbit_ste = bool(int(os.environ.get("LOWBIT_STE", "1" if is_frontier else "0")))
        self.ste_mirror_export = env_flag("STE_MIRROR_EXPORT", "1")
        self.lowbit_ste_start_frac = float(os.environ.get("LOWBIT_STE_START_FRAC", "0.80"))
        self.lowbit_ste_lr_scale = float(os.environ.get("LOWBIT_STE_LR_SCALE", "0.20"))
        self.lowbit_ste_name_patterns = parse_patterns(os.environ.get("LOWBIT_STE_NAME_PATTERNS", ".attn.,.mlp."))
        self.swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1" if variant in {"mainline", "leader_parity", "frontier"} else "0")))
        self.swa_start_frac = float(os.environ.get("SWA_START_FRAC", "0.50" if variant in {"leader_parity", "frontier"} else "0.85"))
        self.swa_every_steps = int(os.environ.get("SWA_EVERY_STEPS", os.environ.get("SWA_EVERY", "50")))
        self.lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
        self.lawa_ema_decay = float(os.environ.get("LAWA_EMA_DECAY", "0.995"))
        if self.swa_enabled and self.lawa_enabled:
            raise ValueError("SWA_ENABLED=1 cannot be combined with LAWA_ENABLED=1 in the workbench")
        self.mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", "0"))
        self.mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", "0.01"))
        self.use_bigram_hash = env_flag("USE_BIGRAM_HASH", "1" if variant in {"mainline", "leader_parity", "frontier"} else "0")
        self.use_smeargate = env_flag("USE_SMEARGATE", "1" if variant in {"mainline", "leader_parity", "frontier"} else "0")
        self.bigram_buckets = int(os.environ.get("BIGRAM_BUCKETS", os.environ.get("BIGRAM_VOCAB_SIZE", "2048" if is_leader_parity else "4096")))
        self.bigram_dim = int(os.environ.get("BIGRAM_DIM", "128"))
        self.smeargate_init = float(os.environ.get("SMEARGATE_INIT", "3.0"))
        self.attn_backend = os.environ.get("ATTN_BACKEND", "auto").strip().lower()
        if self.attn_backend not in {"auto", "sdpa", "fa3"}:
            raise ValueError(f"ATTN_BACKEND must be 'auto', 'sdpa', or 'fa3', got {self.attn_backend!r}")
        self.artifact_dir = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
        self.artifact_keep_top_k = max(1, int(os.environ.get("ARTIFACT_KEEP_TOP_K", "3")))
        self.compile_model = env_flag("COMPILE_MODEL", "1")
        self.compile_backend = os.environ.get("COMPILE_BACKEND", "default")


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"{message}\n")


class SmearGate(nn.Module):
    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), init_value, dtype=torch.float32))

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate).to(dtype=current.dtype)[None, None, :]
        return gate * current + (1.0 - gate) * previous


class BigramHash(nn.Module):
    def __init__(self, buckets: int, dim: int, model_dim: int):
        super().__init__()
        self.buckets = buckets
        self.table = nn.Embedding(buckets, dim)
        self.proj = CastedLinear(dim, model_dim, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        prev = torch.zeros_like(input_ids)
        prev[:, 1:] = input_ids[:, :-1]
        hashed = (prev * 92821 + input_ids) % self.buckets
        return self.proj(self.table(hashed))


class RecordGPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tie_embeddings = args.tie_embeddings
        self.logit_softcap = args.logit_softcap
        self.qat_active = False
        self.num_layers = args.num_layers
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.bigram = BigramHash(args.bigram_buckets, args.bigram_dim, args.model_dim) if args.use_bigram_hash else None
        self.smear = SmearGate(args.model_dim, args.smeargate_init) if args.use_smeargate else None
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))
        self.attn_scales = nn.Parameter(torch.ones(args.num_layers, args.model_dim, dtype=torch.float32))
        self.mlp_scales = nn.Parameter(torch.ones(args.num_layers, args.model_dim, dtype=torch.float32))
        self.resid_mixes = nn.Parameter(
            torch.stack(
                [torch.stack((torch.ones(args.model_dim), torch.zeros(args.model_dim))).float() for _ in range(args.num_layers)],
                dim=0,
            )
        )
        self.q_gains = nn.Parameter(torch.full((args.num_layers, args.num_heads), args.qk_gain_init, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                WorkbenchBlock(
                    args.model_dim,
                    args.num_heads,
                    args.num_kv_heads,
                    args.mlp_mult,
                    args.mlp_hidden,
                    args.mlp_kind,
                    args.swiglu_hidden_mult,
                    args.rope_base,
                    args.attn_backend,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.final_norm = pg_root.RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList([CastedLinear(args.model_dim, args.vocab_size, bias=False) for _ in range(args.mtp_num_heads)])
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.args.tied_embed_init_std)
            if self.bigram is not None:
                nn.init.normal_(self.bigram.table.weight, mean=0.0, std=self.args.tied_embed_init_std)
            proj_scale = 1.0 / math.sqrt(2.0 * self.args.num_layers)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
                    if getattr(module, "_zero_init", False):
                        module.weight.mul_(proj_scale)
            if self.lm_head is not None:
                self.lm_head.weight.mul_(proj_scale)

    def set_qat_active(self, enabled: bool) -> None:
        self.qat_active = enabled
        for module in self.modules():
            if isinstance(module, CastedLinear):
                module.fake_quant = enabled
                module.fake_quant_bits = 8

    def set_lowbit_ste(
        self,
        enabled: bool,
        patterns: tuple[str, ...],
        default_bits: int,
        bit_overrides: tuple[tuple[str, int], ...],
        *,
        mirror_export: bool = False,
        lowbit_name_patterns: tuple[str, ...] = (),
        keep_float_name_patterns: tuple[str, ...] = (),
    ) -> None:
        for name, module in self.named_modules():
            if isinstance(module, CastedLinear):
                weight_name = f"{name}.weight"
                module.fake_quant = enabled and any(pattern in name for pattern in patterns)
                module.fake_quant_bits = default_bits
                if mirror_export:
                    keep_float = any(pattern in weight_name for pattern in keep_float_name_patterns) or any(
                        pattern in weight_name for pattern in CONTROL_TENSOR_NAME_PATTERNS
                    )
                    base_bits = default_bits if any(pattern in weight_name for pattern in lowbit_name_patterns) else 8
                    resolved_bits = base_bits
                    for pattern, bits in bit_overrides:
                        if pattern in weight_name:
                            resolved_bits = bits
                            break
                    module.fake_quant = enabled and not keep_float and resolved_bits < 8
                    module.fake_quant_bits = resolved_bits
                    continue
                for pattern, bits in bit_overrides:
                    if pattern in name:
                        module.fake_quant_bits = bits
                        break

    def _embed_tokens(self, input_ids: Tensor) -> Tensor:
        emb_weight = fake_quantize_tensor(self.tok_emb.weight, num_bits=8, clip_q=CLIP_Q) if self.qat_active else self.tok_emb.weight
        tok = F.embedding(input_ids, emb_weight)
        current = tok if self.bigram is None else tok + self.bigram(input_ids)
        if self.smear is None:
            return current
        prev = torch.zeros_like(current)
        prev[:, 1:, :] = current[:, :-1, :]
        return self.smear(current, prev)

    def _encode(self, input_ids: Tensor) -> Tensor:
        x = F.rms_norm(self._embed_tokens(input_ids), (self.args.model_dim,))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](
                x,
                x0,
                resid_mix=self.resid_mixes[i],
                attn_scale=self.attn_scales[i],
                mlp_scale=self.mlp_scales[i],
                q_gain=self.q_gains[i],
            )
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            logical_idx = self.num_encoder_layers + i
            x = self.blocks[logical_idx](
                x,
                x0,
                resid_mix=self.resid_mixes[logical_idx],
                attn_scale=self.attn_scales[logical_idx],
                mlp_scale=self.mlp_scales[logical_idx],
                q_gain=self.q_gains[logical_idx],
            )
        return self.final_norm(x)

    def _project_logits(self, hidden: Tensor) -> Tensor:
        flat = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            head_weight = fake_quantize_tensor(self.tok_emb.weight, num_bits=8, clip_q=CLIP_Q) if self.qat_active else self.tok_emb.weight
            logits_proj = F.linear(flat, head_weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.view(*hidden.shape[:2], -1)

    def forward_per_position(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self._project_logits(self._encode(input_ids))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="none").view_as(target_ids)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self._encode(input_ids)
        logits = self._project_logits(hidden)
        main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")
        if self.training and self.args.mtp_num_heads > 0 and self.args.mtp_loss_weight > 0.0:
            aux = hidden.new_zeros(())
            count = 0
            for k, head in enumerate(self.mtp_heads):
                offset = k + 1
                if hidden.size(1) <= offset:
                    break
                mtp_logits = self.logit_softcap * torch.tanh(head(hidden[:, :-offset, :]) / self.logit_softcap)
                aux = aux + F.cross_entropy(
                    mtp_logits.reshape(-1, mtp_logits.size(-1)).float(),
                    target_ids[:, offset:].reshape(-1),
                    reduction="mean",
                )
                count += 1
            if count > 0:
                main_loss = main_loss + self.args.mtp_loss_weight * (aux / count)
        return main_loss


def make_logger(args: Hyperparameters) -> Logger:
    return Logger(args.log_path)


def log_config(log: Logger, args: Hyperparameters, code_bytes: int, world_size: int, grad_accum_steps: int) -> None:
    log.log(f"variant:{args.variant}")
    log.log(f"code_bytes:{code_bytes}")
    log.log(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log.log(
        f"layout:vocab={args.vocab_size} layers={args.num_layers} dim={args.model_dim} heads={args.num_heads} kv={args.num_kv_heads} mlp_hidden={args.mlp_hidden}"
    )
    log.log(
        f"train:batch_tokens={args.train_batch_tokens} train_seq_len={args.train_seq_len} eval_seq_len={args.eval_seq_len} eval_mode={args.eval_mode} eval_stride={args.eval_stride}"
    )
    log.log(
        f"optimizer:{args.optimizer_variant} matrix_lr={args.matrix_lr} scalar_lr={args.scalar_lr} tied_embed_lr={args.tied_embed_lr} muon_wd={args.muon_weight_decay} adam_wd={args.token_weight_decay}"
    )
    log.log(
        f"context_features:bigram={int(args.use_bigram_hash)} buckets={args.bigram_buckets} dim={args.bigram_dim} smeargate={int(args.use_smeargate)} rope_base={args.rope_base} swa={int(args.swa_enabled)} swa_start={args.swa_start_frac} swa_every={args.swa_every_steps}"
    )
    log.log(
        f"quant:compressor={args.serial_compressor} zstd_level={args.zstd_level} weight_bits={args.weight_quant_bits} embed_bits={args.embed_quant_bits} lowbit_ste={int(args.lowbit_ste)} ste_mirror_export={int(args.ste_mirror_export)} bit_overrides={args.bit_overrides} keep_float={args.keep_float_name_patterns}"
    )
    log.log(f"export_mode:{args.export_mode} artifact_dir:{args.artifact_dir} attn_backend:{args.attn_backend}")
    log.log(f"averaging:swa={int(args.swa_enabled)} lawa={int(args.lawa_enabled)} lawa_decay={args.lawa_ema_decay}")


def evaluate_model(
    *,
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    doc_offsets: Tensor | None,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "val_loss": None,
        "val_bpb": None,
        "mode_loss": None,
        "mode_bpb": None,
    }
    val_loss, val_bpb = eval_val(
        val_batch_size=args.val_batch_size,
        eval_seq_len=args.eval_seq_len,
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        grad_accum_steps=grad_accum_steps,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    metrics["val_loss"] = val_loss
    metrics["val_bpb"] = val_bpb
    if args.eval_mode == "sliding":
        mode_loss, mode_bpb = eval_val_sliding(
            eval_seq_len=args.eval_seq_len,
            eval_stride=args.eval_stride,
            model=model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        metrics["mode_loss"] = mode_loss
        metrics["mode_bpb"] = mode_bpb
    elif args.eval_mode == "doc_sliding":
        if doc_offsets is None:
            raise ValueError("doc_offsets are required for EVAL_MODE=doc_sliding")
        mode_loss, mode_bpb = eval_val_doc_sliding(
            eval_seq_len=args.eval_seq_len,
            eval_stride=args.eval_stride,
            model=model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            doc_offsets=doc_offsets,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        metrics["mode_loss"] = mode_loss
        metrics["mode_bpb"] = mode_bpb
    return metrics


def mode_metric_label(eval_mode: str) -> str:
    if eval_mode == "sliding":
        return "sliding_window"
    if eval_mode == "doc_sliding":
        return "doc_sliding_window"
    return "contiguous"


def write_frontier_manifest(
    path: Path,
    *,
    args: Hyperparameters,
    code_bytes: int,
    raw_checkpoint_path: Path,
    pre_metrics: dict[str, float | None],
    stage_timings_ms: dict[str, float],
) -> None:
    payload = {
        "workbench_train_gpt_path": str(Path(__file__).resolve()),
        "code_bytes": code_bytes,
        "raw_checkpoint_path": str(raw_checkpoint_path.resolve()),
        "dataset_path": args.data_path,
        "tokenizer_path": args.tokenizer_path,
        "val_doc_offsets_path": str(args.val_doc_offsets_path),
        "config": extract_args_config(args),
        "pre_roundtrip_metrics": pre_metrics,
        "stage_timings_ms": stage_timings_ms,
        "export_defaults": {
            "serial_compressor": args.serial_compressor,
            "zstd_level": args.zstd_level,
            "weight_quant_bits": args.weight_quant_bits,
            "embed_quant_bits": args.embed_quant_bits,
            "bit_overrides": list(args.bit_overrides),
            "keep_float_name_patterns": list(args.keep_float_name_patterns),
            "grouped_int8_name_patterns": list(args.grouped_int8_name_patterns),
            "group_size": args.group_size,
            "fp16_embed_export": args.fp16_embed_export,
            "eval_mode": args.eval_mode,
            "eval_stride": args.eval_stride,
        },
    }
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_lawa_shadow(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().float().clone() for name, tensor in state_dict.items()}


def update_lawa_shadow(shadow_state: dict[str, Tensor], state_dict: dict[str, Tensor], decay: float) -> None:
    for name, tensor in state_dict.items():
        current = tensor.detach().float()
        if name not in shadow_state:
            shadow_state[name] = current.clone()
            continue
        shadow_state[name].mul_(decay).add_(current, alpha=1.0 - decay)


def main() -> None:
    global zeropower_via_newtonschulz5

    args = Hyperparameters()
    logger = make_logger(args)
    code = Path(__file__).read_text(encoding="utf-8")
    code_bytes = len(code.encode("utf-8"))
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    if rank == 0:
        logger.log(code)
        logger.log(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout)
        log_config(logger, args, code_bytes, world_size, grad_accum_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    val_tokens = load_validation_tokens(args.val_files, max(args.train_seq_len, args.eval_seq_len))
    doc_offsets = None
    if args.eval_mode == "doc_sliding":
        doc_offsets = clip_doc_offsets_to_total_tokens(load_doc_offsets(args.val_doc_offsets_path), int(val_tokens.numel()))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    if rank == 0:
        logger.log(
            f"attention_backend:{describe_attention_backend(args.attn_backend, device=device, dtype=torch.bfloat16, head_dim=args.model_dim // args.num_heads)}"
        )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    base_model = RecordGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if args.compile_model else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    named_params = list(base_model.named_parameters())
    embed_names = {"tok_emb.weight"}
    if base_model.bigram is not None:
        embed_names.add("bigram.table.weight")
    matrix_params = [
        p
        for name, p in named_params
        if name not in embed_names | {"lm_head.weight"}
        and p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name not in embed_names | {"lm_head.weight"}
        and (p.ndim != 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    ]
    embed_params = [p for name, p in named_params if name in embed_names]
    optimizer_tok = torch.optim.AdamW(
        [{"params": embed_params, "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr, "weight_decay": args.token_weight_decay}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    muon_cls = Muon if args.optimizer_variant == "muon" else NorMuon
    optimizer_muon = muon_cls(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
        beta2=args.beta2,
        eps=args.adam_eps,
    ) if args.optimizer_variant == "normuon" else muon_cls(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": args.scalar_weight_decay}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizers.insert(
            1,
            torch.optim.AdamW(
                [{"params": [base_model.lm_head.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr, "weight_decay": args.token_weight_decay}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                fused=True,
            ),
        )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def progress_frac(step: int, elapsed_ms: float) -> float:
        frac_step = step / max(args.iterations, 1)
        frac_time = elapsed_ms / max(max_wallclock_ms, 1.0) if max_wallclock_ms is not None else 0.0
        return min(max(frac_step, frac_time), 1.0)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def apply_quant_modes(enabled: bool) -> None:
        if args.lowbit_ste:
            base_model.set_lowbit_ste(
                enabled,
                args.lowbit_ste_name_patterns,
                args.weight_quant_bits,
                args.bit_overrides,
                mirror_export=args.ste_mirror_export,
                lowbit_name_patterns=args.lowbit_name_patterns,
                keep_float_name_patterns=args.keep_float_name_patterns,
            )
        else:
            base_model.set_qat_active(enabled)

    lawa_state: dict[str, Tensor] | None = None

    def export_state() -> dict[str, Tensor]:
        if args.lawa_enabled and lawa_state is not None:
            return {name: tensor.detach().cpu().contiguous() for name, tensor in lawa_state.items()}
        return export_state_dict_without_mtp(base_model.state_dict())

    def build_quantized() -> tuple[dict[str, object], dict[str, int]]:
        export_state = export_state()
        quant_obj, quant_stats = quantize_state_dict(
            export_state,
            weight_quant_bits=args.weight_quant_bits,
            embed_quant_bits=args.embed_quant_bits,
            lowbit_name_patterns=args.lowbit_name_patterns,
            keep_float_name_patterns=args.keep_float_name_patterns,
            grouped_int8_name_patterns=args.grouped_int8_name_patterns,
            group_size=args.group_size,
            keep_float_max_numel=KEEP_FLOAT_MAX_NUMEL,
            keep_float_fp32_name_patterns=KEEP_FLOAT_FP32_NAME_PATTERNS,
            keep_float_store_dtype=KEEP_FLOAT_STORE_DTYPE,
            per_row_scale_dtype=PER_ROW_SCALE_DTYPE,
            clip_q=CLIP_Q,
            fp16_embed_export=args.fp16_embed_export,
            bit_overrides=args.bit_overrides,
        )
        return quant_obj, quant_stats

    def compress_export(quant_obj: dict[str, object]) -> tuple[bytes, int]:
        return compress_quantized(quant_obj, args.serial_compressor, zstd_level=args.zstd_level)

    initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
    initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
    if args.warmup_steps > 0:
        model.train()
        for _ in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    if args.lawa_enabled:
        lawa_state = build_lawa_shadow(export_state_dict_without_mtp(base_model.state_dict()))

    training_time_ms = 0.0
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    lawa_count = 1 if args.lawa_enabled else 0
    step = 0
    latest_val_loss = None
    latest_val_bpb = None
    latest_mode_loss = None
    latest_mode_bpb = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            metrics = evaluate_model(
                args=args,
                model=base_model,
                rank=rank,
                world_size=world_size,
                device=device,
                grad_accum_steps=grad_accum_steps,
                val_tokens=val_tokens,
                doc_offsets=doc_offsets,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            val_loss = metrics["val_loss"]
            val_bpb = metrics["val_bpb"]
            latest_val_loss = val_loss
            latest_val_bpb = val_bpb
            latest_mode_loss = metrics["mode_loss"]
            latest_mode_bpb = metrics["mode_bpb"]
            if rank == 0:
                logger.log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms")
            if args.eval_mode in {"sliding", "doc_sliding"} and latest_mode_loss is not None and latest_mode_bpb is not None:
                label = "sliding" if args.eval_mode == "sliding" else "doc_sliding"
                if rank == 0:
                    logger.log(f"step:{step}/{args.iterations} {label}_val_loss:{latest_mode_loss:.4f} {label}_val_bpb:{latest_mode_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        lowbit_phase = progress_frac(step, elapsed_ms) >= (args.lowbit_ste_start_frac if args.lowbit_ste else 2.0)
        if lowbit_phase:
            scale *= args.lowbit_ste_lr_scale
        apply_quant_modes(lowbit_phase)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        current_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = current_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if rank == 0 and args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            logger.log(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        if args.swa_enabled and step > 0 and step % max(args.swa_every_steps, 1) == 0 and progress_frac(step, approx_training_time_ms) >= args.swa_start_frac:
            state = {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items() if "mtp_heads" not in name}
            if swa_state is None:
                swa_state = state
                swa_count = 1
            else:
                swa_count += 1
                for name, tensor in state.items():
                    swa_state[name].mul_((swa_count - 1) / swa_count).add_(tensor, alpha=1.0 / swa_count)
        if args.lawa_enabled and lawa_state is not None:
            update_lawa_shadow(lawa_state, export_state_dict_without_mtp(base_model.state_dict()), args.lawa_ema_decay)
            lawa_count += 1
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if args.swa_enabled and swa_state is not None:
        load_export_state_dict(base_model, swa_state)
        if rank == 0:
            logger.log(f"loading_swa_state count:{swa_count}")
    elif args.lawa_enabled and lawa_state is not None and rank == 0:
        logger.log(f"using_lawa_state count:{lawa_count} decay:{args.lawa_ema_decay}")

    raw_model_path = Path("final_model.pt")
    quant_model_path = Path("final_model.ptz")
    frontier_candidate_path = Path("frontier_candidate.pt")
    frontier_manifest_path = Path("frontier_manifest.json")
    quant_stats = None
    stage_timings_ms: dict[str, float] = {"train": training_time_ms}
    quant_blob = None
    if rank == 0:
        logger.log("export_phase:start")
        if args.save_raw_model:
            logger.log("export_phase:raw_save_start")
            raw_save_t0 = time.perf_counter()
            torch.save(base_model.state_dict(), raw_model_path)
            stage_timings_ms["raw_save"] = 1000.0 * (time.perf_counter() - raw_save_t0)
            logger.log("export_phase:raw_save_done")
        if args.export_mode in {"deferred", "both"}:
            logger.log("export_phase:frontier_candidate_save_start")
            candidate_t0 = time.perf_counter()
            torch.save(export_state(), frontier_candidate_path)
            stage_timings_ms["frontier_candidate_save"] = 1000.0 * (time.perf_counter() - candidate_t0)
            write_frontier_manifest(
                frontier_manifest_path,
                args=args,
                code_bytes=code_bytes,
                raw_checkpoint_path=frontier_candidate_path,
                pre_metrics={
                    "val_loss": latest_val_loss,
                    "val_bpb": latest_val_bpb,
                    "mode_loss": latest_mode_loss,
                    "mode_bpb": latest_mode_bpb,
                    "mode_label": mode_metric_label(args.eval_mode),
                },
                stage_timings_ms=stage_timings_ms,
            )
            logger.log(f"frontier_manifest_path:{frontier_manifest_path.resolve()}")
            logger.log("export_phase:frontier_candidate_save_done")
        if args.export_mode == "deferred":
            logger.log("export_phase:deferred_exit")
        else:
            logger.log("export_phase:quantize_start")
            quantize_t0 = time.perf_counter()
            quant_obj, quant_stats = build_quantized()
            stage_timings_ms["quantize"] = 1000.0 * (time.perf_counter() - quantize_t0)
            compress_t0 = time.perf_counter()
            quant_blob, quant_raw_bytes = compress_export(quant_obj)
            stage_timings_ms["compress"] = 1000.0 * (time.perf_counter() - compress_t0)
            logger.log(
                f"export_phase:quantize_done bytes:{len(quant_blob)} payload:{quant_stats['int8_payload_bytes']} raw_bytes:{quant_raw_bytes}"
            )
            write_t0 = time.perf_counter()
            quant_model_path.write_bytes(quant_blob)
            stage_timings_ms["quant_write"] = 1000.0 * (time.perf_counter() - write_t0)
            logger.log("export_phase:quant_write_done")
    if distributed:
        dist.barrier()

    if args.export_mode != "deferred":
        if rank == 0:
            logger.log("export_phase:roundtrip_eval_start")
            reload_t0 = time.perf_counter()
        quant_state = decompress_quantized(quant_model_path.read_bytes(), args.serial_compressor)
        load_export_state_dict(base_model, dequantize_state_dict(quant_state))
        if rank == 0:
            stage_timings_ms["reload"] = 1000.0 * (time.perf_counter() - reload_t0)
        eval_t0 = time.perf_counter()
        q_metrics = evaluate_model(
            args=args,
            model=base_model,
            rank=rank,
            world_size=world_size,
            device=device,
            grad_accum_steps=grad_accum_steps,
            val_tokens=val_tokens,
            doc_offsets=doc_offsets,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        if rank == 0:
            stage_timings_ms["roundtrip_eval"] = 1000.0 * (time.perf_counter() - eval_t0)
        q_val_loss = q_metrics["val_loss"]
        q_val_bpb = q_metrics["val_bpb"]
        q_mode_loss = q_metrics["mode_loss"]
        q_mode_bpb = q_metrics["mode_bpb"]
        if rank == 0:
            raw_bytes = raw_model_path.stat().st_size if raw_model_path.exists() else 0
            total_bytes = quant_model_path.stat().st_size + code_bytes
            logger.log(f"Serialized model: {raw_bytes} bytes")
            logger.log(f"Serialized quantized model: {quant_model_path.stat().st_size} bytes (payload:{quant_stats['int8_payload_bytes']})")
            logger.log(f"Code size: {code_bytes} bytes")
            logger.log(f"Total submission size: {total_bytes} bytes")
            logger.log(
                "chosen_export_candidate "
                f"compressor:{args.serial_compressor} zstd_level:{args.zstd_level} "
                f"bit_overrides:{','.join(f'{pattern}:{bits}' for pattern, bits in args.bit_overrides)} "
                f"keep_float:{','.join(args.keep_float_name_patterns)} grouped_int8:{','.join(args.grouped_int8_name_patterns)} "
                f"eval_mode:{args.eval_mode} stride:{args.eval_stride} "
                f"averager:{'swa' if args.swa_enabled else 'lawa' if args.lawa_enabled else 'live'} "
                f"bytes_total:{total_bytes}"
            )
            for stage_name, duration_ms in stage_timings_ms.items():
                logger.log(f"stage_timing:{stage_name} ms:{duration_ms:.3f}")
            logger.log(f"final_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
            if q_mode_loss is not None and q_mode_bpb is not None:
                logger.log(
                    f"final_{mode_metric_label(args.eval_mode)}_exact val_loss:{q_mode_loss:.8f} val_bpb:{q_mode_bpb:.8f} stride:{args.eval_stride}"
                )
    elif rank == 0:
        for stage_name, duration_ms in stage_timings_ms.items():
            logger.log(f"stage_timing:{stage_name} ms:{duration_ms:.3f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
