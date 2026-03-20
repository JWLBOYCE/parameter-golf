This folder is a local records-first workbench for current SOTA-style Parameter Golf runs.

It is not a scored submission yet. The purpose is to let you run the strongest public recipe families from inside a `/records/...` folder, then package the winning run into a clean final submission folder once you have logs and measured metrics.

Included variants:
- `MODEL_VARIANT=mainline`: `10L`, mixed `int5/int6`, `SmearGate`, `BigramHash`, `MuonWD`, `SWA`
- `MODEL_VARIANT=challenger`: `11L`, all-`int6`, defaulting to the simpler PR179-style recipe first, with optional `SmearGate` / `BigramHash` / `SWA` toggles if you want to layer them back in
- `MODEL_VARIANT=leader_parity`: `11L`, all-`int6`, `MLP 3x`, `SmearGate`, `BigramHash(2048x128)`, `MuonWD=0.04`, `AdamWD=0.04`, `SWA(50,start=0.50)`, and `sliding stride=64` to mirror the current leading public stack
- `MODEL_VARIANT=frontier`: `11L`, all-`int6`, `SmearGate`, `BigramHash(4096x128)`, `LOWBIT_STE`, `RoPE50K`, `MuonWD=0.04`, and `SWA(50, start=0.50)` to mirror the stronger current PR stack more closely

Key mechanics in this workbench:
- `EVAL_MODE=contiguous|sliding|doc_sliding`
- sliding-window exact evaluation with `EVAL_SEQ_LEN` and `EVAL_STRIDE`
- doc-aware sliding eval with `VAL_DOC_OFFSETS_PATH`
- mixed tensor-group precision via `BIT_OVERRIDES`, `KEEP_FLOAT_NAME_PATTERNS`, and `LOWBIT_NAME_PATTERNS`
- `zstd` and `zlib`
- exact serialize -> compress -> reload -> evaluate roundtrip
- `EXPORT_MODE=inline|deferred|both` with `frontier_candidate.pt` + `frontier_manifest.json`
- `SmearGate`
- `BigramHash`
- orthogonal init with scaled output projections
- `Muon` / `NorMuon` with decoupled weight decay
- `SWA`
- optional `LAWA_ENABLED=1` / `LAWA_EMA_DECAY=0.995` challenger path, mutually exclusive with `SWA`
- optional selective `LOWBIT_STE`
- `STE_MIRROR_EXPORT=1` so fake quant tracks the actual export bit map
- optional `ATTN_BACKEND=auto|sdpa|fa3`

Packaging helpers:
- `python3 experiments/snapshot_record_candidate.py <folder>` creates a portable candidate with vendored helper files
- `python3 experiments/package_record_submission.py <folder> --author ... --github-id ...` also copies `train.log` and fills `submission.json` / `README.md`
- `python3 experiments/tensor_group_sensitivity.py final_model.pt --target-total-bytes 16000000 --code-bytes <n>` searches mixed-bit tensor-group allocations by compressed size
- `python3 data/build_val_doc_offsets.py --docs-jsonl data/docs_selected.jsonl --tokenizer-path data/tokenizers/fineweb_1024_bpe.model --val-files 'data/datasets/fineweb10B_sp1024/fineweb_val_*.bin' --output data/datasets/fineweb10B_sp1024/fineweb_val_doc_offsets.npy` generates doc boundaries for `EVAL_MODE=doc_sliding`
- `python3 experiments/export_frontier_sweep.py frontier_candidate.pt --manifest frontier_manifest.json --eval-modes doc_sliding,sliding --strides 64,128 --bit-overrides '' --bit-overrides '.mlp.:5'` prescreens legal export candidates and exact-evaluates the best ones
- `python3 experiments/log_workbench_run.py train.log --run-name ... --train-shards ... --public-best-bpb ... --public-best-ref ... --changes ... --summary-note ...` parses a finished workbench run into the in-repo competition ledger

Mainline command:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=mainline \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train.log
```

Challenger command:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=challenger \
MUON_WEIGHT_DECAY=0.038 \
USE_BIGRAM_HASH=0 \
USE_SMEARGATE=0 \
SWA_ENABLED=0 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py | tee train.log
```

Leader-parity command:
```bash
./run_leader_parity.sh
```

Frontier command:
```bash
./run_frontier.sh
```

The frontier profile also accepts aliases commonly used in the current PRs:
- `MUON_WD` as an alias for `MUON_WEIGHT_DECAY`
- `ADAM_WEIGHT_DECAY` / `ADAM_WD`
- `SWA_EVERY` as an alias for `SWA_EVERY_STEPS`
- `BIGRAM_VOCAB_SIZE` as an alias for `BIGRAM_BUCKETS`
- `DOC_SLIDE_STRIDE` as an alias for `EVAL_STRIDE`
- `ZSTD_LEVEL` for compression tuning

Suggested 1xH100 challenger proxy:
```bash
./run_challenger_proxy.sh
```

Suggested 1xH100 leader-parity proxy:
```bash
./run_leader_parity_proxy.sh
```

Suggested 1xH100 frontier proxy:
```bash
./run_frontier_proxy.sh
```

Suggested 1xH100 smoke:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=mainline \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=400 \
VAL_LOSS_EVERY=0 \
EVAL_STRIDE=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train.log
```

Suggested 1xH100 sliding smoke:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=challenger \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=400 \
VAL_LOSS_EVERY=0 \
EVAL_STRIDE=64 \
COMPILE_MODEL=0 \
USE_BIGRAM_HASH=0 \
USE_SMEARGATE=0 \
SWA_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train.log
```

Suggested deferred-export smoke:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=challenger \
EXPORT_MODE=deferred \
COMPILE_MODEL=0 \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=400 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train.log
```

Suggested doc-aware sliding smoke after generating offsets:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=frontier \
EVAL_MODE=doc_sliding \
EVAL_STRIDE=64 \
EXPORT_MODE=deferred \
COMPILE_MODEL=0 \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=400 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train.log
```

Suggested 1xH100 selective-STE smoke:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=mainline \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=400 \
VAL_LOSS_EVERY=0 \
LOWBIT_STE=1 \
LOWBIT_STE_NAME_PATTERNS=.attn.,.mlp. \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train.log
```

After a deferred run, sweep the export frontier locally:
```bash
./run_frontier_export_sweep.sh
```

For `leader_parity`, the sweep defaults intentionally stay narrow:
- `eval_mode=sliding`
- `stride=64`
- `bit_overrides in {'', '.mlp.:5'}`
- `keep_float in {'tok_emb.weight', 'tok_emb.weight + late c_k'}`
- `zstd_level in {19,22}`

Suggested LAWA challenger proxy:
```bash
NCCL_IB_DISABLE=1 \
MODEL_VARIANT=leader_parity \
SWA_ENABLED=0 \
LAWA_ENABLED=1 \
LAWA_EMA_DECAY=0.995 \
COMPILE_MODEL=0 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=2000 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py | tee train.log
```

RunPod checklist:
1. Clone the repo into `/workspace/parameter-golf`.
2. Download `sp1024` with `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
3. Run a root baseline smoke once.
4. `cd` into this folder and run the 1x smoke commands above.
5. Only after the script is stable, download the full dataset and rerun on `8xH100`.
6. When one variant is good enough, package it into a new dated submission folder with `python3 experiments/package_record_submission.py ...`.
7. If the result is PR-worthy, inspect the packaged folder and only then flatten dependencies further if you want a single-file artifact.
