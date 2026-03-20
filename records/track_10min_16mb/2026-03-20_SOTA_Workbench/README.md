This folder is a local records-first workbench for current SOTA-style Parameter Golf runs.

It is not a scored submission yet. The purpose is to let you run the strongest public recipe families from inside a `/records/...` folder, then package the winning run into a clean final submission folder once you have logs and measured metrics.

Included variants:
- `MODEL_VARIANT=mainline`: `10L`, mixed `int5/int6`, `SmearGate`, `BigramHash`, `MuonWD`, `SWA`
- `MODEL_VARIANT=challenger`: `11L`, all-`int6`, defaulting to the simpler PR179-style recipe first, with optional `SmearGate` / `BigramHash` / `SWA` toggles if you want to layer them back in

Key mechanics in this workbench:
- sliding-window exact evaluation with `EVAL_SEQ_LEN` and `EVAL_STRIDE`
- mixed tensor-group precision via `BIT_OVERRIDES`, `KEEP_FLOAT_NAME_PATTERNS`, and `LOWBIT_NAME_PATTERNS`
- `zstd` and `zlib`
- exact serialize -> compress -> reload -> evaluate roundtrip
- `SmearGate`
- `BigramHash`
- orthogonal init with scaled output projections
- `Muon` / `NorMuon` with decoupled weight decay
- `SWA`
- optional selective `LOWBIT_STE`

Packaging helpers:
- `python3 experiments/snapshot_record_candidate.py <folder>` creates a portable candidate with vendored helper files
- `python3 experiments/package_record_submission.py <folder> --author ... --github-id ...` also copies `train.log` and fills `submission.json` / `README.md`
- `python3 experiments/tensor_group_sensitivity.py final_model.pt --target-total-bytes 16000000 --code-bytes <n>` searches mixed-bit tensor-group allocations by compressed size
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

Suggested 1xH100 challenger proxy:
```bash
./run_challenger_proxy.sh
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

RunPod checklist:
1. Clone the repo into `/workspace/parameter-golf`.
2. Download `sp1024` with `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
3. Run a root baseline smoke once.
4. `cd` into this folder and run the 1x smoke commands above.
5. Only after the script is stable, download the full dataset and rerun on `8xH100`.
6. When one variant is good enough, package it into a new dated submission folder with `python3 experiments/package_record_submission.py ...`.
7. If the result is PR-worthy, inspect the packaged folder and only then flatten dependencies further if you want a single-file artifact.
