#!/usr/bin/env bash
set -euo pipefail
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
MODEL_VARIANT=leader_parity \
ATTN_BACKEND="${ATTN_BACKEND:-auto}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py 2>&1 | tee train.log
