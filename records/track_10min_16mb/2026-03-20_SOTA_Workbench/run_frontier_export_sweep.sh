#!/usr/bin/env bash
set -euo pipefail
python3 ../../../../experiments/export_frontier_sweep.py \
  "${CHECKPOINT_PATH:-frontier_candidate.pt}" \
  --manifest "${MANIFEST_PATH:-frontier_manifest.json}" \
  --keep-top-k "${KEEP_TOP_K:-3}" \
  --max-exact-candidates "${MAX_EXACT_CANDIDATES:-8}" \
  ${EXTRA_SWEEP_ARGS:-}
