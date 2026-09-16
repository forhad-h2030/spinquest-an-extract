#!/usr/bin/env bash
# Flattens the raw June30_refit_2026 up/down EXP files into exp_tagged_data/.
# Works from any directory. Override RAW/TAG/PYROOT to flatten a different pair.
set -euo pipefail

OURS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYROOT="${PYROOT:-/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python}"
RAW="${RAW:-/Users/spin/SpinQuest-TSSA-Ana/data}"
TAG="${TAG:-June30_refit_2026}"   # raw files: exp_data_{up,down}_${TAG}.root

for SPIN in up down; do
  "$PYROOT" "$OURS/post_processing/flatten_exp_single.py" \
    --input "$RAW/exp_data_${SPIN}_${TAG}.root" \
    --output "$OURS/exp_tagged_data/exp_tgt_data_june30_${SPIN}.root"
done
