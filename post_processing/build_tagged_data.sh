#!/usr/bin/env bash
# Flattens raw up/down EXP files and tags them with the 12-feat production ensemble.
# Usage: bash post_processing/build_tagged_data.sh [raw_up.root] [raw_down.root]
#
# Output: exp_tagged_data/exp_tagged_tgt_data_20063581.root
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"
TIN="$ROOT_DIR/exp_tagged_data"
PYROOT="${PYROOT:-/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python}"
HADD="${HADD:-/opt/homebrew/bin/hadd}"
CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/checkpoints/outputs_12feat_adamw_ep120_maxbest80_20063581/multiclass_dnn_12feat_adamw_ep120_nogapstop_20seed}"
MASS_MIN="${MASS_MIN:-0.0}"
MASS_MAX="${MASS_MAX:-8.0}"
OUT_NAME="${OUT_NAME:-exp_tagged_tgt_data_20063581}"
SCRATCH="${SCRATCH:-/tmp}"
RAW_UP="${1:-/Users/spin/SpinQuest-TSSA-Ana/data/exp_data_up_June30_refit_2026.root}"
RAW_DOWN="${2:-/Users/spin/SpinQuest-TSSA-Ana/data/exp_data_down_June30_refit_2026.root}"

mkdir -p "$TIN"

echo "[1/3] flattening up"
"$PYROOT" "$HERE/flatten_exp_single.py" --input "$RAW_UP"   --output "$TIN/exp_tgt_data_june30_up.root"

echo "[1/3] flattening down"
"$PYROOT" "$HERE/flatten_exp_single.py" --input "$RAW_DOWN" --output "$TIN/exp_tgt_data_june30_down.root"

echo "[2/3] tagging up"
"$PYROOT" "$HERE/tag_exp_12feat.py" \
  --input "$TIN/exp_tgt_data_june30_up.root" \
  --output "$SCRATCH/${OUT_NAME}_up.root" \
  --spin up --mass-min "$MASS_MIN" --mass-max "$MASS_MAX" \
  --ckpt-dir "$CKPT_DIR"

echo "[2/3] tagging down"
"$PYROOT" "$HERE/tag_exp_12feat.py" \
  --input "$TIN/exp_tgt_data_june30_down.root" \
  --output "$SCRATCH/${OUT_NAME}_down.root" \
  --spin down --mass-min "$MASS_MIN" --mass-max "$MASS_MAX" \
  --ckpt-dir "$CKPT_DIR"

echo "[3/3] combining up+down"
"$HADD" -f "$TIN/${OUT_NAME}.root" \
  "$SCRATCH/${OUT_NAME}_up.root" "$SCRATCH/${OUT_NAME}_down.root"
rm -f "$SCRATCH/${OUT_NAME}_up.root" "$SCRATCH/${OUT_NAME}_down.root"

echo "[DONE] $TIN/${OUT_NAME}.root"
