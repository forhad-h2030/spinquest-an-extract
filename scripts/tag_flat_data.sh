#!/usr/bin/env bash
# Tags already-flattened up/down EXP files (spin-preserving) with the
# 20-seed production ensemble, hadds into exp_tagged_data/.
# Usage: scripts/tag_flat_data.sh <flat_up.root> <flat_down.root>
set -euo pipefail

FLAT_UP="${1:?usage: tag_flat_data.sh <flat_up.root> <flat_down.root>}"
FLAT_DOWN="${2:?usage: tag_flat_data.sh <flat_up.root> <flat_down.root>}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIN="$ROOT_DIR/exp_tagged_data"
PYROOT="${PYROOT:-/opt/homebrew/Caskroom/miniconda/base/envs/root_env/bin/python}"
HADD="${HADD:-/opt/homebrew/bin/hadd}"
SCRATCH="${SCRATCH:-/tmp}"
MASS_MIN="${MASS_MIN:-1.0}"
MASS_MAX="${MASS_MAX:-8.0}"

CKPT_DIR="$ROOT_DIR/checkpoints/multiclass_12_feat_sep05_newjpsi_20seed"
RUN_NAME="multiclass_dnn_12feat_classweighted_20seed"
CKPTS=()
while IFS= read -r -d '' f; do CKPTS+=("$f"); done \
  < <(find "$CKPT_DIR" -name "${RUN_NAME}.best.pth" -print0 | sort -z)
[[ ${#CKPTS[@]} -gt 0 ]] || { echo "[ERROR] no checkpoints found under $CKPT_DIR" >&2; exit 1; }
echo "[INFO] ensemble: ${#CKPTS[@]} seed(s)"

TAGGED="$TIN/exp_tagged_tgt_data_multiclass_12_feat_sep05_newjpsi_20seed.root"

echo "[1/2] tagging (spin up/down separately)"
"$PYROOT" -m an.tag \
  --input "$FLAT_UP" --output "$SCRATCH/exp_tagged_up.root" \
  --tree tree --mass-min "$MASS_MIN" --mass-max "$MASS_MAX" --spin up --ckpt "${CKPTS[@]}"
"$PYROOT" -m an.tag \
  --input "$FLAT_DOWN" --output "$SCRATCH/exp_tagged_down.root" \
  --tree tree --mass-min "$MASS_MIN" --mass-max "$MASS_MAX" --spin down --ckpt "${CKPTS[@]}"

echo "[2/2] hadd -> $TAGGED"
"$HADD" -f "$TAGGED" "$SCRATCH/exp_tagged_up.root" "$SCRATCH/exp_tagged_down.root"
rm -f "$SCRATCH/exp_tagged_up.root" "$SCRATCH/exp_tagged_down.root"

echo "[DONE] $TAGGED"
