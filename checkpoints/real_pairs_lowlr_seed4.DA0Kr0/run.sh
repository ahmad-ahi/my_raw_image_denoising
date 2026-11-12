#!/bin/bash
set -euo pipefail

cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export RUN_DIR="$1"

python3 "$RUN_DIR/train_real_pairs.py" \
  2>&1 | tee "$RUN_DIR/train.log"
