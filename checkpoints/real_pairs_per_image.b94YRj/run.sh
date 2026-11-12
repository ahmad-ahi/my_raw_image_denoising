#!/bin/bash
set -euo pipefail

RUN_DIR="$1"
export RUN_DIR
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python3 "$RUN_DIR/analyze.py" 2>&1 | tee "$RUN_DIR/run.log"
