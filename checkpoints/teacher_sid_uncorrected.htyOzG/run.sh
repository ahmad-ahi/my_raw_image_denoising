#!/bin/bash
set -euo pipefail

RUN_DIR="$1"
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MODEL="$PWD/checkpoints/real_pairs_teacher.AeHCkq/unet_best_validation.pth"
EVALUATOR="$PWD/checkpoints/eld_no_reference_correction.Bx6fDN/eval_no_correction.py"
PREFIX=$(basename "$RUN_DIR")

exec > >(tee -a "$RUN_DIR/run.log") 2>&1

for ratio in 100 250 300; do
    echo "=== Teacher / uncorrected SID x$ratio ==="

    python3 "$EVALUATOR" \
        --task "${PREFIX}_r${ratio}" \
        --testset_type sid \
        --eval_ratio "$ratio" \
        --cp_dir "$MODEL" \
        2>&1 | tee "$RUN_DIR/r${ratio}.log"
done

echo
echo "===== RESULTS SUMMARY ====="

for ratio in 100 250 300; do
    echo "=== SID x$ratio ==="
    grep 'valid_psnr:' "$RUN_DIR/r${ratio}.log"
done

echo "Results and logs: $RUN_DIR"
