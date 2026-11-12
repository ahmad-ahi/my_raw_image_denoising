#!/bin/bash
set -euo pipefail

EVAL_DIR="$1"

cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MODEL="$PWD/checkpoints/real_pairs_lowlr.obkQ28/unet_best_validation.pth"
NO_CORRECT="$PWD/checkpoints/eld_no_reference_correction.Bx6fDN/eval_no_correction.py"
PREFIX=$(basename "$EVAL_DIR")

exec > >(tee -a "$EVAL_DIR/run.log") 2>&1

echo "=== Started: $(date) ==="

python3 - <<'PY'
import torch

path = "checkpoints/real_pairs_lowlr.obkQ28/unet_best_validation.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

print("Selected epoch:", checkpoint.get("epoch"))
print("Validation L1:", checkpoint.get("val_l1"))
PY

echo "=== SID: reference-corrected ==="

for ratio in 100 250 300; do
    python3 test_denoise_sideld.py \
        --task "${PREFIX}_sid_r${ratio}" \
        --testset_type sid \
        --eval_ratio "$ratio" \
        --cp_dir "$MODEL" \
        2>&1 | tee "$EVAL_DIR/sid_r${ratio}.log"
done

echo "=== ELD: reference-corrected and uncorrected ==="

for correction in corrected uncorrected; do
    if [ "$correction" = corrected ]; then
        script="test_denoise_sideld.py"
    else
        script="$NO_CORRECT"
    fi

    for ratio in 100 200; do
        python3 "$script" \
            --task "${PREFIX}_eld_${correction}_r${ratio}" \
            --testset_type eld \
            --eval_ratio "$ratio" \
            --cp_dir "$MODEL" \
            2>&1 | tee "$EVAL_DIR/eld_${correction}_r${ratio}.log"
    done
done

echo
echo "===== RESULTS SUMMARY ====="

for log in \
    "$EVAL_DIR"/sid_r*.log \
    "$EVAL_DIR"/eld_corrected_r*.log \
    "$EVAL_DIR"/eld_uncorrected_r*.log
do
    echo "=== $(basename "$log") ==="
    grep 'valid_psnr:' "$log"
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $EVAL_DIR"
