#!/bin/bash
set -euo pipefail

EVAL_DIR="$1"
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

SONY="$PWD/checkpoints/sonya7s2.pth"
BLEND="$PWD/checkpoints/sony_blend50.GHX2q1/unet_blend.pth"
REAL="$PWD/checkpoints/real_pairs.wEZGga/unet_best_validation.pth"
NO_CORRECT="$PWD/checkpoints/eld_no_reference_correction.Bx6fDN/eval_no_correction.py"

PREFIX=$(basename "$EVAL_DIR")
exec > >(tee -a "$EVAL_DIR/run.log") 2>&1

echo "=== Started: $(date) ==="

for model in sony blend real_pairs; do
    case "$model" in
        sony) checkpoint="$SONY" ;;
        blend) checkpoint="$BLEND" ;;
        real_pairs) checkpoint="$REAL" ;;
    esac

    for correction in corrected uncorrected; do
        if [ "$correction" = corrected ]; then
            script="test_denoise_sideld.py"
        else
            script="$NO_CORRECT"
        fi

        for ratio in 100 200; do
            echo "=== $model / $correction / ELD x$ratio ==="

            python3 "$script" \
                --task "${PREFIX}_${model}_${correction}_r${ratio}" \
                --testset_type eld \
                --eval_ratio "$ratio" \
                --cp_dir "$checkpoint" \
                2>&1 | tee "$EVAL_DIR/${model}_${correction}_r${ratio}.log"
        done
    done
done

echo
echo "===== RESULTS SUMMARY ====="

for model in sony blend real_pairs; do
    for correction in corrected uncorrected; do
        for ratio in 100 200; do
            echo "=== $model / $correction / x$ratio ==="
            grep 'valid_psnr:' \
                "$EVAL_DIR/${model}_${correction}_r${ratio}.log"
        done
    done
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $EVAL_DIR"
