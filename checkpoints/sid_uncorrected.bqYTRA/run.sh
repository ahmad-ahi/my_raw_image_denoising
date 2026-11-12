#!/bin/bash
set -euo pipefail

EVAL_DIR="$1"
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EVALUATOR="$PWD/checkpoints/eld_no_reference_correction.Bx6fDN/eval_no_correction.py"
SONY="$PWD/checkpoints/sonya7s2.pth"
SEED3="$PWD/checkpoints/real_pairs_lowlr.obkQ28/unet_best_validation.pth"
SEED4="$PWD/checkpoints/real_pairs_lowlr_seed4.DA0Kr0/unet_best_validation.pth"

PREFIX=$(basename "$EVAL_DIR")
exec > >(tee -a "$EVAL_DIR/run.log") 2>&1

echo "=== Started: $(date) ==="

for model in sony seed3 seed4; do
    case "$model" in
        sony)  checkpoint="$SONY" ;;
        seed3) checkpoint="$SEED3" ;;
        seed4) checkpoint="$SEED4" ;;
    esac

    for ratio in 100 250 300; do
        echo "=== $model / uncorrected SID x$ratio ==="

        python3 "$EVALUATOR" \
            --task "${PREFIX}_${model}_r${ratio}" \
            --testset_type sid \
            --eval_ratio "$ratio" \
            --cp_dir "$checkpoint" \
            2>&1 | tee "$EVAL_DIR/${model}_r${ratio}.log"
    done
done

echo
echo "===== RESULTS SUMMARY ====="

for model in sony seed3 seed4; do
    for ratio in 100 250 300; do
        echo "=== $model / x$ratio ==="
        grep 'valid_psnr:' "$EVAL_DIR/${model}_r${ratio}.log"
    done
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $EVAL_DIR"
