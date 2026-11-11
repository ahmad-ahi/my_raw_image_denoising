#!/bin/bash
set -euo pipefail

EVAL_DIR="$1"
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

REAL="$PWD/checkpoints/real_pairs.wEZGga/unet_best_validation.pth"
SONY="$PWD/checkpoints/sonya7s2.pth"
BLEND="$PWD/checkpoints/sony_blend50.GHX2q1/unet_blend.pth"

PREFIX=$(basename "$EVAL_DIR")

exec > >(tee -a "$EVAL_DIR/run.log") 2>&1

echo "=== Started: $(date) ==="
echo "Results directory: $EVAL_DIR"

python3 - <<'PY'
import torch

path = "checkpoints/real_pairs.wEZGga/unet_best_validation.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)

print("Real-pair checkpoint epoch:", checkpoint.get("epoch"))
print("Validation L1:", checkpoint.get("val_l1"))
PY

for model in sony blend real_pairs; do
    case "$model" in
        sony)       checkpoint="$SONY" ;;
        blend)      checkpoint="$BLEND" ;;
        real_pairs) checkpoint="$REAL" ;;
    esac

    for ratio in 100 250 300; do
        echo "=== $model / SID x$ratio ==="

        python3 test_denoise_sideld.py \
            --task "${PREFIX}_${model}_r${ratio}" \
            --testset_type sid \
            --eval_ratio "$ratio" \
            --cp_dir "$checkpoint" \
            2>&1 | tee "$EVAL_DIR/${model}_r${ratio}.log"
    done
done

echo
echo "===== RESULTS SUMMARY ====="

for model in sony blend real_pairs; do
    for ratio in 100 250 300; do
        echo "=== $model / x$ratio ==="
        grep 'valid_psnr:' "$EVAL_DIR/${model}_r${ratio}.log"
    done
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $EVAL_DIR"
