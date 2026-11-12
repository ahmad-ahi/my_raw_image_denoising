#!/bin/bash
set -euo pipefail

export RUN_DIR="$1"

cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PREFIX=$(basename "$RUN_DIR")

exec > >(tee -a "$RUN_DIR/run.log") 2>&1

echo "=== Started: $(date) ==="
echo "=== Fit fixed multipliers on the 20 validation patches ==="

python3 "$RUN_DIR/calibrate.py"

for model in sony seed3; do
    case "$model" in
        sony)
            checkpoint="$PWD/checkpoints/sonya7s2.pth"
            ;;
        seed3)
            checkpoint="$PWD/checkpoints/real_pairs_lowlr.obkQ28/unet_best_validation.pth"
            ;;
    esac

    scale=$(python3 - "$RUN_DIR/calibration.json" "$model" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    values = json.load(f)

print(values[sys.argv[2]])
PY
)

    for ratio in 100 250 300; do
        echo "=== $model / fixed brightness / SID x$ratio ==="

        FIXED_SCALE="$scale" python3 "$RUN_DIR/eval_fixed.py" \
            --task "${PREFIX}_${model}_sid_r${ratio}" \
            --testset_type sid \
            --eval_ratio "$ratio" \
            --cp_dir "$checkpoint" \
            2>&1 | tee "$RUN_DIR/${model}_sid_r${ratio}.log"
    done

    for ratio in 100 200; do
        echo "=== $model / fixed brightness / ELD x$ratio ==="

        FIXED_SCALE="$scale" python3 "$RUN_DIR/eval_fixed.py" \
            --task "${PREFIX}_${model}_eld_r${ratio}" \
            --testset_type eld \
            --eval_ratio "$ratio" \
            --cp_dir "$checkpoint" \
            2>&1 | tee "$RUN_DIR/${model}_eld_r${ratio}.log"
    done
done

echo
echo "===== RESULTS SUMMARY ====="
cat "$RUN_DIR/calibration.json"

for model in sony seed3; do
    for ratio in 100 250 300; do
        echo "=== $model / SID x$ratio ==="
        grep 'valid_psnr:' "$RUN_DIR/${model}_sid_r${ratio}.log"
    done

    for ratio in 100 200; do
        echo "=== $model / ELD x$ratio ==="
        grep 'valid_psnr:' "$RUN_DIR/${model}_eld_r${ratio}.log"
    done
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $RUN_DIR"
