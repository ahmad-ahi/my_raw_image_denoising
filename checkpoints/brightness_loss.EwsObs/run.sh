#!/bin/bash
set -euo pipefail

RUN_DIR="$1"
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

export NOISE_MODE=real
export REAL_NOISE_DIR="$HOME/data/sony_bias_nov03"
export REAL_NOISE_EXTRA_DIR="$HOME/data/sony_bias_iso800_3200"
export REAL_NOISE_NEW_DIR="$HOME/data/sony_bias_iso250_320"

export SHOTFIX_INIT_CHECKPOINT="$PWD/checkpoints/sony_6iso_finetune.t4si24/sony_initial_weights.pth"
export SONY_CHECKPOINT="$PWD/checkpoints/sonya7s2.pth"

NO_CORRECT="$PWD/checkpoints/eld_no_reference_correction.Bx6fDN/eval_no_correction.py"

exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'echo "Run stopped because a command failed. See run.log."' ERR

echo "=== Started: $(date) ==="
echo "Run directory: $RUN_DIR"
echo "SID-only training; real dark frames; seed 3"
echo "Brightness-loss weight: 0.1"

sha256sum \
    "$RUN_DIR/train.py" \
    "$SHOTFIX_INIT_CHECKPOINT" \
    "$SONY_CHECKPOINT" \
    infos/SID_train_ospool.info

COMMON_ARGS=(
    --model unet
    --n_per_image 1
    --batch_size 1
    --num_workers 2
    --lr 0.00001
    --seed 3
    --qe_min 1.0
    --qe_max 1.0
    --read_noise_std 0.00003
    --info_paths ./infos/SID_train_ospool.info
    --device cuda:0
)

echo "=== One-epoch smoke test ==="

python3 "$RUN_DIR/train.py" \
    "${COMMON_ARGS[@]}" \
    --epochs 1 \
    --save_every 1 \
    --save_dir "$RUN_DIR/smoke" \
    2>&1 | tee "$RUN_DIR/smoke.log"

echo "=== Training for 50 epochs ==="

python3 "$RUN_DIR/train.py" \
    "${COMMON_ARGS[@]}" \
    --epochs 50 \
    --save_every 25 \
    --save_dir "$RUN_DIR/model" \
    2>&1 | tee "$RUN_DIR/train.log"

echo "=== Creating 50/50 blend with Sony ==="

export RUN_DIR

python3 - <<'PY'
import os
from pathlib import Path
import torch

run_dir = Path(os.environ["RUN_DIR"])

def weights(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

sony = weights(os.environ["SONY_CHECKPOINT"])
trained = weights(run_dir / "model/unet_ep50.pth")

if set(sony) != set(trained):
    raise SystemExit("Model parameter names do not match.")

blend = {}

for name in sony:
    a, b = sony[name], trained[name]

    if a.shape != b.shape or a.dtype != b.dtype:
        raise SystemExit(f"Incompatible parameter: {name}")

    if torch.is_floating_point(a):
        blend[name] = 0.5 * a + 0.5 * b
    else:
        if not torch.equal(a, b):
            raise SystemExit(f"Non-floating parameter differs: {name}")
        blend[name] = a.clone()

output = run_dir / "unet_blend50.pth"
torch.save({"model": blend}, output)
print("Saved:", output)
PY

CHECKPOINT="$RUN_DIR/unet_blend50.pth"
TASK_PREFIX=$(basename "$RUN_DIR")

echo "=== SID evaluations with benchmark correction ==="

for ratio in 100 250 300; do
    python3 test_denoise_sideld.py \
        --task "${TASK_PREFIX}_sid_r${ratio}" \
        --testset_type sid \
        --eval_ratio "$ratio" \
        --cp_dir "$CHECKPOINT" \
        2>&1 | tee "$RUN_DIR/sid_r${ratio}.log"
done

echo "=== ELD evaluations with benchmark correction ==="

for ratio in 100 200; do
    python3 test_denoise_sideld.py \
        --task "${TASK_PREFIX}_eld_corrected_r${ratio}" \
        --testset_type eld \
        --eval_ratio "$ratio" \
        --cp_dir "$CHECKPOINT" \
        2>&1 | tee "$RUN_DIR/eld_corrected_r${ratio}.log"
done

echo "=== ELD evaluations without reference correction ==="

for ratio in 100 200; do
    python3 "$NO_CORRECT" \
        --task "${TASK_PREFIX}_eld_uncorrected_r${ratio}" \
        --testset_type eld \
        --eval_ratio "$ratio" \
        --cp_dir "$CHECKPOINT" \
        2>&1 | tee "$RUN_DIR/eld_uncorrected_r${ratio}.log"
done

echo
echo "===== RESULTS SUMMARY ====="

for file in \
    "$RUN_DIR"/sid_r*.log \
    "$RUN_DIR"/eld_corrected_r*.log \
    "$RUN_DIR"/eld_uncorrected_r*.log
do
    echo "=== $(basename "$file") ==="
    grep 'valid_psnr:' "$file" || echo "Result not found"
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $RUN_DIR"
