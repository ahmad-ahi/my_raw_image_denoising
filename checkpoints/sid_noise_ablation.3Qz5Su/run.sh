#!/bin/bash
set -euo pipefail

RUN_DIR="$1"
cd "$HOME/raw_image_denoising"
source "$HOME/venv_denoising/bin/activate"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

export REAL_NOISE_DIR="$HOME/data/sony_bias_nov03"
export REAL_NOISE_EXTRA_DIR="$HOME/data/sony_bias_iso800_3200"
export REAL_NOISE_NEW_DIR="$HOME/data/sony_bias_iso250_320"

export SHOTFIX_INIT_CHECKPOINT="$PWD/checkpoints/sony_6iso_finetune.t4si24/sony_initial_weights.pth"
export SONY_CHECKPOINT="$PWD/checkpoints/sonya7s2.pth"

exec > >(tee -a "$RUN_DIR/run.log") 2>&1
trap 'echo "Experiment stopped because a command failed. See run.log."' ERR

echo "=== Started: $(date) ==="
echo "Run directory: $RUN_DIR"
echo "Training images: SID only"

# Check that the saved initialization really matches the Sony checkpoint.
python3 - <<'PY'
import os
import torch

def weights(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

sony = weights(os.environ["SONY_CHECKPOINT"])
initial = weights(os.environ["SHOTFIX_INIT_CHECKPOINT"])

if set(sony) != set(initial):
    raise SystemExit("Sony and initialization have different parameter names.")

different = [
    name for name in sony
    if not torch.is_tensor(sony[name])
    or not torch.is_tensor(initial[name])
    or not torch.equal(sony[name], initial[name])
]

if different:
    raise SystemExit(
        "Initialization differs from Sony checkpoint. "
        f"First differences: {different[:5]}"
    )

print("Verified: initialization weights match Sony checkpoint.")
PY

# Confirm all dark frames used by the six-ISO branch are present.
python3 - <<'PY'
import os
from pathlib import Path

folders = {
    100: os.environ["REAL_NOISE_DIR"],
    1600: os.environ["REAL_NOISE_DIR"],
    800: os.environ["REAL_NOISE_EXTRA_DIR"],
    3200: os.environ["REAL_NOISE_EXTRA_DIR"],
    250: os.environ["REAL_NOISE_NEW_DIR"],
    320: os.environ["REAL_NOISE_NEW_DIR"],
}

missing = []
for iso, folder in folders.items():
    for number in range(1, 9):
        path = Path(folder) / f"{number:04d}_ISO{iso}.mat"
        if not path.is_file():
            missing.append(str(path))

if missing:
    print("Missing dark-frame files:")
    print("\n".join(missing[:20]))
    raise SystemExit(1)

print("Verified: all 48 training dark frames are present.")
PY

echo "=== Input checksums ==="
sha256sum \
    "$RUN_DIR/train.py" \
    "$SONY_CHECKPOINT" \
    "$SHOTFIX_INIT_CHECKPOINT" \
    infos/SID_train_ospool.info

COMMON_ARGS=(
    --model unet
    --n_per_image 1
    --batch_size 1
    --num_workers 2
    --lr 0.00001
    --seed 2
    --qe_min 1.0
    --qe_max 1.0
    --read_noise_std 0.00003
    --info_paths ./infos/SID_train_ospool.info
    --device cuda:0
)

# Smoke tests use separate folders so they cannot be resumed by the real runs.
for mode in gaussian real; do
    echo "=== Smoke test: $mode ==="
    NOISE_MODE="$mode" python3 "$RUN_DIR/train.py" \
        "${COMMON_ARGS[@]}" \
        --epochs 1 \
        --save_every 1 \
        --save_dir "$RUN_DIR/smoke_$mode" \
        2>&1 | tee "$RUN_DIR/smoke_${mode}.log"
done

# Run one after the other because this machine currently has one GPU.
for mode in gaussian real; do
    echo "=== Training: $mode ==="
    NOISE_MODE="$mode" python3 "$RUN_DIR/train.py" \
        "${COMMON_ARGS[@]}" \
        --epochs 50 \
        --save_every 25 \
        --save_dir "$RUN_DIR/$mode" \
        2>&1 | tee "$RUN_DIR/train_${mode}.log"
done

echo "=== Creating 50/50 blends with Sony ==="

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

for mode in ("gaussian", "real"):
    trained = weights(run_dir / mode / "unet_ep50.pth")

    if set(sony) != set(trained):
        raise SystemExit(f"{mode}: model parameter names do not match Sony.")

    blended = {}

    for name in sony:
        a, b = sony[name], trained[name]

        if a.shape != b.shape or a.dtype != b.dtype:
            raise SystemExit(f"{mode}: incompatible parameter {name}")

        if torch.is_floating_point(a):
            blended[name] = 0.5 * a + 0.5 * b
        else:
            if not torch.equal(a, b):
                raise SystemExit(f"{mode}: non-floating parameter differs: {name}")
            blended[name] = a.clone()

    output = run_dir / mode / "unet_blend50.pth"
    torch.save({"model": blended}, output)
    print("Saved:", output)
PY

echo "=== Single-pass SID evaluations ==="

TASK_PREFIX=$(basename "$RUN_DIR")

for mode in gaussian real; do
    for model in ep50 blend50; do
        checkpoint="$RUN_DIR/$mode/unet_${model}.pth"

        for ratio in 100 250 300; do
            echo "=== $mode / $model / SID x$ratio ==="

            python3 test_denoise_sideld.py \
                --task "${TASK_PREFIX}_${mode}_${model}_r${ratio}" \
                --testset_type sid \
                --eval_ratio "$ratio" \
                --cp_dir "$checkpoint" \
                2>&1 | tee "$RUN_DIR/${mode}_${model}_r${ratio}.log"
        done
    done
done

echo
echo "===== RESULTS SUMMARY ====="

for mode in gaussian real; do
    for model in ep50 blend50; do
        for ratio in 100 250 300; do
            echo "=== $mode / $model / x$ratio ==="
            grep 'valid_psnr:' "$RUN_DIR/${mode}_${model}_r${ratio}.log" \
                || echo "Result not found"
        done
    done
done

echo "=== Completed: $(date) ==="
echo "Results and logs: $RUN_DIR"
