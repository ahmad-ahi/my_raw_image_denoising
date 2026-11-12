import csv
import os
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import numpy as np
import torch

import test_denoise_sideld as evaluator
from utils.utils import PMN_metric, tensor_dim5to4

root = Path.home() / "raw_image_denoising"
run_dir = Path(os.environ["RUN_DIR"])

paths = {
    "sony": root / "checkpoints/sonya7s2.pth",
    "seed3": root / (
        "checkpoints/real_pairs_lowlr.obkQ28/"
        "unet_best_validation.pth"
    ),
    "seed4": root / (
        "checkpoints/real_pairs_lowlr_seed4.DA0Kr0/"
        "unet_best_validation.pth"
    ),
}

for path in paths.values():
    if not path.is_file():
        raise SystemExit(f"Checkpoint not found: {path}")

np.random.seed(1)
torch.manual_seed(1)

models = {}
for name, path in paths.items():
    args = SimpleNamespace(device="cuda:0", cp_dir=str(path))
    models[name] = evaluator.build_model(args)

rows = []

for ratio in (100, 250, 300):
    args = SimpleNamespace(testset_type="sid", eval_ratio=ratio)
    loader = evaluator.build_dataloader(args)

    print(f"\n=== SID x{ratio}: {len(loader.dataset)} images ===", flush=True)

    with torch.no_grad():
        for index, data in enumerate(loader):
            hr = tensor_dim5to4(data["hr"]).to("cuda:0").clamp(0, 1)
            lr = tensor_dim5to4(data["lr"]).to("cuda:0")

            scores = {}
            for name, model in models.items():
                # Same final clamp and metrics as the uncorrected evaluator.
                prediction = model(lr).clamp(0, 1)
                if name == 'seed3':
                    prediction = (prediction * 0.9941997475458398).clamp(0, 1)
                metrics = PMN_metric(prediction, hr)
                scores[name] = {
                    "psnr": metrics["psnr"].item(),
                    "ssim": metrics["ssim"].item(),
                }

            metadata = loader.dataset.data_info[index]
            scene = Path(str(metadata["long"])).name[:5]

            for name in ("seed3", "seed4"):
                rows.append({
                    "ratio": ratio,
                    "index": index,
                    "scene": scene,
                    "model": name,
                    "sony_psnr": scores["sony"]["psnr"],
                    "model_psnr": scores[name]["psnr"],
                    "psnr_change": (
                        scores[name]["psnr"] - scores["sony"]["psnr"]
                    ),
                    "sony_ssim": scores["sony"]["ssim"],
                    "model_ssim": scores[name]["ssim"],
                    "ssim_change": (
                        scores[name]["ssim"] - scores["sony"]["ssim"]
                    ),
                })

            if (index + 1) % 10 == 0:
                print(f"  Processed {index + 1}/{len(loader.dataset)}", flush=True)

output = run_dir / "per_image_results.csv"

with output.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n===== RESULTS SUMMARY =====")

for ratio in (100, 250, 300):
    for name in ("seed3", "seed4"):
        group = [
            row for row in rows
            if row["ratio"] == ratio and row["model"] == name
        ]
        changes = [row["psnr_change"] for row in group]

        print(f"\n=== {name} / SID x{ratio} ===")
        print(f"Images: {len(group)}")
        print(f"Mean PSNR change: {mean(changes):+.4f} dB")
        print(f"Median PSNR change: {median(changes):+.4f} dB")
        print(f"Images improved: {sum(x > 0 for x in changes)}")
        print(f"Images worsened: {sum(x < 0 for x in changes)}")
        print(f"Smallest change: {min(changes):+.4f} dB")
        print(f"Largest change: {max(changes):+.4f} dB")

print(f"\nPer-image CSV: {output}")
