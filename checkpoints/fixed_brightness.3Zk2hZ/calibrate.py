import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import test_denoise_sideld as evaluator

project = Path.home() / "raw_image_denoising"
old_run = project / "checkpoints/real_pairs.wEZGga"
run_dir = Path(os.environ["RUN_DIR"])

# Reuse the exact validation dataset and scene split from the real-pair run.
training_script = old_run / "train_real_pairs.py"
source = training_script.read_text()
marker = "train_data = RealPairs(training_scenes, training=True)"

if source.count(marker) != 1:
    raise SystemExit("Could not safely locate the validation dataset setup.")

setup = source.split(marker, 1)[0]
namespace = {"__name__": "fixed_brightness_setup"}
exec(compile(setup, str(training_script), "exec"), namespace)

original_split = json.loads((old_run / "scene_split.json").read_text())
current_split = json.loads((run_dir / "scene_split.json").read_text())

if original_split != current_split:
    raise SystemExit("Validation split does not match the original run.")

dataset = namespace["RealPairs"](
    namespace["validation_scenes"], training=False
)

paths = {
    "sony": project / "checkpoints/sonya7s2.pth",
    "seed3": project / (
        "checkpoints/real_pairs_lowlr.obkQ28/"
        "unet_best_validation.pth"
    ),
}

models = {}
for name, path in paths.items():
    args = SimpleNamespace(device="cuda:0", cp_dir=str(path))
    models[name] = evaluator.build_model(args)

totals = {
    name: {"numerator": 0.0, "denominator": 0.0}
    for name in models
}

with torch.no_grad():
    for index in range(len(dataset)):
        sample = dataset[index]
        low = sample["lr"].unsqueeze(0).cuda()
        high = sample["hr"].unsqueeze(0).cuda().clamp(0, 1)
        mask = high != 1

        for name, model in models.items():
            prediction = model(low).clamp(0, 1)
            pred_pixels = prediction[mask]
            ref_pixels = high[mask]

            totals[name]["numerator"] += torch.sum(
                pred_pixels.double() * ref_pixels.double()
            ).item()

            totals[name]["denominator"] += torch.sum(
                pred_pixels.double() * pred_pixels.double()
            ).item()

        print(f"Calibration patch {index + 1}/{len(dataset)}", flush=True)

scales = {}
for name, values in totals.items():
    if values["denominator"] <= 0:
        raise SystemExit(f"Invalid calibration denominator for {name}")

    scale = values["numerator"] / values["denominator"]

    if not 0.5 < scale < 1.5:
        raise SystemExit(f"Unexpected calibration scale for {name}: {scale}")

    scales[name] = scale
    print(f"{name} fixed brightness multiplier: {scale:.6f}")

(run_dir / "calibration.json").write_text(
    json.dumps(scales, indent=2)
)

# Create an evaluator that uses a fixed multiplier rather than the reference.
no_correct = project / (
    "checkpoints/eld_no_reference_correction.Bx6fDN/"
    "eval_no_correction.py"
)

source = no_correct.read_text()
old = "        imgs_dn = model(imgs_lr)\n"

new = (
    "        imgs_dn = torch.clamp(model(imgs_lr), 0, 1) "
    "* float(os.environ['FIXED_SCALE'])\n"
)

if source.count(old) != 1:
    raise SystemExit("Could not safely locate model inference in evaluator.")

modified = source.replace(old, new, 1)
compile(modified, str(run_dir / "eval_fixed.py"), "exec")
(run_dir / "eval_fixed.py").write_text(modified)

print(f"Created fixed-scale evaluator: {run_dir / 'eval_fixed.py'}")
