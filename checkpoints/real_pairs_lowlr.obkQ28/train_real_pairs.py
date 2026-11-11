import json
import os
import pickle
import random
import re
from pathlib import Path

import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset, DataLoader

from models.ELD_models import UNetSeeInDark


ROOT = Path.home() / "raw_image_denoising"
DATA = Path.home() / "data"
RUN = Path(os.environ["RUN_DIR"])

INFO = ROOT / "infos/SID_train_ospool.info"
SHORT_DIR = DATA / "SID/Sony/short"
SHADING_DIR = ROOT / "resources/SonyA7S2"
CHECKPOINT = ROOT / "checkpoints/sonya7s2.pth"

SEED = 3
EPOCHS = 30
LEARNING_RATE = 0.000001
PACKED_PATCH = 512

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not available.")

pattern = re.compile(r"^(\d{5})_\d+_([\d.]+)s\.ARW$", re.I)

with INFO.open("rb") as f:
    metadata = pickle.load(f)

shorts = {}
for path in sorted(SHORT_DIR.glob("*.ARW")):
    match = pattern.match(path.name)
    if match:
        exposure = float(match.group(2))
        if exposure > 0:
            shorts.setdefault(match.group(1), []).append((path, exposure))

scenes = []
for group in metadata:
    clean = [item for item in group if item.get("ratio") == 1]
    if len(clean) != 1:
        raise SystemExit(f"Expected one long exposure per scene, got {len(clean)}")

    item = clean[0]
    long_path = DATA / item["data"]
    match = pattern.match(long_path.name)

    if not match or not long_path.is_file():
        raise SystemExit(f"Missing or unrecognized long exposure: {long_path}")

    scene_id = match.group(1)
    if scene_id not in shorts:
        raise SystemExit(f"No short exposures for scene {scene_id}")

    scenes.append({
        "scene": scene_id,
        "long": str(long_path),
        "long_exposure": float(match.group(2)),
        "iso": int(item["ISO"]),
        "shorts": [(str(p), exp) for p, exp in shorts[scene_id]],
    })

if len(scenes) != 161 or len({x["scene"] for x in scenes}) != len(scenes):
    raise SystemExit("Unexpected scene count or duplicate scene IDs.")

random.Random(SEED).shuffle(scenes)
validation_scenes = scenes[:20]
training_scenes = scenes[20:]

split = {
    "seed": SEED,
    "training": [x["scene"] for x in training_scenes],
    "validation": [x["scene"] for x in validation_scenes],
    "gain_cap": 300,
}
(RUN / "scene_split.json").write_text(json.dumps(split, indent=2))

with (SHADING_DIR / "darkshading_BLE.pkl").open("rb") as f:
    ble = pickle.load(f)

shading = {
    "low_k": np.load(SHADING_DIR / "darkshading_lowISO_k.npy"),
    "low_b": np.load(SHADING_DIR / "darkshading_lowISO_b.npy"),
    "high_k": np.load(SHADING_DIR / "darkshading_highISO_k.npy"),
    "high_b": np.load(SHADING_DIR / "darkshading_highISO_b.npy"),
}


def read_raw(path):
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image_visible.astype(np.float32)


def pack(image):
    return np.stack([
        image[0::2, 0::2],
        image[0::2, 1::2],
        image[1::2, 0::2],
        image[1::2, 1::2],
    ], axis=0).astype(np.float32)


class RealPairs(Dataset):
    def __init__(self, items, training):
        self.items = items
        self.training = training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        options = item["shorts"]

        if self.training:
            short_path, short_exposure = random.choice(options)
        else:
            # A fixed exposure close to 250x makes validation repeatable.
            short_path, short_exposure = min(
                options,
                key=lambda x: abs(
                    min(item["long_exposure"] / x[1], 300) - 250
                ),
            )

        short = read_raw(short_path)
        long = read_raw(item["long"])

        if short.shape != long.shape:
            raise ValueError(
                f"Different image sizes for scene {item['scene']}: "
                f"{short.shape} and {long.shape}"
            )

        iso = item["iso"]
        if iso not in ble:
            raise ValueError(f"No dark-shading calibration for ISO {iso}")

        prefix = "low" if iso <= 1600 else "high"
        k = shading[f"{prefix}_k"]
        b = shading[f"{prefix}_b"]

        if k.shape != short.shape or b.shape != short.shape:
            raise ValueError(
                f"Dark-shading dimensions do not match scene {item['scene']}"
            )

        raw_patch = PACKED_PATCH * 2
        height, width = short.shape

        if height < raw_patch or width < raw_patch:
            raise ValueError(f"Image too small for patch: {item['scene']}")

        if self.training:
            # Even coordinates preserve the camera's color-filter pattern.
            y = 2 * random.randint(0, (height - raw_patch) // 2)
            x = 2 * random.randint(0, (width - raw_patch) // 2)
        else:
            y = ((height - raw_patch) // 4) * 2
            x = ((width - raw_patch) // 4) * 2

        region = np.s_[y:y + raw_patch, x:x + raw_patch]
        dark = k[region] * iso + b[region] + ble[iso]

        gain = min(item["long_exposure"] / short_exposure, 300.0)

        # Match SIDEvalDataset: subtract shading, normalize, then apply gain.
        low = pack(short[region] - dark)
        low = (low - 512.0) / (16383.0 - 512.0)
        low = np.minimum(low * gain, 1.0)

        high = pack(long[region])
        high = np.clip((high - 512.0) / (16383.0 - 512.0), 0, 1)

        if self.training:
            if random.random() < 0.5:
                low = low[:, ::-1, :]
                high = high[:, ::-1, :]
            if random.random() < 0.5:
                low = low[:, :, ::-1]
                high = high[:, :, ::-1]

        return {
            "lr": torch.from_numpy(low.copy()),
            "hr": torch.from_numpy(high.copy()),
            "scene": item["scene"],
        }


train_data = RealPairs(training_scenes, training=True)
val_data = RealPairs(validation_scenes, training=False)

train_loader = DataLoader(
    train_data, batch_size=1, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=1, shuffle=False, num_workers=0
)

# Check one training pair and one validation pair before training.
for label, dataset in [("training", train_data), ("validation", val_data)]:
    sample = dataset[0]
    print(
        f"Pre-flight {label}: scene={sample['scene']}, "
        f"input={tuple(sample['lr'].shape)}, "
        f"target={tuple(sample['hr'].shape)}, "
        f"input range={sample['lr'].min():.4f} to "
        f"{sample['lr'].max():.4f}"
    )

model = UNetSeeInDark().cuda()
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
weights = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
model.load_state_dict(weights, strict=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.L1Loss()


def validate():
    model.eval()
    values = []
    with torch.no_grad():
        for batch in val_loader:
            low = batch["lr"].cuda()
            high = batch["hr"].cuda()
            prediction = model(low).clamp(0, 1)
            values.append(criterion(prediction, high).item())
    return float(np.mean(values))


print(f"Training scenes: {len(training_scenes)}")
print(f"Validation scenes: {len(validation_scenes)}")
print(f"Initial validation L1: {validate():.6f}")

best = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []

    for batch in train_loader:
        low = batch["lr"].cuda()
        high = batch["hr"].cuda()

        optimizer.zero_grad()
        prediction = model(low).clamp(0, 1)
        loss = criterion(prediction, high)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    message = f"Epoch {epoch}/{EPOCHS} | training L1: {np.mean(losses):.6f}"

    if epoch % 5 == 0 or epoch == EPOCHS:
        val_loss = validate()
        message += f" | validation L1: {val_loss:.6f}"

        if val_loss < best:
            best = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_l1": best},
                RUN / "unet_best_validation.pth",
            )
            message += " | saved best"

    if epoch in (10, 20, 30):
        torch.save(
            {"model": model.state_dict(), "epoch": epoch},
            RUN / f"unet_ep{epoch}.pth",
        )

    print(message, flush=True)

print("Training complete.")
print(f"Best validation L1: {best:.6f}")
print(f"Results: {RUN}")
