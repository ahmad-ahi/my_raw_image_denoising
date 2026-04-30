"""
Training script for improved RAW image denoising.
Improvement 1: K sampled from QE range [0.30, 0.70] instead of fixed 0.40
Improvement 2: NAFNet backbone instead of U-Net
"""
import os
os.environ["OPENMP_NUM_THREADS"] = "4"

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.nafnet import NAFNet
from models.ELD_models import UNetSeeInDark


def build_model(args):
    if args.model == "nafnet":
        model = NAFNet(in_nc=4, out_nc=4, width=args.width,
                       enc_blocks=[2, 2, 4, 8],
                       dec_blocks=[2, 2, 2, 2]).to(args.device)
        print(f"NAFNet: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    else:
        model = UNetSeeInDark().to(args.device)
        print(f"UNet: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        lr = batch["lr"].squeeze(1).to(device)   # [B, 4, H, W]
        hr = batch["hr"].squeeze(1).to(device)   # [B, 4, H, W]

        pred = model(lr)
        pred = torch.clamp(pred, 0, 1)
        hr   = torch.clamp(hr, 0, 1)

        loss = criterion(pred, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(losses)


def main(args):
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    # We use ELD info file + dark frames from resources/
    # Dark frames are stored as pre-computed dark shading .npy files
    # We'll build a simple training set from ELD clean frames
    import pickle as pkl, rawpy
    from torch.utils.data import Dataset

    class SimpleTrainDataset(Dataset):
        """
        Simplified training dataset:
        - Clean frames from ELD dataset (ratio=1 images)
        - Noise synthesized on-the-fly
        - K sampled from QE range (Improvement 1)
        - Dark frames sampled directly (Improvement 2 of paper)
        """
        def __init__(self, info_path, dark_shading_dir, patch_size=512,
                     qe_range=(0.30, 0.70), wl=16383, bl=512, n_per_image=4):
            with open(info_path, "rb") as f:
                data = pkl.load(f)
            self.samples = []
            for scene in data:
                for item in scene:
                    if item["ratio"] == 1:
                        self.samples.append(item)
            self.patch_size = patch_size
            self.qe_range   = qe_range
            self.wl, self.bl = wl, bl
            self.n = n_per_image

            # Load precomputed dark shadings from resources
            dsk_high = np.load(f"{dark_shading_dir}/darkshading_highISO_k.npy")
            dsb_high = np.load(f"{dark_shading_dir}/darkshading_highISO_b.npy")
            dsk_low  = np.load(f"{dark_shading_dir}/darkshading_lowISO_k.npy")
            dsb_low  = np.load(f"{dark_shading_dir}/darkshading_lowISO_b.npy")
            with open(f"{dark_shading_dir}/darkshading_BLE.pkl", "rb") as f:
                ble = pkl.load(f)
            self.dsk_high = dsk_high
            self.dsb_high = dsb_high
            self.dsk_low  = dsk_low
            self.dsb_low  = dsb_low
            self.ble = ble
            print(f"Dataset: {len(self.samples)} clean frames × {n_per_image} patches")

        def get_darkshading(self, iso):
            if iso <= 1600:
                return self.dsk_low * iso + self.dsb_low + self.ble[iso]
            else:
                return self.dsk_high * iso + self.dsb_high + self.ble[iso]

        def sample_K(self, iso):
            # IMPROVEMENT 1: random QE instead of fixed 0.40
            qe = np.random.uniform(self.qe_range[0], self.qe_range[1])
            K  = qe * (iso / 100) * 0.1
            return max(K, 1e-4)

        def pack(self, img, norm=True, clip=False):
            out = np.stack([img[0::2,0::2], img[0::2,1::2],
                            img[1::2,0::2], img[1::2,1::2]], axis=-1).astype(np.float32)
            if norm:
                out = (out - self.bl) / (self.wl - self.bl)
            if clip:
                out = np.clip(out, 0, 1)
            return out

        def __len__(self):
            return len(self.samples) * self.n

        def __getitem__(self, idx):
            item = self.samples[idx % len(self.samples)]
            iso   = item["ISO"]
            ratio = float(np.random.choice([100, 200]))

            raw   = rawpy.imread(item["data"]).raw_image_visible.astype(np.float32)
            # HR: normalized clean
            hr_raw = raw.copy()

            # Subtract dark shading for LR
            ds  = self.get_darkshading(iso)
            lr_raw = raw - ds

            # Pack HR
            hr = self.pack(hr_raw, norm=True, clip=True)   # [H/2, W/2, 4]

            # Add shot noise to LR
            K   = self.sample_K(iso)
            lr_signal = np.clip(lr_raw / (self.wl - self.bl) / ratio, 0, None)
            shot = np.random.poisson(np.maximum(lr_signal / K, 1e-6)).astype(np.float32) * K
            lr   = self.pack(ds + shot * (self.wl - self.bl) * ratio + ds, norm=True, clip=False)

            # Crop
            ps = self.patch_size
            H, W = hr.shape[:2]
            if H > ps and W > ps:
                h0 = np.random.randint(0, H - ps)
                w0 = np.random.randint(0, W - ps)
                hr = hr[h0:h0+ps, w0:w0+ps]
                lr = lr[h0:h0+ps, w0:w0+ps]

            # Flip augmentation
            if np.random.rand() > .5: hr = hr[::-1]; lr = lr[::-1]
            if np.random.rand() > .5: hr = hr[:,::-1]; lr = lr[:,::-1]

            hr = torch.FloatTensor(hr.copy()).permute(2,0,1).unsqueeze(0)
            lr = torch.FloatTensor(lr.copy()).permute(2,0,1).unsqueeze(0)
            lr = torch.clamp(lr * ratio, -1, 10)
            return {"hr": hr, "lr": lr, "iso": iso, "ratio": ratio}

    dataset = SimpleTrainDataset(
        info_path        = args.info_path,
        dark_shading_dir = args.dark_shading_dir,
        patch_size       = args.patch_size,
        qe_range         = (args.qe_min, args.qe_max),
        n_per_image      = args.n_per_image,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    model     = build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    #  Training loop 
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion,
                               args.device, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"{args.model}_best.pth"))
            print(f"  → Saved best model (loss={best_loss:.5f})")

        if epoch % args.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"{args.model}_ep{epoch}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   default="nafnet",  choices=["nafnet","unet"])
    parser.add_argument("--width",           type=int,   default=32,        help="NAFNet base channels")
    parser.add_argument("--info_path",       type=str,   default="./infos/ELD_SonyA7S2.info")
    parser.add_argument("--dark_shading_dir",type=str,   default="./resources/SonyA7S2")
    parser.add_argument("--save_dir",        type=str,   default="./checkpoints/improved")
    parser.add_argument("--device",          type=str,   default="cuda:0")
    parser.add_argument("--patch_size",      type=int,   default=512)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--n_per_image",     type=int,   default=4)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--qe_min",          type=float, default=0.30,      help="Min QE for K sampling")
    parser.add_argument("--qe_max",          type=float, default=0.70,      help="Max QE for K sampling")
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--save_every",      type=int,   default=50)
    parser.add_argument("--seed",            type=int,   default=1)
    _args = parser.parse_args()
    main(_args)
