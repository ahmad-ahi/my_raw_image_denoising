"""
Training script for improved RAW image denoising.
Improvement 1: K sampled from QE range [0.30, 0.70] instead of fixed 0.40
Improvement 2: NAFNet backbone instead of U-Net
Improvement 3: Combined SID + ELD clean training pool (161 + 70 = 231 images)
Improvement 4 (experimental): approximate additive read-noise term to partially
    account for the real per-pixel sensor noise the paper's method captures via
    real dark frames, which we do not have access to (only calibrated shading).
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
                       dec_blocks=[2, 2, 2, 2],
                       middle_blk_num=args.middle_blk_num).to(args.device)
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


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, args):
    """Save a full checkpoint (model + optimizer + scheduler + metadata) so
    training can resume exactly where it left off, mid-schedule."""
    tmp_path = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "args": vars(args),
    }, tmp_path)
    # Atomic rename: avoids a half-written checkpoint if the job is killed
    # mid-save (e.g. HTCondor timeout landing exactly here).
    os.replace(tmp_path, path)


def find_resume_checkpoint(args):
    """Resolve which checkpoint to resume from.

    Priority:
    1. Explicit --resume path, if given and it exists.
    2. <save_dir>/<model>_latest.pth, if it exists (auto-resume — this is
       what lets the OSPool wrapper script just re-run the same command
       after every 4-hour timeout without any extra bookkeeping).
    3. None -> start from scratch.
    """
    if args.resume:
        if os.path.exists(args.resume):
            return args.resume
        print(f"WARNING: --resume path '{args.resume}' not found, "
              f"falling back to auto-detect.")

    latest_path = os.path.join(args.save_dir, f"{args.model}_latest.pth")
    if os.path.exists(latest_path):
        return latest_path

    return None


def main(args):
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    import pickle as pkl, rawpy
    from torch.utils.data import Dataset

    class SimpleTrainDataset(Dataset):
        """
        Simplified training dataset:
        - Clean frames pooled from multiple sources (SID train split + ELD
          ratio=1 frames), given as a comma-separated list of .info files
        - Noise synthesized on-the-fly
        - K sampled from QE range (Improvement 1)
        - Dark shading subtracted/re-applied to match eval's convention

        NOTE on `ratio`: this matches the paper's eval-time convention
        (see datasets/real_dataset.py) where `ratio` is a brightness-matching
        digital gain applied EXACTLY ONCE, after normalization, uniformly to
        the whole signal (including its noise):
            lr_raw = raw - dark_shading
            lr     = pack_normalize(lr_raw)
            lr     = lr * ratio

        NOTE on read noise: real eval images contain the sensor's actual
        signal-independent noise (read noise, banding, etc.) baked in, which
        the paper's own pipeline synthesizes by adding a REAL sampled dark
        frame. We don't have real dark frame samples, only the calibrated
        (deterministic) shading coefficients, so as an approximation we
        optionally add zero-mean Gaussian noise scaled by --read_noise_std
        and by sqrt(iso/100) as a simple ISO-dependent heuristic. This is a
        best-effort approximation, not a calibrated match to the paper's
        approach — worth ablating (--read_noise_std 0 disables it).
        """
        def __init__(self, info_paths, dark_shading_dir, data_root, patch_size=512,
                     qe_range=(0.30, 0.70), wl=16383, bl=512, n_per_image=4,
                     read_noise_std=0.0):
            self.samples = []
            for info_path in info_paths:
                with open(info_path, "rb") as f:
                    data = pkl.load(f)
                n_added = 0
                for scene in data:
                    for item in scene:
                        if item["ratio"] == 1:
                            self.samples.append(item)
                            n_added += 1
                print(f"  Loaded {n_added} clean frames from {info_path}")

            self.data_root = data_root
            self.patch_size = patch_size
            self.qe_range   = qe_range
            self.wl, self.bl = wl, bl
            self.n = n_per_image
            self.read_noise_std = read_noise_std

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
            print(f"Dataset: {len(self.samples)} total clean frames × {n_per_image} patches "
                  f"(read_noise_std={read_noise_std})")

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

            raw_path = os.path.join(self.data_root, item["data"])
            raw = rawpy.imread(raw_path).raw_image_visible.astype(np.float32)
            # HR: normalized clean
            hr_raw = raw.copy()

            # Subtract dark shading for LR
            ds  = self.get_darkshading(iso)
            lr_raw = raw - ds

            # Pack HR
            hr = self.pack(hr_raw, norm=True, clip=True)   # [H/2, W/2, 4]

            # Shot noise: attenuate signal by ratio (simulating a shorter/
            # dimmer exposure), sample Poisson, matches paper's K*P(I/K).
            K   = self.sample_K(iso)
            lr_signal = np.clip(lr_raw / (self.wl - self.bl) / ratio, 0, None)
            shot = np.random.poisson(np.maximum(lr_signal / K, 1e-6)).astype(np.float32) * K

            # Approximate signal-independent read noise (see class docstring).
            # Added in the same normalized-domain units as `shot`, BEFORE the
            # single ratio re-inflation below, so it gets scaled uniformly
            # along with the signal -- matching how ratio is applied to real
            # data's whole signal (including its inherent noise) at eval time.
            if self.read_noise_std > 0:
                iso_scale = np.sqrt(max(iso, 100) / 100.0)
                read_noise = np.random.normal(
                    0, self.read_noise_std * iso_scale, size=shot.shape
                ).astype(np.float32)
                shot = shot + read_noise

            # Re-inflate by `ratio` exactly once here (matches eval's single
            # `lr_crops *= self.eval_ratio` applied after normalization), and
            # do NOT re-add dark shading — it was already subtracted above and
            # eval never adds it back either.
            lr = self.pack(shot * (self.wl - self.bl) * ratio, norm=True, clip=False)

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
            # No extra ratio multiplication here — already applied once above.
            lr = torch.clamp(lr, -1, 10)
            return {"hr": hr, "lr": lr, "iso": iso, "ratio": ratio}

    info_paths = [p.strip() for p in args.info_paths.split(",") if p.strip()]
    dataset = SimpleTrainDataset(
        info_paths       = info_paths,
        dark_shading_dir = args.dark_shading_dir,
        data_root        = args.data_root,
        patch_size       = args.patch_size,
        qe_range         = (args.qe_min, args.qe_max),
        n_per_image      = args.n_per_image,
        read_noise_std   = args.read_noise_std,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    model     = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.9), weight_decay=0.0)
    warmup_epochs = max(1, int(0.02 * args.epochs))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    criterion = nn.L1Loss()

    #  Training loop 
    start_epoch = 1
    best_loss = float("inf")

    resume_path = find_resume_checkpoint(args)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from '{resume_path}': epoch {ckpt.get('epoch', '?')}, "
              f"best_loss={best_loss:.5f}. Continuing at epoch {start_epoch}.")
    else:
        print("No checkpoint found, starting from scratch.")

    if start_epoch > args.epochs:
        print(f"start_epoch ({start_epoch}) > total epochs ({args.epochs}); "
              f"nothing left to train. Exiting cleanly.")
        return

    latest_path = os.path.join(args.save_dir, f"{args.model}_latest.pth")
    best_path   = os.path.join(args.save_dir, f"{args.model}_best.pth")

    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion,
                               args.device, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_loss, args)

        if is_best:
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_loss, args)
            print(f"  -> Saved best model (loss={best_loss:.5f})")

        if epoch % args.save_every == 0:
            epoch_path = os.path.join(args.save_dir, f"{args.model}_ep{epoch}.pth")
            save_checkpoint(epoch_path, model, optimizer, scheduler, epoch, best_loss, args)

    print(f"Training complete: {args.epochs} epochs, best_loss={best_loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   default="nafnet",  choices=["nafnet","unet"])
    parser.add_argument("--width", type=int, default=32, help="NAFNet base width")
    parser.add_argument("--middle_blk_num", type=int, default=12, help="NAFNet bottleneck depth")
    parser.add_argument("--info_paths",      type=str,   default="./infos/SID_train.info,./infos/ELD_SonyA7S2.info",
                         help="Comma-separated list of .info files to pool clean training frames from.")
    parser.add_argument("--dark_shading_dir",type=str,   default="./resources/SonyA7S2")
    parser.add_argument("--data_root",       type=str,   default="../data",
                         help="Root directory prepended to relative paths in the info files. "
                              "Leave empty if info files already store absolute/self-contained paths "
                              "(both SID_train.info and ELD_SonyA7S2.info do by default on am56).")
    parser.add_argument("--save_dir",        type=str,   default="./checkpoints/improved")
    parser.add_argument("--device",          type=str,   default="cuda:0")
    parser.add_argument("--patch_size",      type=int,   default=512)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--n_per_image",     type=int,   default=4)
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--qe_min",          type=float, default=0.30,      help="Min QE for K sampling")
    parser.add_argument("--qe_max",          type=float, default=0.70,      help="Max QE for K sampling")
    parser.add_argument("--read_noise_std",  type=float, default=0.0,
                         help="Approximate additive read-noise std (normalized-domain units, "
                              "pre-ratio). 0 disables it. Experimental heuristic -- see docstring.")
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--save_every",      type=int,   default=50)
    parser.add_argument("--resume", type=str, default="",
                         help="Path to a checkpoint to resume from. If omitted "
                              "(or the path doesn't exist), auto-resumes from "
                              "<save_dir>/<model>_latest.pth if present.")
    parser.add_argument("--seed",            type=int,   default=1)
    _args = parser.parse_args()
    main(_args)