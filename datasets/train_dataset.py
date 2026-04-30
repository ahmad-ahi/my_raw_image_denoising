import os
import pickle as pkl
import numpy as np
import rawpy
import torch
from torch.utils.data import Dataset


class SonyA7S2TrainDataset(Dataset):
    """
    Self-supervised training dataset using dark frame sampling + hypothesized K.
    Improvement 1: K is sampled from a range (QE 30-70%) instead of fixed value.
    """
    def __init__(self, 
                 dark_frame_dir,        # path to dark frames
                 info_path,             # path to ELD or SID .info file
                 patch_size=512,
                 n_patches=8,
                 qe_range=(0.30, 0.70), # QE range - paper uses 0.40 fixed
                 wl=16383, bl=512):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.qe_range = qe_range
        self.wl = wl
        self.bl = bl

        # Load dataset info
        with open(info_path, "rb") as f:
            self.data_info = pkl.load(f)
        # flatten scenes
        self.samples = []
        for scene in self.data_info:
            for item in scene:
                if item["ratio"] == 1:  # only use clean reference frames
                    self.samples.append(item)

        # Load dark frames
        self.dark_frames = []
        if os.path.isdir(dark_frame_dir):
            for fname in sorted(os.listdir(dark_frame_dir)):
                if fname.endswith(".npy"):
                    self.dark_frames.append(np.load(os.path.join(dark_frame_dir, fname)))
                elif fname.endswith(".ARW") or fname.endswith(".arw"):
                    try:
                        raw = rawpy.imread(os.path.join(dark_frame_dir, fname))
                        self.dark_frames.append(raw.raw_image_visible.astype(np.float32))
                    except:
                        pass
        
        if len(self.dark_frames) == 0:
            raise RuntimeError(f"No dark frames found in {dark_frame_dir}")
        
        # Precompute dark shading (mean of all dark frames)
        self.dark_shading = np.mean(np.stack(self.dark_frames, axis=0), axis=0)
        print(f"Loaded {len(self.dark_frames)} dark frames, {len(self.samples)} clean frames")

    def sample_K(self, iso):
        """
        Improvement 1: Sample K from a range based on QE physics.
        K = QE * AnalogGain = QE * (ISO / base_ISO)
        QE is sampled uniformly from qe_range instead of fixed 0.40
        """
        qe = np.random.uniform(self.qe_range[0], self.qe_range[1])
        # base_ISO ~= 100, analog_gain = ISO/100 * base_analog_gain
        # simplified: K = qe * ISO / 100 * 0.25 (0.25 is a scaling constant)
        K = qe * (iso / 100) * 0.25
        return K

    def sample_dark_frame(self):
        """Directly sample a random dark frame (no statistical modeling)."""
        idx = np.random.randint(len(self.dark_frames))
        return self.dark_frames[idx]

    def pack_raw(self, img, norm=True, clip=False):
        out = np.stack([img[0::2, 0::2], img[0::2, 1::2],
                        img[1::2, 0::2], img[1::2, 1::2]], axis=-1)
        out = (out - self.bl) / (self.wl - self.bl) if norm else out
        out = np.clip(out, 0, 1) if clip else out
        return out.astype(np.float32)

    def synthesize_noisy(self, clean_raw, iso, ratio):
        """
        Synthesize noisy image from clean:
        Noisy = Ka*(X + Np) + Ka*N1 + N2
        """
        K = self.sample_K(iso)
        
        # Clean signal in photon counts
        x = clean_raw / (self.wl - self.bl) * ratio  # scale up by ratio
        x = np.clip(x, 0, None)
        
        # Shot noise (Poisson)
        shot = np.random.poisson(np.maximum(x / K, 1e-6)).astype(np.float32) * K
        
        # Signal-independent noise: directly sample from dark frame
        dark = self.sample_dark_frame()
        dark_noise = dark - self.dark_shading  # remove fixed pattern
        
        # Combine
        noisy = shot + dark_noise
        noisy = noisy / (self.wl - self.bl)
        noisy = noisy * ratio
        
        return noisy.astype(np.float32)

    def random_crop(self, img, psize):
        h, w = img.shape[:2]
        hs = np.random.randint(0, h - psize + 1)
        ws = np.random.randint(0, w - psize + 1)
        return img[hs:hs+psize, ws:ws+psize]

    def augment(self, img):
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=0)
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)
        return img

    def __len__(self):
        return len(self.samples) * self.n_patches

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        iso = sample["ISO"]
        ratio = np.random.choice([100, 200])  # simulate different exposure ratios

        # Load clean raw
        raw = rawpy.imread(sample["data"])
        clean_raw = raw.raw_image_visible.astype(np.float32)

        # Synthesize noisy
        noisy_raw = self.synthesize_noisy(clean_raw - self.bl, iso, ratio)

        # Pack to 4 channels
        clean_packed = self.pack_raw(clean_raw, norm=True, clip=True)
        noisy_packed = noisy_raw.reshape(
            clean_raw.shape[0]//2, 2, clean_raw.shape[1]//2, 2
        )
        # Use clean_packed as HR, noisy as LR
        hr = clean_packed
        lr = self.pack_raw(clean_raw - self.bl + noisy_raw.reshape(clean_raw.shape), norm=False)

        # Random crop
        ps = self.patch_size
        h, w = hr.shape[:2]
        if h > ps and w > ps:
            hs = np.random.randint(0, h - ps)
            ws = np.random.randint(0, w - ps)
            hr = hr[hs:hs+ps, ws:ws+ps]
            lr = lr[hs:hs+ps, ws:ws+ps]

        # Augment
        hr = self.augment(hr)
        lr = self.augment(lr)

        hr = torch.FloatTensor(hr.copy()).permute(2, 0, 1)
        lr = torch.FloatTensor(lr.copy()).permute(2, 0, 1)
        lr = torch.clamp(lr * ratio, -1, 10)

        return {"hr": hr.unsqueeze(0), "lr": lr.unsqueeze(0),
                "iso": iso, "ratio": ratio}
