"""dataset3d.py — serves whole 3D volumes for the 3D diffusion model.

Mirrors src/dataset.py exactly, one dimension up: each item is ONE PATIENT's volume rather than
one slice. Channel layout is identical to 2D, so the conditioning contract never changes:
    channel 0   = FLAIR   (the target we generate)
    channel 1   = T1      (patient anatomy)
    channel 2   = tumour mask
    channels 3-8 = 6 atlas lobe maps
"""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class VolumeDataset(Dataset):
    """Serves one preprocessed volume: target FLAIR + conditioning (T1 + mask + 6 atlas)."""

    def __init__(self, split, root="data/processed/volumes", augment=None):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy volumes in {root}/{split} — run 05_preprocess_3d.py first")
        # AUG_3D=1 turns on augmentation. It matters here far more than it did in 2D: there are
        # only ~875 training volumes (vs 164k slices), so without it the model sees the same few
        # hundred examples hundreds of times over and has very little to generalise from.
        self.augment = (os.environ.get("AUG_3D", "1") == "1") if augment is None else augment
        self.augment = self.augment and split == "train"     # never augment val/test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol = np.load(self.files[idx]).astype(np.float32)      # (9,64,64,64) stored as float16
        if self.augment:
            vol = self._aug(vol)
        target = torch.from_numpy(vol[0:1].copy())             # (1,64,64,64) FLAIR
        cond   = torch.from_numpy(vol[1:9].copy())             # (8,64,64,64) T1 + mask + atlas
        return target, cond

    @staticmethod
    def _aug(vol):
        """Augment ALL 9 channels together, so the (conditioning, target) pairing stays valid.

        Axis 0 is left-right (BraTS/SRI24 is stored LPS, verified via nib.aff2axcodes), so a flip
        along it is a clean anatomical mirror: a mirrored brain with a mirrored T1, mask and atlas
        is still a self-consistent example. Small shifts teach translation robustness. Both are
        label-preserving, which is what makes them safe -- we are not inventing new anatomy, just
        showing the same anatomy in a new pose.
        """
        if np.random.rand() < 0.5:
            vol = vol[:, ::-1]                                 # left-right mirror
        # +/-3 voxel jitter; the brain is centred with empty margin, so nothing wraps around
        sh = np.random.randint(-3, 4, size=3)
        if sh.any():
            vol = np.roll(vol, shift=tuple(sh), axis=(1, 2, 3))
        return vol

if __name__ == "__main__":
    ds = VolumeDataset("train")
    print("dataset size:", len(ds))
    target, cond = ds[0]
    print("one example:", target.shape, cond.shape)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
    tb, cb = next(iter(loader))
    print("one batch:", tb.shape, cb.shape)
