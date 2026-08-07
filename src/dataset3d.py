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

    def __init__(self, split, root="data/processed/volumes"):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy volumes in {root}/{split} — run 05_preprocess_3d.py first")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vol    = np.load(self.files[idx]).astype(np.float32)   # (9,64,64,64) stored as float16
        target = torch.from_numpy(vol[0:1])                    # (1,64,64,64) FLAIR
        cond   = torch.from_numpy(vol[1:9])                    # (8,64,64,64) T1 + mask + atlas
        return target, cond

if __name__ == "__main__":
    ds = VolumeDataset("train")
    print("dataset size:", len(ds))
    target, cond = ds[0]
    print("one example:", target.shape, cond.shape)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
    tb, cb = next(iter(loader))
    print("one batch:", tb.shape, cb.shape)
