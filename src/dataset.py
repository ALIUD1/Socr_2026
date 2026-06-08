import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SliceDataset(Dataset):
    """Serves one preprocessed slice: target FLAIR + conditioning (mask + 6 atlas)."""

    def __init__(self, split, root="data/processed/slices"):
        # gather all .npy files for this split (train/val/test)
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {root}/{split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ---- TODO ----
        # 1. load the file: stack = np.load(self.files[idx]).astype(np.float32)   # (8,256,256)
        # 2. target  = the FLAIR channel, KEEPING the channel axis:  stack[0:1]   # (1,256,256)
        # 3. cond    = the mask+atlas channels:                      stack[1:8]   # (7,256,256)
        # 4. convert both to tensors with torch.from_numpy(...)
        # 5. return target, cond
        stack = np.load(self.files[idx]).astype(np.float32)
        target = stack[0:1]
        cond = stack[1:8]
        return torch.from_numpy(target), torch.from_numpy(cond)
        
