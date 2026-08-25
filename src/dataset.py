import os, glob, csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SliceDataset(Dataset):
    """Serves one preprocessed slice: target FLAIR + conditioning (T1 + mask + 6 atlas).

    If return_caption=True it also returns the slice's text caption (from captions.csv),
    for text conditioning. Default False keeps the old (target, cond) behaviour so the
    existing train/eval scripts are unaffected.
    """

    def __init__(self, split, root="data/processed/slices", return_caption=False):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {root}/{split}")

        self.return_caption = return_caption
        self.captions = {}
        if return_caption:
            # load captions.csv ONCE into a {filename -> caption} lookup table
            # CAPTIONS_CSV picks which caption set to train on: captions.csv (v1: lobe+side)
            # or captions_v2.csv (v2: adds size, slice level and dominant component). Swapping
            # the file is the whole experiment -- no other code changes.
            cap_path = os.path.join(root, os.environ.get("CAPTIONS_CSV", "captions.csv"))
            with open(cap_path, newline="") as f:
                for row in csv.DictReader(f):
                    self.captions[row["file"]] = row["caption"]
            if not self.captions:
                raise FileNotFoundError(f"No captions loaded from {cap_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path   = self.files[idx]
        stack  = np.load(path).astype(np.float32)      # (9,256,256): FLAIR, T1, mask, 6 atlas
        target = torch.from_numpy(stack[0:1])          # FLAIR — the channel we generate
        cond   = torch.from_numpy(stack[1:9])          # T1 + mask + 6 atlas (8 channels)
        if self.return_caption:
            caption = self.captions[os.path.basename(path)]   # look up by filename
            return target, cond, caption
        return target, cond

if __name__ == "__main__":
    # quick smoke test of the caption path
    ds = SliceDataset("train", return_caption=True)
    print("dataset size:", len(ds))
    target, cond, cap = ds[0]
    print("one example:", target.shape, cond.shape)
    print("caption:", cap)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    tb, cb, capb = next(iter(loader))
    print("one batch:", tb.shape, cb.shape, "| n captions:", len(capb))
    print("sample caption from batch:", capb[0])
