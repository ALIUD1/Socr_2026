#!/usr/bin/env python3
"""03_preprocess.py — build 2D training slices (FLAIR + mask + 6 atlas) for the diffusion model."""
import os, glob, csv, random
import numpy as np
import nibabel as nib

BRATS_DIR = "data/raw/brats"
ATLAS_DIR = "data/processed/atlas_in_brats"
OUT_DIR   = "data/processed/slices"
LOBES = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]

N_PATIENTS       = 40          # subset to start; raise later
TARGET_SIZE      = 256         # pad 240 -> 256
MIN_BRAIN_VOXELS = 1000        # skip slices with less brain than this
SPLIT            = (0.7, 0.15, 0.15)   # train / val / test, by PATIENT
SEED             = 42

def load_vol(path):
    """Load a NIfTI/nested file as a squeezed 3D numpy array."""
    return np.squeeze(nib.load(path).get_fdata())

def normalize_flair(vol):
    """Percentile [0,1] normalization over brain voxels (your method)."""
    brain = vol[vol > 0]
    p1, p99 = np.percentile(brain, [1, 99])
    out = np.clip((vol - p1) / (p99 - p1), 0, 1)
    out[vol == 0] = 0
    return out.astype(np.float32)

def pad_to(img, size=TARGET_SIZE):
    """Zero-pad a 2D HxW slice to size x size, centered."""
    h, w = img.shape
    ph, pw = size - h, size - w
    top, left = ph // 2, pw // 2
    return np.pad(img, ((top, ph - top), (left, pw - left)))

def find_flair(pdir): return glob.glob(os.path.join(pdir, "*-t2f.nii/*.nii"))[0]
def find_seg(pdir):   return glob.glob(os.path.join(pdir, "*-seg.nii"))[0]

def main():
    # load the shared atlas ONCE: shape (6, 240, 240, 155), each lobe normalized to [0,1]
    atlas = np.stack([
        load_vol(os.path.join(ATLAS_DIR, f"atlas_{l}_in_sri24.nii.gz")) for l in LOBES
    ])
    atlas = atlas / atlas.max()        # all 6 maps to [0,1]

    # patient-level split
    patients = sorted(os.listdir(BRATS_DIR))[:N_PATIENTS]
    random.seed(SEED); random.shuffle(patients)
    n_tr = int(SPLIT[0] * len(patients))
    n_va = int(SPLIT[1] * len(patients))
    split_of = {p: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, p in enumerate(patients)}
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, s), exist_ok=True)

    manifest = []
    for p in patients:
        pdir  = os.path.join(BRATS_DIR, p)
        flair = normalize_flair(load_vol(find_flair(pdir)))
        seg   = load_vol(find_seg(pdir)).astype(np.float32)
        split = split_of[p]

        for z in range(flair.shape[2]):
            # ---- TODO: build & save one slice's 8-channel example ----
            # 1. fl = the FLAIR slice at z  (flair[:, :, z])
            # 2. skip this z if (fl > 0).sum() < MIN_BRAIN_VOXELS   (use `continue`)
            # 3. channels = [fl, seg slice at z] + [atlas[k, :, :, z] for k in range(6)]
            # 4. stack = np.stack([pad_to(c) for c in channels]).astype(np.float16)  # (8,256,256)
            # 5. out_path = os.path.join(OUT_DIR, split, f"{p}_z{z:03d}.npy")
            # 6. np.save(out_path, stack)
            # 7. manifest.append((split, p, z, out_path))
            fl = flair[:, :, z]
            if(fl >0).sum() <MIN_BRAIN_VOXELS:
                continue
            channels = [fl, seg[:,:,z]] + [atlas[k, :, :, z] for k in range(6)]
            stack = np.stack([pad_to(c) for c in channels]).astype(np.float16)
            out_path = os.path.join(OUT_DIR, split, f"{p}_z{z:03d}.npy")
            np.save(out_path, stack)
            manifest.append((split, p, z, out_path))
            pass
        print("done", p, "->", split)

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "patient", "z", "path"]); w.writerows(manifest)
    print(f"saved {len(manifest)} slices total")

if __name__ == "__main__":
    main()