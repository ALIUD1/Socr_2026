#!/usr/bin/env python3
"""05_preprocess_3d.py — build downsampled 3D VOLUMES for the 3D diffusion prototype.

The 2D pipeline (03_preprocess.py) cut each patient into ~130 independent 256x256 slices.
Here we keep each patient as ONE volume so the model can learn through-plane continuity.

WHY 64^3: a native 240x240x155 volume is ~9M voxels, ~150x one 2D slice. Full-resolution 3D
diffusion does not fit on a single GPU, so the standard move (and what the SOCR paper's own 3D
GANs do) is to work at low resolution first. 64^3 = 262k voxels -> comfortably trainable.

SHAPE PIPELINE per patient:
    load 240x240x155  ->  zero-pad z to 240x240x240 (cube, so we don't distort aspect ratio)
                      ->  trilinear resize to 64x64x64  (voxel becomes 3.75 mm isotropic)
Channels match the 2D layout exactly: [FLAIR, T1, mask, atlas x6] -> (9, 64, 64, 64) float16.

Output: data/processed/volumes/{train,val,test}/<patient>.npy   (~4.7 MB each, ~6 GB total)
"""
import os, glob, csv, random
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

BRATS_DIR = "data/raw/brats"
ATLAS_DIR = "data/processed/atlas_in_brats"
OUT_DIR   = "data/processed/volumes"
LOBES = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]

N_PATIENTS = 1300
RES        = 64                    # target cube side
CUBE       = 240                   # pad every volume to this cube before resizing
SPLIT      = (0.7, 0.15, 0.15)     # train / val / test, BY PATIENT (same policy as 2D)
SEED       = 42

def load_vol(path):
    """Load a NIfTI as a squeezed 3D numpy array."""
    return np.squeeze(nib.load(path).get_fdata())

def find_image(pdir, suffix):
    """Find the .nii for a modality — handles BOTH flat files and nested folders (BraTS quirk)."""
    matches = glob.glob(os.path.join(pdir, f"*-{suffix}.nii*"))
    if not matches:
        raise FileNotFoundError(f"No *-{suffix}.nii* in {pdir}")
    entry = matches[0]
    if os.path.isdir(entry):
        inner = glob.glob(os.path.join(entry, "*.nii*"))
        if not inner:
            raise FileNotFoundError(f"No image file inside {entry}")
        return inner[0]
    return entry

def normalize(vol):
    """Percentile [1,99] normalization over brain voxels -> [0,1], background forced to 0."""
    brain = vol[vol > 0]
    if brain.size == 0:
        return vol.astype(np.float32)
    p1, p99 = np.percentile(brain, [1, 99])
    out = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)
    out[vol == 0] = 0
    return out.astype(np.float32)

def pad_cube(v, size=CUBE):
    """Zero-pad (or centre-crop) a 3D volume to size^3 — keeps aspect ratio so we don't distort."""
    out = np.zeros((size, size, size), np.float32)
    src = [min(v.shape[i], size) for i in range(3)]                 # how much actually fits
    s0 = [max((v.shape[i] - size) // 2, 0) for i in range(3)]       # crop start (if too big)
    d0 = [max((size - v.shape[i]) // 2, 0) for i in range(3)]       # paste start (if too small)
    out[d0[0]:d0[0]+src[0], d0[1]:d0[1]+src[1], d0[2]:d0[2]+src[2]] = \
        v[s0[0]:s0[0]+src[0], s0[1]:s0[1]+src[1], s0[2]:s0[2]+src[2]]
    return out

def resize(v, mode="trilinear"):
    """Downsample a (CUBE,CUBE,CUBE) volume to (RES,RES,RES).
    torch.nn.functional.interpolate needs a 5D tensor (N,C,D,H,W), so we add two axes."""
    t = torch.from_numpy(v)[None, None].float()                     # (1,1,240,240,240)
    kw = {"align_corners": False} if mode == "trilinear" else {}
    out = F.interpolate(t, size=(RES, RES, RES), mode=mode, **kw)   # (1,1,64,64,64)
    return out[0, 0].numpy()

def main():
    # the atlas is IDENTICAL for every patient -> load and resize it ONCE
    print("preparing atlas...")
    atlas = np.stack([resize(pad_cube(load_vol(
        os.path.join(ATLAS_DIR, f"atlas_{l}_in_sri24.nii.gz")))) for l in LOBES])   # (6,64,64,64)
    atlas = atlas / max(atlas.max(), 1e-8)

    patients = sorted(os.listdir(BRATS_DIR))[:N_PATIENTS]
    random.seed(SEED); random.shuffle(patients)
    n_tr = int(SPLIT[0] * len(patients)); n_va = int(SPLIT[1] * len(patients))
    split_of = {p: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, p in enumerate(patients)}
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, s), exist_ok=True)

    manifest = []
    for n, p in enumerate(patients):
        pdir = os.path.join(BRATS_DIR, p)
        try:
            flair = resize(pad_cube(normalize(load_vol(find_image(pdir, "t2f")))))
            t1    = resize(pad_cube(normalize(load_vol(find_image(pdir, "t1n")))))
            seg   = resize(pad_cube(load_vol(find_image(pdir, "seg")).astype(np.float32)),
                           mode="nearest")            # NEAREST: labels 1/2/3 must not be blended
        except (FileNotFoundError, IndexError) as e:
            print(f"  skip {p}: {e}")
            continue
        vol = np.concatenate([flair[None], t1[None], seg[None], atlas], 0)   # (9,64,64,64)
        split = split_of[p]
        out_path = os.path.join(OUT_DIR, split, f"{p}.npy")
        np.save(out_path, vol.astype(np.float16))
        manifest.append((split, p, out_path))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(patients)}")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "patient", "path"]); w.writerows(manifest)
    print(f"saved {len(manifest)} volumes of shape (9,{RES},{RES},{RES}) to {OUT_DIR}")

if __name__ == "__main__":
    main()
