#!/usr/bin/env python3
"""
01_inspect_brats.py
====================
Sanity-check ONE BraTS 2023 patient before we build anything on top of the data.

Why this script exists
----------------------
Before training any model you must confirm the raw data is what you think it is:
the files load, the image and the tumor mask line up, and the label values are
the ones the dataset documentation promises. Skipping this step is the #1 cause
of silent bugs later (a model "training" on misaligned or mislabeled data).

What it does
------------
1. Loads the FLAIR image (called `t2f` in BraTS 2023) and the segmentation mask.
2. Prints shapes, voxel spacing, and intensity range.
3. Prints which tumor labels are present and how many voxels each covers.
4. Saves a PNG overlaying the tumor mask on the FLAIR slice that has the most
   tumor, so you can eyeball that the mask sits on the lesion.

Run it ON GREAT LAKES (that is where the data lives):
    python scripts/01_inspect_brats.py /path/to/BraTS-GLI-00000-000
"""

import argparse
import glob
import os

import numpy as np
import nibabel as nib          # nibabel = the standard Python library for medical image volumes

import matplotlib
matplotlib.use("Agg")          # "Agg" = render straight to an image file. A compute
                               # node has no screen, so we must NOT try to open a window.
import matplotlib.pyplot as plt

# In BraTS 2023 the segmentation file stores these integer codes per voxel.
# (Earlier BraTS versions used 4 for enhancing tumor; 2023 renumbered it to 3.)
LABELS = {
    0: "background",
    1: "necrotic tumor core (NCR)",
    2: "peritumoral edema (ED)",
    3: "enhancing tumor (ET)",
}


def find_file(patient_dir, suffix):
    """
    BraTS files are named  <patient_id>-<modality>.nii.gz
    e.g.  BraTS-GLI-00000-000-t2f.nii.gz
    We glob for the one ending in the modality we want.
    """
    matches = glob.glob(os.path.join(patient_dir, f"*-{suffix}.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"No *-{suffix}.nii.gz file found in {patient_dir}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Inspect one BraTS 2023 patient.")
    parser.add_argument("patient_dir", help="path to a single BraTS patient folder")
    parser.add_argument("--out", default="outputs/brats_overlay.png",
                        help="where to save the overlay PNG")
    args = parser.parse_args()

    # --- locate the two files we care about for the first prototype ---
    flair_path = find_file(args.patient_dir, "t2f")   # t2f == FLAIR in BraTS 2023
    seg_path = find_file(args.patient_dir, "seg")     # the tumor label map

    # --- load them. nib.load is lazy; get_fdata() pulls the voxels into a numpy array ---
    flair_img = nib.load(flair_path)
    seg_img = nib.load(seg_path)
    flair = flair_img.get_fdata()                     # float array, shape (X, Y, Z)
    seg = seg_img.get_fdata().astype(int)             # label map -> force integers

    print(f"FLAIR file : {os.path.basename(flair_path)}")
    print(f"SEG   file : {os.path.basename(seg_path)}")
    print(f"FLAIR shape: {flair.shape}   voxel size (mm): {flair_img.header.get_zooms()}")
    print(f"SEG   shape: {seg.shape}")
    print(f"Shapes match: {flair.shape == seg.shape}")   # MUST be True
    print(f"FLAIR intensity min/max: {flair.min():.1f} / {flair.max():.1f}")

    # --- which labels are actually present, and how big is each region? ---
    print("\nSegmentation labels present:")
    values, counts = np.unique(seg, return_counts=True)
    for val, count in zip(values, counts):
        name = LABELS.get(int(val), "UNKNOWN LABEL (!)")
        print(f"  {int(val)} = {name:30s} {count:>9d} voxels")

    # --- choose the most informative slice to display ---
    # seg > 0 is a boolean tumor mask; summing over X and Y gives tumor voxels per
    # axial slice (the Z axis). We show the slice with the most tumor.
    tumor_per_slice = (seg > 0).sum(axis=(0, 1))
    if tumor_per_slice.max() > 0:
        z = int(tumor_per_slice.argmax())
    else:
        z = seg.shape[2] // 2                          # fallback: middle slice
    print(f"\nShowing axial slice z={z} (the one with the most tumor)")

    flair_slice = flair[:, :, z]
    seg_slice = seg[:, :, z]

    # --- draw FLAIR alone, and FLAIR with the colored mask on top ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # .T transposes so the image is oriented the way radiologists expect on screen;
    # origin="lower" puts row 0 at the bottom. These are display conventions only.
    axes[0].imshow(flair_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR")

    axes[1].imshow(flair_slice.T, cmap="gray", origin="lower")
    # mask out the background (label 0) so only tumor voxels get colored
    colored_mask = np.ma.masked_where(seg_slice.T == 0, seg_slice.T)
    axes[1].imshow(colored_mask, cmap="jet", origin="lower", alpha=0.5, vmin=0, vmax=3)
    axes[1].set_title("FLAIR + tumor mask")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved overlay -> {args.out}")


if __name__ == "__main__":
    main()
