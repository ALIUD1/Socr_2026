#!/usr/bin/env python3
"""
02_register_atlas.py
====================
Register the ICBM452 atlas into BraTS / SRI24 space ONE time, then warp all six
lobe probability maps into that space so they line up with every BraTS patient.

Why a script (not the REPL): the transforms ANTs produces land in /tmp, which is
wiped when your job ends, and an SSH drop kills an interactive session. A script
runs end-to-end and SAVES everything to a permanent folder, so you never lose work
and can re-run with one command.

The idea (recap):
  fixed  = SRI24 T1  (the space BraTS lives in)        -> the target
  moving = ICBM452 T1 (the space the atlas lives in)   -> gets warped to match
  We compute ONE transform (moving -> fixed), then APPLY that same transform to all
  six probability maps. Probabilities are smooth, so we use LINEAR interpolation.

Run on a COMPUTE NODE (registration is heavy):
    salloc --account=engin1 --partition=standard --cpus-per-task=4 --mem=16G --time=1:00:00
    module load python && source ~/envs/brainmri/bin/activate
    cd ~/Summer2026
    python scripts/02_register_atlas.py
"""

import os
import shutil

import numpy as np
import ants

import matplotlib
matplotlib.use("Agg")          # compute node has no screen -> render to file
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths (relative to the project root ~/Summer2026, so run the script from there)
# ----------------------------------------------------------------------------
FIXED = "data/raw/templates/sri24/inst/extdata/spgr.nii.gz"                               # SRI24 T1 (BraTS space)
MOVING = "data/raw/templates/icbm452_atlas_air12.zipfile/icbm452_atlas_air12_sinc.hdr"   # ICBM452 T1 (atlas space)

# each lobe's probability map: a folder "<name>.zipfile" containing a .hdr/.img pair
PROB_TMPL = "data/raw/icbm/icbm452_atlas_probability_{lobe}.zipfile/icbm452_atlas_probability_{lobe}.hdr"
LOBES = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]

OUT_DIR = "data/processed/atlas_in_brats"   # warped maps + transforms land here (permanent)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. load the two anatomical templates ---
    print("Loading templates...")
    fixed = ants.image_read(FIXED)
    moving = ants.image_read(MOVING)
    print(f"  fixed  (SRI24)   {fixed.shape}")
    print(f"  moving (ICBM452) {moving.shape}")

    # --- 2. compute the transform: ICBM452 -> SRI24 (rigid -> affine -> nonlinear) ---
    print("Registering (SyN)... this takes a few minutes")
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN")

    # --- 3. save the warped T1 (for a quality check) and the transforms (so we
    #        never have to recompute them) into our PERMANENT output folder ---
    ants.image_write(reg["warpedmovout"], os.path.join(OUT_DIR, "icbm452_T1_in_sri24.nii.gz"))
    saved_transforms = []
    for tpath in reg["fwdtransforms"]:                 # copy each transform out of /tmp
        dest = os.path.join(OUT_DIR, os.path.basename(tpath))
        shutil.copy(tpath, dest)
        saved_transforms.append(dest)
    print("Saved warped T1 and transforms to", OUT_DIR)

    # --- 4. apply the SAME transform to all six lobe probability maps ---
    #     interpolator="linear" because probabilities are smooth (a hard mask would
    #     use "nearestNeighbor" instead, to keep integer labels clean).
    fnp = np.squeeze(fixed.numpy())                    # for the QA overlay below
    warped_frontal_np = None
    for lobe in LOBES:
        prob_path = PROB_TMPL.format(lobe=lobe)
        prob = ants.image_read(prob_path)
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=prob,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )
        out_path = os.path.join(OUT_DIR, f"atlas_{lobe}_in_sri24.nii.gz")
        ants.image_write(warped, out_path)
        print(f"  warped {lobe:11s} -> {out_path}")
        if lobe == "frontal":
            warped_frontal_np = np.squeeze(warped.numpy())

    # --- 5. quality-check image: frontal-lobe probability overlaid on SRI24 ---
    #     If registration worked, the frontal "cloud" sits over the FRONT of the brain.
    z = fnp.shape[2] // 2
    prob_norm = warped_frontal_np / warped_frontal_np.max()      # scale to 0..1 for display
    masked = np.ma.masked_where(prob_norm[:, :, z].T < 0.05, prob_norm[:, :, z].T)
    plt.figure(figsize=(6, 6))
    plt.imshow(fnp[:, :, z].T, cmap="gray", origin="lower")
    plt.imshow(masked, cmap="hot", origin="lower", alpha=0.5)
    plt.title("Frontal-lobe probability on SRI24 (should sit at the FRONT)")
    plt.axis("off")
    qa_path = os.path.join("outputs", "registration_frontal_check.png")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(qa_path, dpi=120)
    print("Saved QA overlay ->", qa_path)
    print("\nDone. Six warped lobe maps are in", OUT_DIR)


if __name__ == "__main__":
    main()
