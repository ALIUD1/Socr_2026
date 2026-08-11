#!/usr/bin/env python3
"""export_3d.py — turn a generated 64^3 volume into things you can actually explore in 3D.

The .npy from sample_3d.py IS a full volume; the PNG was only three fixed slices through it.
This produces two better artefacts:

  1. <name>.nii.gz  — a real NIfTI with the correct 3.75 mm isotropic voxel spacing, so you can
     open it in ITK-SNAP / 3D Slicer / FSLeyes and scroll, rotate, window, and 3D-render freely.
  2. <name>.gif     — an animation sweeping through all 64 slices in axial, coronal and sagittal
     at once. This is the presentation asset: it shows the volume is coherent in every plane,
     which a static figure cannot.

Usage:
    python scripts/export_3d.py                       # newest volume in outputs/gen3d/
    python scripts/export_3d.py outputs/gen3d/vol3d_20260811_120000.npy
No GPU needed.
"""
import sys, os, glob
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

VOXEL_MM = 3.75          # 240 mm field of view / 64 voxels -> each voxel is 3.75 mm isotropic
FPS      = 12

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        cands = sorted(glob.glob("outputs/gen3d/*.npy"), key=os.path.getmtime)
        if not cands:
            raise SystemExit("no volumes found — run scripts/sample_3d.py first")
        path = cands[-1]                                  # newest by modification time
    vol = np.load(path).astype(np.float32)
    if vol.ndim != 3:
        raise SystemExit(f"expected a 3D volume, got shape {vol.shape}")
    stem = os.path.splitext(path)[0]
    D, H, W = vol.shape
    print(f"{path}  shape {vol.shape}  range [{vol.min():.2f}, {vol.max():.2f}]")

    # ---- 1. NIfTI export -------------------------------------------------------------
    # The affine maps voxel indices -> millimetres. A diagonal matrix of the voxel size is
    # exactly "x mm per step along each axis", so viewers show correct real-world distances.
    affine = np.diag([VOXEL_MM, VOXEL_MM, VOXEL_MM, 1.0])
    nib.save(nib.Nifti1Image(vol, affine), f"{stem}.nii.gz")
    print(f"saved {stem}.nii.gz  ({VOXEL_MM} mm isotropic — open in ITK-SNAP / 3D Slicer / FSLeyes)")

    # ---- 2. scrolling GIF ------------------------------------------------------------
    n = min(D, H, W)                                      # frames = slices we can sweep in all planes
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    ims, titles = [], ["axial", "coronal", "sagittal"]
    # seed each panel with its middle slice so the artists exist before the animation starts
    for a, t, first in zip(ax, titles,
                           [vol[:, :, n//2], vol[:, n//2, :], vol[n//2, :, :]]):
        ims.append(a.imshow(first.T, cmap="gray", origin="lower", vmin=0, vmax=1))
        a.set_title(t); a.axis("off")
    txt = fig.suptitle("")

    def frame(i):
        """Called once per frame: swap in slice i along each of the three axes."""
        ims[0].set_data(vol[:, :, i].T)                   # axial   — vary the 3rd index
        ims[1].set_data(vol[:, i, :].T)                   # coronal — vary the 2nd
        ims[2].set_data(vol[i, :, :].T)                   # sagittal— vary the 1st
        txt.set_text(f"generated 64³ volume — slice {i+1}/{n}")
        return ims + [txt]

    anim = FuncAnimation(fig, frame, frames=n, interval=1000 // FPS, blit=False)
    anim.save(f"{stem}.gif", writer=PillowWriter(fps=FPS))
    print(f"saved {stem}.gif  ({n} frames, all three planes)")

if __name__ == "__main__":
    main()
