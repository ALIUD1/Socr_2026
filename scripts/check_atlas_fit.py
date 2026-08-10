#!/usr/bin/env python3
"""check_atlas_fit.py — verify the synthetic tumour actually conforms to the atlas.

Tumour SHAPE is decided entirely by synth_mask(), which is pure numpy. So we check it WITHOUT
running the diffusion model at all: no GPU, a couple of seconds, and you can iterate on the
shape parameters (AREA_FRAC / COMPACT / NOISE_SIGMA / NOISE_AMP in augment_tumor.py) freely
while the GPU is busy training.

Renders all 6 lobes at once as an OVERLAY -- the lobe outlined in cyan, the lesion filled in
red/yellow on top of the patient T1 -- so containment is visible, not inferred. Reports:

  containment  % of lesion pixels inside the lobe        -> must be 100% (it is a hard constraint)
  coverage     lesion area / lobe area                   -> should land in AREA_FRAC (15-35%)
  circularity  4*pi*Area / Perimeter^2                   -> 1.00 = a perfect disc, lower = organic
                                                            (the old fixed-circle version scored ~1)

Usage:  python scripts/check_atlas_fit.py          # runs on a LOGIN NODE, no GPU needed
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from datetime import datetime
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from augment_tumor import synth_mask, LOBES, ATLAS0, PROB_TH, MIN_BRAIN, MIN_LOBE

def circularity(blob):
    """4*pi*A / P^2 — the standard shape-compactness measure. A disc scores 1.0; the more
    ragged the outline, the longer P is for the same A, so the score falls."""
    area = float(blob.sum())
    if area == 0:
        return 0.0
    # perimeter ~= the pixels that are in the blob but NOT in its erosion (i.e. the boundary ring)
    per = float((blob & ~binary_erosion(blob)).sum())
    return 4 * np.pi * area / (per ** 2 + 1e-8)

def main():
    files = sorted(glob.glob("data/processed/slices/val/*.npy"))
    if not files:
        raise SystemExit("no val slices found — expected data/processed/slices/val/*.npy")
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10)); ax = ax.ravel()
    print(f"{'lobe':<12}{'lobe px':>9}{'lesion px':>11}{'coverage':>10}"
          f"{'contain':>9}{'circ':>7}   (circ 1.00 = perfect disc)")

    for k, lobe in enumerate(LOBES):
        li = LOBES.index(lobe)
        stack = atlas_lobe = None
        for f in files:                                  # find a tumour-free slice showing this lobe
            s = np.load(f).astype(np.float32)
            brain = s[1] > 0.05
            al = s[ATLAS0 + li]
            if (brain.sum() > MIN_BRAIN and (s[2] > 0.5).sum() < 50
                    and ((al > PROB_TH) & brain).sum() > MIN_LOBE):
                stack, atlas_lobe = s, al
                break
        if stack is None:
            ax[k].set_title(f"{lobe}: no suitable slice"); ax[k].axis("off")
            print(f"{lobe:<12}{'— no suitable slice —':>46}")
            continue

        brain = stack[1] > 0.05
        m, centre = synth_mask(atlas_lobe, brain, rng)
        if m is None:
            ax[k].set_title(f"{lobe}: no lesion grown"); ax[k].axis("off"); continue

        lobe_mask = (atlas_lobe > PROB_TH) & brain
        lesion    = m > 0
        n_lobe, n_les = int(lobe_mask.sum()), int(lesion.sum())
        contain = 100.0 * float((lesion & lobe_mask).sum()) / max(n_les, 1)
        circ    = circularity(lesion)
        print(f"{lobe:<12}{n_lobe:>9}{n_les:>11}{100*n_les/n_lobe:>9.0f}%"
              f"{contain:>8.0f}%{circ:>7.2f}")

        # --- overlay: T1 in grey, lobe outline in cyan, lesion filled on top ---
        ax[k].imshow(stack[1].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[k].contour(lobe_mask.T, levels=[0.5], colors="cyan", linewidths=1.2)
        show = np.ma.masked_where(m.T == 0, m.T)         # masked array -> zeros stay transparent
        ax[k].imshow(show, cmap="autumn", origin="lower", alpha=0.75, vmin=2, vmax=3)
        ax[k].set_title(f"{lobe}  |  {100*n_les/n_lobe:.0f}% of lobe  |  circ {circ:.2f}", fontsize=10)
        ax[k].axis("off")

    fig.suptitle("synthetic lesion (red/yellow) vs the lobe it must stay inside (cyan outline)",
                 fontsize=12)
    os.makedirs("outputs/augment", exist_ok=True)
    out = f"outputs/augment/atlas_fit_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print("\nsaved", out)

if __name__ == "__main__":
    main()
