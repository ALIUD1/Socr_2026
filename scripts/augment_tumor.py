#!/usr/bin/env python3
"""augment_tumor.py — atlas-guided synthetic tumour augmentation (Prof. Dinov's suggestion).

Manufacture a NEW labelled training example: place a synthetic tumour mask in a chosen lobe
(using the atlas as the guide), then let the EXISTING spatial diffusion model render a FLAIR
with a tumour there. Output = a (generated FLAIR, mask) pair -- and the mask is ground truth
because we drew it.

Why this works with no retraining: the spatial model faithfully renders a tumour wherever the
mask says (our "mask owns anatomy" finding). Here that reliability is the whole point -- we
control placement via a synthetic mask, and the atlas keeps that placement anatomically valid.

Usage:
    python scripts/augment_tumor.py             # random lobe
    python scripts/augment_tumor.py temporal    # a specific lobe
"""
import sys, os, glob, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from src.model import build_model

DEVICE, STEPS = "cuda", 200
LOBES   = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]
ATLAS0  = 3                 # in the 9-channel stack, atlas lobe k lives at channel 3+k
CORE_R, EDEMA_R = 9, 20     # blob radii in pixels (1 mm/px) -> ~18mm core, ~40mm lesion
PROB_TH = 0.3               # "inside this lobe" = atlas probability above this

def synth_mask(atlas_lobe, brain, rng):
    """Draw a 2-zone tumour (enhancing core + edema halo) at a random in-lobe, in-brain spot.
    atlas_lobe,(256,256) probabilities;  brain,(256,256) bool.  Returns (mask, centre) or (None,None)."""
    valid = (atlas_lobe > PROB_TH) & brain          # pixels that are both inside the lobe and inside the brain
    ys, xs = np.nonzero(valid)                       # coordinate lists of every valid pixel
    if len(ys) == 0:
        return None, None
    j = int(rng.integers(len(ys)))                  # pick one valid pixel at random -> the tumour centre
    c0, c1 = ys[j], xs[j]

    idx0 = np.arange(256)[:, None]                  # (256,1) row index
    idx1 = np.arange(256)[None, :]                  # (1,256) col index
    dist = np.sqrt((idx0 - c0) ** 2 + (idx1 - c1) ** 2)   # (256,256) distance of every pixel from the centre

    m = np.zeros((256, 256), np.float32)
    m[dist <= EDEMA_R] = 2                           # label 2 = edema (the bright FLAIR halo)
    m[dist <= CORE_R]  = 3                           # label 3 = enhancing tumour (the core)
    m[~brain] = 0                                    # never let the tumour spill outside the brain
    return m, (int(c0), int(c1))

def main():
    lobe = sys.argv[1] if len(sys.argv) > 1 else random.choice(LOBES)
    li   = LOBES.index(lobe)                         # ValueError here = you typed a lobe that doesn't exist
    rng  = np.random.default_rng(0)                  # seeded -> reproducible; change seed for variety

    model = build_model().to(DEVICE)                # the SPATIAL model (no text), same as the paper
    model.load_state_dict(torch.load("models/diffusion_ema.pt", map_location=DEVICE))
    model.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True); sched.set_timesteps(STEPS)

    # find a TUMOUR-FREE slice with plenty of brain AND the target lobe present, so we cleanly
    # ADD one controlled tumour to an otherwise-healthy slice.
    stack = atlas_lobe = None
    for f in sorted(glob.glob("data/processed/slices/val/*.npy")):
        s = np.load(f).astype(np.float32)
        brain = s[1] > 0.05                          # channel 1 = T1
        al    = s[ATLAS0 + li]                       # this slice's map for the target lobe
        if brain.sum() > 5000 and (s[2] > 0.5).sum() < 50 and ((al > PROB_TH) & brain).sum() > 200:
            stack, atlas_lobe = s, al
            break
    if stack is None:
        raise RuntimeError(f"no suitable tumour-free slice found with the {lobe} lobe present")

    brain = stack[1] > 0.05
    m, centre = synth_mask(atlas_lobe, brain, rng)
    print(f"placed a synthetic tumour in the {lobe} lobe, centre (axis0,axis1)={centre}")

    # build conditioning = [T1, OUR mask, atlas] by swapping our mask into channel 2
    stack[2] = m
    cond = torch.from_numpy(stack[1:9]).unsqueeze(0).to(DEVICE)   # (1,8,256,256)

    # render a FLAIR with the existing model (DDIM reverse loop)
    x = torch.randn(1, 1, 256, 256, device=DEVICE)
    for t in sched.timesteps:
        with torch.no_grad():
            npred = model(torch.cat([x, cond], dim=1), t).sample
        x = sched.step(npred, t, x).prev_sample
    flair = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

    # save the labelled pair (this is the augmentation output) + a visual
    os.makedirs("outputs/augment", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"outputs/augment/aug_{lobe}_{stamp}.npy", np.stack([flair, m]))   # (2,256,256): [FLAIR, mask]

    t1 = stack[1]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(t1.T,         cmap="gray",    origin="lower", vmin=0, vmax=1); ax[0].set_title("patient T1 (anatomy)")
    ax[1].imshow(atlas_lobe.T, cmap="viridis", origin="lower");                ax[1].set_title(f"{lobe} atlas (the guide)")
    ax[2].imshow(m.T,          cmap="hot",     origin="lower");                ax[2].set_title("synthetic tumour mask")
    ax[3].imshow(flair.T,      cmap="gray",    origin="lower", vmin=0, vmax=1); ax[3].set_title("GENERATED FLAIR + tumour")
    for a in ax: a.axis("off")
    out = f"outputs/augment/aug_{lobe}_{stamp}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110); print("saved", out)

if __name__ == "__main__":
    main()
