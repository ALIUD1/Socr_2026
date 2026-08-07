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
from scipy.ndimage import gaussian_filter, label
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from src.model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # set CUDA_VISIBLE_DEVICES="" to force CPU
STEPS  = int(os.environ.get("AUG_STEPS", "200"))          # e.g. AUG_STEPS=60 for a faster CPU test
LOBES   = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]
ATLAS0  = 3                 # in the 9-channel stack, atlas lobe k lives at channel 3+k
PROB_TH   = 0.4             # "inside this lobe" = atlas probability above this
MIN_BRAIN = 15000           # require a full mid-axial slice, not a tiny inferior sliver
MIN_LOBE  = 500             # the target lobe must be genuinely prominent in the chosen slice

# --- tumour SIZE is now RELATIVE to the lobe, not a fixed radius -------------------------
AREA_FRAC   = (0.15, 0.35)  # lesion area as a fraction of the lobe's area IN THIS SLICE
CORE_FRAC   = 0.30          # enhancing core as a fraction of the lesion's area
MIN_AREA    = 60            # px; below this a "tumour" is too small to be meaningful
# --- tumour SHAPE is now organic + atlas-conforming --------------------------------------
COMPACT     = 0.95          # >1 = looser/more sprawling blob, <1 = tighter around the seed
NOISE_SIGMA = 6.0           # smoothing of the random field: larger = smoother, blobbier edges
NOISE_AMP   = 0.9           # 0 = perfect ellipse, ~1 = strongly irregular outline

def synth_mask(atlas_lobe, brain, rng):
    """Grow an organic tumour that CONFORMS to the lobe and is SIZED relative to it.

    Instead of stamping a fixed circle, we score every pixel by
        atlas probability  x  compactness-around-a-seed  x  smooth random field
    and take the top-N pixels, where N is a fraction of the lobe's area in this slice.
    The atlas term makes the lesion hug the lobe's real shape; the random field makes the
    outline irregular like a real tumour; the compactness term keeps it a single mass.

    atlas_lobe,(256,256) probabilities;  brain,(256,256) bool.
    Returns (mask,(256,256) with labels 2=edema/3=core, centre) or (None,None).
    """
    lobe = (atlas_lobe > PROB_TH) & brain            # the lobe's footprint in THIS slice
    n_lobe = int(lobe.sum())
    if n_lobe < MIN_LOBE:
        return None, None

    # --- 1. seed: a point well inside the lobe (sampled by atlas probability) ---
    ys, xs = np.nonzero(lobe)
    w = atlas_lobe[ys, xs].astype(np.float64); w = w / w.sum()
    j = int(rng.choice(len(ys), p=w))
    c0, c1 = int(ys[j]), int(xs[j])

    # --- 2. target size: a fraction of THIS lobe's area -> big lobe, big tumour ---
    n_les = int(rng.uniform(*AREA_FRAC) * n_lobe)
    if n_les < MIN_AREA:
        return None, None

    # --- 3. score field = atlas x compactness x irregularity ---
    idx0 = np.arange(256, dtype=np.float32)[:, None]      # (256,1) row index
    idx1 = np.arange(256, dtype=np.float32)[None, :]      # (1,256) col index
    dist = np.sqrt((idx0 - c0) ** 2 + (idx1 - c1) ** 2)   # (256,256) distance from the seed
    r_eq = np.sqrt(n_les / np.pi)                         # radius a circle of that area would have
    compact = np.exp(-(dist / (COMPACT * r_eq)) ** 2)     # smooth falloff -> keeps it one mass

    noise = gaussian_filter(rng.standard_normal((256, 256)).astype(np.float32), NOISE_SIGMA)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)   # -> [0,1]
    irregular = 1.0 + NOISE_AMP * (noise - 0.5)           # ~[0.55,1.45] multiplier

    score = atlas_lobe.astype(np.float32) * compact * irregular
    score[~lobe] = 0.0                                    # HARD constraint: never leave the lobe

    # --- 4. take the top-N scoring pixels = the lesion ---
    flat = score.ravel()
    n_les = min(n_les, int((flat > 0).sum()))
    if n_les < MIN_AREA:
        return None, None
    top = np.argpartition(flat, -n_les)[-n_les:]          # indices of the N highest scores
    blob = np.zeros(256 * 256, bool); blob[top] = True
    blob = blob.reshape(256, 256)

    # --- 5. keep only the largest connected piece (no scattered islands) ---
    lab, n = label(blob)                                  # scipy: tag each connected component
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0    # index 0 = background, ignore it
        blob = lab == sizes.argmax()

    # --- 6. core = the highest-scoring pixels INSIDE the lesion ---
    m = np.zeros((256, 256), np.float32)
    m[blob] = 2                                           # label 2 = edema (bright FLAIR halo)
    inside = score * blob                                 # scores, zeroed outside the lesion
    n_core = max(1, int(CORE_FRAC * blob.sum()))
    core_idx = np.argpartition(inside.ravel(), -n_core)[-n_core:]
    core = np.zeros(256 * 256, bool); core[core_idx] = True
    m[core.reshape(256, 256) & blob] = 3                  # label 3 = enhancing tumour core

    m[~brain] = 0                                         # never spill outside the brain
    return m, (c0, c1)

def main():
    print(f"device: {DEVICE}  steps: {STEPS}" + ("  (CPU -> slow, be patient)" if DEVICE == "cpu" else ""))
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
        if brain.sum() > MIN_BRAIN and (s[2] > 0.5).sum() < 50 and ((al > PROB_TH) & brain).sum() > MIN_LOBE:
            stack, atlas_lobe = s, al
            break
    if stack is None:
        raise RuntimeError(f"no suitable tumour-free slice found with the {lobe} lobe present")

    brain = stack[1] > 0.05
    m, centre = synth_mask(atlas_lobe, brain, rng)
    if m is None:
        raise RuntimeError(f"could not grow a tumour in the {lobe} lobe on this slice")
    n_lobe = int(((atlas_lobe > PROB_TH) & brain).sum())
    n_les, n_core = int((m > 0).sum()), int((m == 3).sum())
    print(f"placed a synthetic tumour in the {lobe} lobe, centre (axis0,axis1)={centre}")
    print(f"  lobe area {n_lobe} px | lesion {n_les} px ({100*n_les/n_lobe:.0f}% of the lobe) "
          f"| core {n_core} px | lesion diameter ~{2*np.sqrt(n_les/np.pi):.0f} mm")

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
