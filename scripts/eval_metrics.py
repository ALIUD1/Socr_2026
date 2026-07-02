#!/usr/bin/env python3
"""eval_metrics.py — quantitative eval over MANY val slices (not one).

Generates one sample per sampled val slice and reports mean +/- std of
PSNR, SSIM, and background-SNR vs the real FLAIR. This is the citable
number (e.g. "SSIM 0.83 +/- 0.04 over 100 held-out slices"), unlike
eval_grid.py which shows a single slice.

Usage:  python scripts/eval_metrics.py [N_SLICES]   # default 100
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from datetime import datetime
from diffusers import DDIMScheduler
from src.dataset import SliceDataset
from src.model import build_model

try:                                   # SSIM lives in scikit-image; degrade gracefully if missing
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False

DEVICE   = "cuda"
N_SLICES = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # random subset of val, for speed
STEPS    = 200                                              # DDIM steps per sample (match eval_grid)
SEED     = 0                                                # reproducible slice pick + sampling noise

def psnr(gen, real, brain, data_range=1.0):
    mse = np.mean((gen[brain] - real[brain]) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(data_range ** 2 / mse)

def bg_snr(gen, brain, bg):
    noise = gen[bg].std()
    return float("inf") if noise == 0 else gen[brain].mean() / noise

def main():
    torch.manual_seed(SEED)
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("models/diffusion_ema.pt", map_location=DEVICE))
    model.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(STEPS)

    ds = SliceDataset("val")
    random.seed(SEED)
    idxs = random.sample(range(len(ds)), min(N_SLICES, len(ds)))
    if not HAVE_SSIM:
        print("NOTE: scikit-image not installed -> SSIM skipped (pip install scikit-image)")
    print(f"evaluating {len(idxs)} of {len(ds)} val slices, {STEPS} DDIM steps each...")

    rows = []
    for n, i in enumerate(idxs):
        target, cond = ds[i]
        real   = target[0].numpy()
        cond_b = cond.unsqueeze(0).to(DEVICE)
        brain  = real > 0.05
        bg     = real == 0.0
        if brain.sum() < 100:          # skip near-empty slices (shouldn't happen post-preprocess)
            continue
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            with torch.no_grad():
                npred = model(torch.cat([x, cond_b], dim=1), t).sample
            x = sched.step(npred, t, x).prev_sample
        g = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)
        rows.append((psnr(g, real, brain),
                     _ssim(real, g, data_range=1.0) if HAVE_SSIM else float("nan"),
                     bg_snr(g, brain, bg)))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(idxs)} done")

    arr = np.array(rows, dtype=float)
    arr = arr[np.isfinite(arr[:, 0])]        # drop any inf PSNR so stats stay finite
    def ms(col): return f"{np.nanmean(col):.3f} +/- {np.nanstd(col):.3f}"
    summary = (f"=== over {len(arr)} val slices, {STEPS} DDIM steps ===\n"
               f"PSNR (dB): {ms(arr[:,0])}\n"
               f"SSIM     : {ms(arr[:,1])}\n"
               f"bgSNR    : {ms(arr[:,2])}\n")
    print("\n" + summary)

    out = f"outputs/eval_metrics_{datetime.now():%Y%m%d_%H%M%S}.txt"
    with open(out, "w") as f:
        f.write(summary)
    print("saved", out)

if __name__ == "__main__":
    main()
