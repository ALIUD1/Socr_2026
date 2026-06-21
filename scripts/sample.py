#!/usr/bin/env python3
"""sample.py — generate a synthetic FLAIR from the trained diffusion model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDIMScheduler
from src.dataset import SliceDataset
from src.model import build_model

DEVICE = "cuda"

def main():
    # 1. rebuild the SAME architecture, then load the trained weights
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("models/diffusion_ema.pt", map_location=DEVICE))
    model.eval()

    # 2. the scheduler holds the reverse-step math (DDIM = deterministic, stable, fewer steps)
    scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    scheduler.set_timesteps(200)

    # 3. grab conditioning (mask+atlas) from a validation slice, + the real FLAIR to compare
    ds = SliceDataset("val")
    # find the first validation slice that actually contains a tumor
    for i in range(len(ds)):
        target, cond = ds[i]
        if cond[1].sum() > 100:        # channel 1 of cond is the mask now (T1 is channel 0); >100 tumor voxels
            print("using slice", i, "with tumor voxels:", cond[1].sum().item())
            break
    cond = cond.unsqueeze(0).to(DEVICE)              # (1, 7, 256, 256)

    # 4. start from PURE NOISE and denoise step by step
    x = torch.randn(1, 1, 256, 256, device=DEVICE)
    for t in scheduler.timesteps:                    # 1000 steps, noisy -> clean
        with torch.no_grad():
            noise_pred = model(torch.cat([x, cond], dim=1), t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    # 5. rescale [-1,1] -> [0,1] and show generated vs real vs the mask we conditioned on
    gen  = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)
    real = target[0].numpy()
    mask = cond[0, 1].cpu().numpy()   # mask is conditioning channel 1 (T1 is 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(real.T, cmap="gray", origin="lower", vmin=0, vmax=1); ax[0].set_title("real FLAIR")
    ax[1].imshow(gen.T,  cmap="gray", origin="lower", vmin=0, vmax=1); ax[1].set_title("GENERATED FLAIR")
    ax[2].imshow(mask.T, cmap="hot",  origin="lower"); ax[2].set_title("tumor mask (conditioning)")
    for a in ax: a.axis("off")
    out = f"outputs/generated_sample_{datetime.now():%Y%m%d_%H%M%S}.png"   # unique per run
    plt.savefig(out, dpi=120)
    print("saved", out)

if __name__ == "__main__":
    main()