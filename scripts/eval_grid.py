#!/usr/bin/env python3
"""eval_grid.py — generate several samples on one conditioning + report brightness stats."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDIMScheduler
from src.dataset import SliceDataset
from src.model import build_model

DEVICE, N_SAMPLES = "cuda", 6

def main():
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("models/diffusion_ema.pt", map_location=DEVICE))
    model.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(200)  

    ds = SliceDataset("val")
    for i in range(len(ds)):                      # find a tumor slice
        target, cond = ds[i]
        if cond[0].sum() > 100:
            break
    real   = target[0].numpy()
    cond_b = cond.unsqueeze(0).to(DEVICE)
    print(f"REAL brain mean intensity = {real[real > 0.05].mean():.3f}")

    samples = []
    for s in range(N_SAMPLES):
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            with torch.no_grad():
                npred = model(torch.cat([x, cond_b], dim=1), t).sample
            x = sched.step(npred, t, x).prev_sample
        g = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)
        samples.append(g)
        print(f"  gen sample {s}: brain mean intensity = {g[g > 0.05].mean():.3f}")

    fig, ax = plt.subplots(2, 4, figsize=(16, 8)); ax = ax.ravel()
    ax[0].imshow(real.T, cmap="gray", origin="lower", vmin=0, vmax=1); ax[0].set_title("REAL"); ax[0].axis("off")
    for k, g in enumerate(samples):
        ax[k+1].imshow(g.T, cmap="gray", origin="lower", vmin=0, vmax=1); ax[k+1].set_title(f"gen {k}"); ax[k+1].axis("off")
    for a in ax[N_SAMPLES+1:]: a.axis("off")
    plt.tight_layout(); plt.savefig("outputs/eval_grid_ddim_4.png", dpi=110); print("saved outputs/eval_grid_ddim_4.png")

if __name__ == "__main__":
    main()