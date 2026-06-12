#!/usr/bin/env python3
"""sample.py — generate a synthetic FLAIR from the trained diffusion model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from src.dataset import SliceDataset

DEVICE = "cuda"

def main():
    # 1. rebuild the SAME architecture, then load the trained weights
    model = UNet2DModel(
        sample_size=256, in_channels=8, out_channels=1, layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(DEVICE)
    model.load_state_dict(torch.load("models/diffusion_last.pt", map_location=DEVICE))
    model.eval()

    # 2. the scheduler holds the reverse-step math (same schedule we trained with)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 3. grab conditioning (mask+atlas) from a validation slice, + the real FLAIR to compare
    ds = SliceDataset("val")
    ds = SliceDataset("val")
    # find the first validation slice that actually contains a tumor
    for i in range(len(ds)):
        target, cond = ds[i]
        if cond[0].sum() > 100:        # channel 0 of cond is the mask; >100 tumor voxels
            print("using slice", i, "with tumor voxels:", cond[0].sum().item())
            break
    cond = cond.unsqueeze(0).to(DEVICE)              # (1, 7, 256, 256)

    # 4. start from PURE NOISE and denoise step by step
    x = torch.randn(1, 1, 256, 256, device=DEVICE)
    for t in scheduler.timesteps:                    # 1000 steps, noisy -> clean
        with torch.no_grad():
            noise_pred = model(torch.cat([x, cond], dim=1), t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    # 5. rescale [-1,1] -> [0,1] and show generated vs real vs the mask we conditioned on
    gen  = (x[0, 0].cpu().numpy() + 1) / 2
    real = target[0].numpy()
    mask = cond[0, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(real.T, cmap="gray", origin="lower"); ax[0].set_title("real FLAIR")
    ax[1].imshow(gen.T,  cmap="gray", origin="lower"); ax[1].set_title("GENERATED FLAIR")
    ax[2].imshow(mask.T, cmap="hot",  origin="lower"); ax[2].set_title("tumor mask (conditioning)")
    for a in ax: a.axis("off")
    plt.savefig("outputs/generated_sample_v2.png", dpi=120)
    print("saved outputs/generated_sample_v2.png")

if __name__ == "__main__":
    main()