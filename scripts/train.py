#!/usr/bin/env python3
"""train.py — train a conditional diffusion model: mask+atlas -> FLAIR (mixed precision)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from src.dataset import SliceDataset

DEVICE, T, BATCH, EPOCHS, LR = "cuda", 1000, 16, 30, 1e-4
CKPT_DIR = "models"

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    loader = DataLoader(SliceDataset("train"), batch_size=BATCH, shuffle=True, num_workers=4)

    model = UNet2DModel(
        sample_size=256, in_channels=8, out_channels=1, layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")          # NEW: manages 16-bit gradient scaling

    for epoch in range(EPOCHS):
        running = 0.0
        for target, cond in loader:
            target = target.to(DEVICE) * 2 - 1
            cond   = cond.to(DEVICE)
            B = target.shape[0]

            t     = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target)
            a     = alpha_bars[t].view(B, 1, 1, 1)
            noisy = a.sqrt() * target + (1 - a).sqrt() * noise
            x_in  = torch.cat([noisy, cond], dim=1)

            opt.zero_grad()
            with torch.amp.autocast("cuda"):       # NEW: run forward+loss in 16-bit where safe
                pred = model(x_in, t).sample
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()          # NEW: scale loss up, then backprop
            scaler.step(opt)                       # NEW: unscale grads, then optimizer step
            scaler.update()                        # NEW: adjust the scale factor for next time
            running += loss.item()

        print(f"epoch {epoch}: loss {running/len(loader):.4f}")
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "diffusion_last.pt"))

if __name__ == "__main__":
    main()