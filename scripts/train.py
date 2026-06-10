#!/usr/bin/env python3
"""train.py — train a conditional diffusion model: mask+atlas -> FLAIR."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find src/

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from src.dataset import SliceDataset

DEVICE, T, BATCH, EPOCHS, LR = "cuda", 1000, 16, 20, 1e-4
CKPT_DIR = "models"

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # noise schedule (on the GPU)
    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    loader = DataLoader(SliceDataset("train"), batch_size=BATCH, shuffle=True, num_workers=4)

    model = UNet2DModel(
        sample_size=256, in_channels=8, out_channels=1, layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        running = 0.0
        for target, cond in loader:
            target = target.to(DEVICE) * 2 - 1     # FLAIR, rescaled [0,1] -> [-1,1]
            cond   = cond.to(DEVICE)
            B = target.shape[0]

            # ---- TODO: one diffusion training step ----
            # 1. t     = torch.randint(0, T, (B,), device=DEVICE)        # random timesteps
            # 2. noise = torch.randn_like(target)                         # the noise to add (the answer)
            # 3. a     = alpha_bars[t].view(B, 1, 1, 1)                    # signal fraction, shaped to broadcast
            # 4. noisy = a.sqrt() * target + (1 - a).sqrt() * noise        # your forward formula
            # 5. x_in  = torch.cat([noisy, cond], dim=1)                   # (B,8,256,256)
            # 6. pred  = model(x_in, t).sample                            # model's noise guess
            # 7. loss  = F.mse_loss(pred, noise)                          # how wrong vs the real noise
            # 8. opt.zero_grad(); loss.backward(); opt.step()             # learn
            # 9. running += loss.item()
            t = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target)
            a = alpha_bars[t].view(B, 1, 1, 1)
            noisy = a.sqrt() * target + (1-a).sqrt() * noise
            x_in = torch.cat([noisy, cond], dim = 1)
            pred = model(x_in, t).sample
            loss = F.mse_loss(pred,noise)
            opt.zero_grad();
            loss.backward();
            opt.step()
            running += loss.item()

        print(f"epoch {epoch}: loss {running/len(loader):.4f}")
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "diffusion_last.pt"))

if __name__ == "__main__":
    main()