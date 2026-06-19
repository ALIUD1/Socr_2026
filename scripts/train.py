#!/usr/bin/env python3
"""train.py — conditional diffusion (mixed precision + EMA)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from diffusers.training_utils import EMAModel          # NEW: the EMA helper
from src.dataset import SliceDataset
from src.model import build_model

DEVICE, T, BATCH, EPOCHS, LR = "cuda", 1000, 64, 80, 1e-4
CKPT_DIR = "models"

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    loader = DataLoader(SliceDataset("train"), batch_size=BATCH, shuffle=True, num_workers=4)

    model = build_model().to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    ema    = EMAModel(model.parameters(), decay=0.999)   # NEW: shadow average of the weights

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
            with torch.amp.autocast("cuda"):
                pred = model(x_in, t).sample
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.step(model.parameters())                 # NEW: update the EMA after each step
            running += loss.item()

        print(f"epoch {epoch}: loss {running/len(loader):.4f}")

        # save the raw weights, then the EMA weights (the ones we'll sample from)
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "diffusion_last.pt"))
        ema.store(model.parameters())                    # NEW: stash the live training weights
        ema.copy_to(model.parameters())                  # NEW: load EMA weights into the model
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "diffusion_ema.pt"))  # NEW: save EMA
        ema.restore(model.parameters())                  # NEW: put the training weights back

if __name__ == "__main__":
    main()
