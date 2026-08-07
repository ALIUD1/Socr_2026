#!/usr/bin/env python3
"""train_3d.py — conditional 3D diffusion on 64^3 volumes (mixed precision + EMA).

Deliberately the SAME algorithm as scripts/train.py — identical schedule, identical
epsilon-prediction MSE objective, identical EMA. Only the data and the network are 3D. Keeping
the method fixed means any difference in behaviour is attributable to dimensionality, not to a
changed recipe.

Checkpoints -> models/diffusion3d_{last,ema}.pt (never touches the 2D models).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from src.dataset3d import VolumeDataset
from src.model3d import build_model_3d

DEVICE, T, LR = "cuda", 1000, 1e-4
BATCH    = int(os.environ.get("BATCH_3D", "4"))    # raise if probe_3d.py says it fits
EPOCHS   = int(os.environ.get("EPOCHS_3D", "100")) # volumes are ~130x fewer items than slices
CKPT_DIR = "models"
TAG      = "diffusion3d"

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # identical noise schedule to the 2D model
    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    loader = DataLoader(VolumeDataset("train"), batch_size=BATCH, shuffle=True, num_workers=4)
    print(f"{len(loader.dataset)} training volumes | batch {BATCH} | {len(loader)} steps/epoch", flush=True)

    model  = build_model_3d().to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {nparam:.1f}M params", flush=True)

    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    ema    = EMAModel(model.parameters(), decay=0.999)

    for epoch in range(EPOCHS):
        running = 0.0
        for target, cond in loader:
            target = target.to(DEVICE) * 2 - 1        # FLAIR [0,1] -> [-1,1]
            cond   = cond.to(DEVICE)                  # conditioning stays in [0,1]
            B = target.shape[0]

            # forward diffusion: noise the volume. The only change from 2D is the view shape --
            # (B,1,1,1,1) instead of (B,1,1,1), so alpha_bar broadcasts over D,H,W as well.
            t     = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target)
            a     = alpha_bars[t].view(B, 1, 1, 1, 1)
            noisy = a.sqrt() * target + (1 - a).sqrt() * noise

            x_in = torch.cat([noisy, cond], dim=1)    # (B,9,64,64,64)

            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(x_in, t)                 # predicted noise
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ema.step(model.parameters())
            running += loss.item()

        print(f"epoch {epoch}: loss {running/len(loader):.4f}", flush=True)

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{TAG}_last.pt"))
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{TAG}_ema.pt"))
        ema.restore(model.parameters())

if __name__ == "__main__":
    main()
