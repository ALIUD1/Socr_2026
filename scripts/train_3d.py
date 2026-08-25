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

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from src.dataset3d import VolumeDataset
from src.model3d import build_3d

DEVICE, T = "cuda", 1000
LR       = float(os.environ.get("LR_3D", "1e-4"))   # drop to ~2e-5 when resuming a plateaued run
BATCH    = int(os.environ.get("BATCH_3D", "16"))    # set from the src/model3d.py self-test
EPOCHS   = int(os.environ.get("EPOCHS_3D", "400"))  # ~875 volumes -> few steps/epoch, so many epochs
# RESUME_3D=models/diffusion3d_last.pt continues from an existing run instead of starting fresh.
# NOTE: only the WEIGHTS are restored, not Adam's momentum buffers -- Adam re-warms within a few
# dozen steps, so this is fine in practice, but it is why the first epoch after a resume can tick up.
RESUME   = os.environ.get("RESUME_3D", "")
# SCHED_3D=cosine switches the noise schedule. The linear schedule was tuned for 2D images and
# can destroy structure too early for volumetric data. IF YOU TRAIN WITH COSINE YOU MUST SAMPLE
# WITH COSINE -- sample_3d.py reads the same variable. Default stays linear.
SCHED    = os.environ.get("SCHED_3D", "linear")
ARCH     = os.environ.get("ARCH_3D", "custom")       # custom = our UNet3D | monai = MONAI's
SPLIT    = os.environ.get("SPLIT_3D", "train")       # "train+val" folds val in for more data
# ZERO_SNR=1 rescales the schedule so alpha_bar at t=T is EXACTLY 0.
# Why: with the standard linear schedule sqrt(alpha_bar_T) ~= 0.0066, so even the noisiest
# training input still carries 0.66% of the real image. In 2D that leak is negligible, but a
# 64^3 volume has 262k highly redundant voxels, so the hint is recoverable by averaging -- and
# the model learns to LEAN on it for global brightness. At sampling we start from pure noise
# (zero signal), the hint is absent, and the model falls back to the mean: output that cannot
# reach black or white. That is exactly the [0.22, 1.00] floor we measured across THREE
# different architectures and data sizes. Must be matched at sampling time.
ZERO_SNR = os.environ.get("ZERO_SNR", "0") == "1"
CKPT_DIR = "models"
TAG      = os.environ.get("TAG_3D", "diffusion3d")   # keeps parallel experiments apart

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    if SCHED == "cosine":
        # Nichol & Dhariwal: alpha_bar(t) = cos((t/T + s)/(1+s) * pi/2)^2, normalised so ab(0)=1.
        s_off = 0.008
        u  = torch.linspace(0, 1, T + 1, device=DEVICE)
        f  = torch.cos((u + s_off) / (1 + s_off) * math.pi / 2) ** 2
        alpha_bars = (f / f[0])[1:]
    else:                                        # identical noise schedule to the 2D model
        betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    if ZERO_SNR:
        # Shift-and-scale sqrt(alpha_bar) so the LAST value is 0 while the FIRST is unchanged.
        sab   = alpha_bars.sqrt()
        sab0, sabT = sab[0].clone(), sab[-1].clone()
        sab   = (sab - sabT) * (sab0 / (sab0 - sabT))
        alpha_bars = (sab ** 2).clamp(min=0.0)
    print(f"noise schedule: {SCHED} | zero-terminal-SNR: {ZERO_SNR} | "
          f"sqrt(alpha_bar) at t=T = {alpha_bars[-1].sqrt():.6f}", flush=True)

    loader = DataLoader(VolumeDataset(SPLIT), batch_size=BATCH, shuffle=True, num_workers=4)
    print(f"{len(loader.dataset)} training volumes (split={SPLIT}) | batch {BATCH} | "
          f"{len(loader)} steps/epoch", flush=True)

    model  = build_3d().to(DEVICE)            # ARCH_3D selects custom vs monai
    if RESUME:
        model.load_state_dict(torch.load(RESUME, map_location=DEVICE))   # shape error = wrong WIDTH_3D
        print(f"RESUMED weights from {RESUME}", flush=True)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {ARCH} | {nparam:.1f}M params (width {os.environ.get('WIDTH_3D','64')}) | lr {LR:g} | "
          f"{EPOCHS} epochs -> {EPOCHS*len(loader)} total gradient steps", flush=True)

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
