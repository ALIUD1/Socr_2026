#!/usr/bin/env python3
"""sample_3d.py — generate a synthetic 3D FLAIR volume and render it for presentation.

The point of going 3D is THROUGH-PLANE COHERENCE: a volume that still looks like a brain when
you re-slice it along the other two axes. Stacked independent 2D slices fail exactly that test.
So the figure shows all three orthogonal views (axial / coronal / sagittal) of the SAME volume.

IMPORTANT: the checkpoint is usable MID-TRAINING. models/diffusion3d_ema.pt is rewritten after
every epoch, so you can run this at any point to see how far along the model is.

Usage:
    python scripts/sample_3d.py                  # 100 DDIM steps, first val tumour volume
    python scripts/sample_3d.py 50               # faster, coarser
    python scripts/sample_3d.py 100 3            # 100 steps, use val volume index 3
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WIDTH_3D", "64")   # MUST match training. Old run1/run2 ckpts were
os.environ.setdefault("BLOCKS_3D", "2")   # width 96 / 1 block -> set those explicitly for them.

import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import DDIMScheduler

from src.dataset3d import VolumeDataset
from src.model3d import build_3d, load_state_compat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CKPT_3D lets you point at any checkpoint, e.g. the pre-resume backup for a before/after:
#   CKPT_3D=models/diffusion3d_ema_run1.pt python scripts/sample_3d.py 100
CKPT   = os.environ.get("CKPT_3D", "models/diffusion3d_ema.pt")
RES    = 64

def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    which = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    model = build_3d().to(DEVICE)   # set ARCH_3D=monai to sample a MONAI checkpoint
    how = load_state_compat(model, CKPT, DEVICE)   # shape error = wrong WIDTH_3D/BLOCKS_3D
    print(f"loaded {CKPT} ({how})")
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device {DEVICE} | width {os.environ['WIDTH_3D']} | {os.environ['BLOCKS_3D']} blocks/level | "
          f"{n:.1f}M params | {steps} DDIM steps")

    # MUST match the schedule used in training, or sampling walks a different trajectory
    # than the model was trained for and the output is garbage.
    sched_name = os.environ.get("SCHED_3D", "linear")
    zero_snr   = os.environ.get("ZERO_SNR", "0") == "1"
    kw = {"beta_schedule": "squaredcos_cap_v2"} if sched_name == "cosine" else {}
    if zero_snr:
        # Rebuild EXACTLY the training schedule and hand it to the scheduler as trained_betas,
        # rather than using diffusers' rescale flag -- that flag sets alpha_bar_T to exactly 0,
        # which makes DDIM's x0 = (x_t - sqrt(1-abar)*eps)/sqrt(abar) divide by zero and return
        # NaN. Deriving betas from our clamped alpha_bars guarantees train and sample match.
        import math as _m
        T_ = 1000
        b  = torch.linspace(1e-4, 0.02, T_)
        ab = torch.cumprod(1.0 - b, dim=0)
        sab = ab.sqrt(); sab0, sabT = sab[0].clone(), sab[-1].clone()
        ab = (((sab - sabT) * (sab0 / (sab0 - sabT))) ** 2).clamp(min=1e-8)
        alphas = torch.cat([ab[0:1], ab[1:] / ab[:-1]])      # alpha_t = abar_t / abar_{t-1}
        kw.update(trained_betas=(1.0 - alphas).clamp(0, 0.999).numpy(),
                  timestep_spacing="trailing")
        print(f"  rebuilt zero-SNR schedule: sqrt(alpha_bar) at t=T = {ab[-1].sqrt():.6f}")
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True, **kw)
    sched.set_timesteps(steps)
    print(f"noise schedule: {sched_name} | zero-terminal-SNR: {zero_snr}")

    # pick a held-out volume that actually contains a tumour, for a more interesting figure
    ds = VolumeDataset("val")
    idx, seen = 0, 0
    for i in range(len(ds)):
        _, c = ds[i]
        if c[1].sum() > 20:                       # cond channel 1 = tumour mask
            if seen == which:
                idx = i; break
            seen += 1
    target, cond = ds[idx]
    real   = target[0].numpy()                                     # (64,64,64) real FLAIR
    t1     = cond[0].numpy()                                       # (64,64,64) conditioning T1
    cond_b = cond.unsqueeze(0).to(DEVICE)                          # (1,8,64,64,64)
    print(f"val volume {idx} | tumour voxels {int((cond[1] > 0.5).sum())}")

    # ---- DDIM reverse loop, identical to 2D but on a 5D tensor ----
    x = torch.randn(1, 1, RES, RES, RES, device=DEVICE)
    for k, t in enumerate(sched.timesteps):
        with torch.no_grad():
            eps = model(torch.cat([x, cond_b], dim=1), t.expand(1).to(DEVICE))
        x = sched.step(eps, t, x).prev_sample
        if (k + 1) % 25 == 0:
            print(f"  step {k+1}/{steps}")
    gen = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)           # (64,64,64) in [0,1]

    os.makedirs("outputs/gen3d", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"outputs/gen3d/vol3d_{stamp}.npy", gen)               # the volume itself

    # ---- figure: 3 rows (T1 cond / real FLAIR / GENERATED) x 3 orthogonal views ----
    # Taking mid-slices along each axis of the SAME volume is the through-plane coherence test.
    m = RES // 2
    views = [("axial", lambda v: v[:, :, m]),
             ("coronal", lambda v: v[:, m, :]),
             ("sagittal", lambda v: v[m, :, :])]
    rows = [("T1 (conditioning)", t1), ("REAL FLAIR", real), ("GENERATED FLAIR", gen)]

    fig, ax = plt.subplots(3, 3, figsize=(11, 11))
    for r, (rname, vol) in enumerate(rows):
        for c, (vname, take) in enumerate(views):
            ax[r, c].imshow(take(vol).T, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax[r, c].set_title(f"{rname} — {vname}", fontsize=10)
            ax[r, c].axis("off")
    fig.suptitle(f"3D conditional diffusion, {RES}^3 volume — same volume viewed three ways",
                 fontsize=12)
    out = f"outputs/gen3d/gen3d_{stamp}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print(f"\nreal brain mean  {real[real>0.05].mean():.3f}")
    print(f"gen  brain mean  {gen[gen>0.05].mean():.3f}")
    print("saved", out)

if __name__ == "__main__":
    main()
