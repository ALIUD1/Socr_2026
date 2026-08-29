#!/usr/bin/env python3
"""bisect_denoise_3d.py — WHERE does the 3D model's dynamic range die?

Every 3D experiment so far has changed the model (architecture, depth, data, schedule) and
measured the SAMPLED volume — the end of a 100-250 step chain. That conflates two entirely
different failures:

    (A) the DENOISER is broken   — given a noised real volume it cannot reconstruct one
                                   with full range, so no sampler could ever succeed
    (B) the TRAJECTORY diverges  — the denoiser is fine at every individual step, but the
                                   250-step reverse chain drifts to the mean

One forward pass separates them. Take a REAL volume, noise it to a KNOWN t, run the model
ONCE, and algebraically recover x0:

    x_t   = sqrt(abar_t) * x0 + sqrt(1-abar_t) * eps          (forward, what training does)
    x0hat = (x_t - sqrt(1-abar_t) * eps_hat) / sqrt(abar_t)   (invert it with the PREDICTION)

Sweeping t traces exactly where reconstruction quality falls off:
    range good at every t   -> denoiser is healthy, the failure is (B), fix the sampler
    range dies past some t  -> (A), and the crossover t is the diagnosis
    range bad even at t=10  -> not a diffusion problem at all; check the data/output scale

CONTROLS (each kills a hypothesis that would otherwise survive):
  * DATA FLOOR   the real volume's own percentiles. If the training data never reaches 0,
                 the model reproducing a floor is CORRECT and there is no bug to fix.
  * eps-MSE      the model's actual training objective at each t. ~1.0 means "predicted
                 nothing"; the training loss floor was ~0.005. A model that scores well here
                 but reconstructs badly is being destroyed by the 1/sqrt(abar) amplification,
                 not by poor denoising.
  * PURE NOISE   x_T ~ N(0,I) with no real signal at all — precisely what sampling starts
                 from. One step, one x0hat. If THIS is already the washed-out mean, the chain
                 is doomed at step 1 and no amount of sampler tuning will help.
  * NO-COND      the same pure-noise pass with T1 zeroed. If x0hat barely moves, the model is
                 ignoring its conditioning at high t, which is a far simpler explanation than
                 zero-terminal-SNR: it has nothing to commit to except the dataset mean.

Usage
    python scripts/bisect_denoise_3d.py                 # default checkpoint, val volume 0
    python scripts/bisect_denoise_3d.py 3               # val volume index 3
Env
    CKPT_3D   checkpoint (default models/diffusion3d_ema.pt)
    ZERO_SNR  1 to mirror a zero-terminal-SNR training run (MUST match how it was trained)
    SCHED_3D  linear | cosine   (MUST match training)
    ARCH_3D / WIDTH_3D / BLOCKS_3D   as for sample_3d.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WIDTH_3D", "64")
os.environ.setdefault("BLOCKS_3D", "2")
os.environ.setdefault("AUG_3D", "0")          # never augment: we need the exact real volume

import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.dataset3d import VolumeDataset
from src.model3d import build_3d, load_state_compat

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = os.environ.get("CKPT_3D", "models/diffusion3d_ema.pt")
T      = 1000
RES    = 64
SEED   = 0
TS     = [10, 50, 100, 200, 400, 600, 800, 999]


def alpha_bars(sched, zero_snr, device):
    """The training schedule, rebuilt exactly as scripts/train_3d.py builds it.

    Printed back as sqrt(alpha_bar_T) so it can be checked against the training log line --
    a train/sample schedule mismatch is what produced the all-NaN volume on 2026-08-28.
    """
    if sched == "cosine":
        s = 0.008
        u = torch.linspace(0, 1, T + 1, device=device)
        f = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        ab = (f / f[0])[1:]
    else:
        ab = torch.cumprod(1.0 - torch.linspace(1e-4, 0.02, T, device=device), dim=0)
    if zero_snr:
        sab = ab.sqrt()
        sab0, sabT = sab[0].clone(), sab[-1].clone()
        ab = (((sab - sabT) * (sab0 / (sab0 - sabT))) ** 2).clamp(min=1e-8)
    return ab


def stats(vol01, bg, brain):
    """Percentiles of a volume already mapped to [0,1], split by real background vs brain."""
    p1, p99 = np.percentile(vol01, [1, 99])
    return dict(min=float(vol01.min()), p1=float(p1), p99=float(p99), max=float(vol01.max()),
                bg=float(vol01[bg].mean()), brain=float(vol01[brain].mean()))


def row(label, s, extra=""):
    return (f"{label:<16}{s['min']:8.3f}{s['p1']:8.3f}{s['p99']:8.3f}{s['max']:8.3f}"
            f"{s['bg']:9.3f}{s['brain']:9.3f}  {extra}")


def main():
    which = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sched_name = os.environ.get("SCHED_3D", "linear")
    zero_snr   = os.environ.get("ZERO_SNR", "0") == "1"

    model = build_3d().to(DEVICE)
    how = load_state_compat(model, CKPT, DEVICE)
    model.eval()
    print(f"loaded {CKPT} ({how}) | device {DEVICE} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    ab = alpha_bars(sched_name, zero_snr, DEVICE)
    print(f"schedule: {sched_name} | zero-terminal-SNR: {zero_snr} | "
          f"sqrt(alpha_bar) at t=T = {ab[-1].sqrt():.6f}   <- must match the training log")

    ds = VolumeDataset("val")
    idx, seen = 0, 0
    for i in range(len(ds)):
        _, c = ds[i]
        if c[1].sum() > 20:
            if seen == which:
                idx = i; break
            seen += 1
    target, cond = ds[idx]
    real01 = target[0].numpy()                                  # (64,64,64) in [0,1]
    x0 = (target.unsqueeze(0).to(DEVICE) * 2 - 1)               # (1,1,64,64,64) in [-1,1]
    c  = cond.unsqueeze(0).to(DEVICE)                           # (1,8,64,64,64)

    bg    = real01 < 0.02                                       # true background voxels
    brain = real01 > 0.05
    print(f"\nval volume {idx} | {int(bg.sum())} background / {int(brain.sum())} brain voxels")

    # ---- CONTROL: does the TRAINING DATA itself reach black? ----
    print("\n" + "=" * 96)
    print("CONTROL 1 — the real volume's own range. If min/bg are not ~0, the model's floor")
    print("is FAITHFUL and there is no bug: 05_preprocess_3d.py forces background to 0, so")
    print("anything above ~0.02 here means the preprocessing is not doing what it claims.")
    print("=" * 96)
    print(f"{'':<16}{'min':>8}{'p1':>8}{'p99':>8}{'max':>8}{'bg mean':>9}{'brain':>9}")
    real_s = stats(real01, bg, brain)
    print(row("REAL x0", real_s))

    torch.manual_seed(SEED)
    eps = torch.randn_like(x0)

    def recover(x_t, t_int, cond_t):
        """One forward pass -> predicted noise -> algebraic x0. Returns (x0hat01, eps_mse)."""
        t = torch.tensor([t_int], device=DEVICE)
        with torch.no_grad():
            eps_hat = model(torch.cat([x_t, cond_t], dim=1), t)
        a = ab[t_int]
        x0hat = ((x_t - (1 - a).sqrt() * eps_hat) / a.sqrt()).clamp(-1, 1)
        mse = float(((eps_hat - eps) ** 2).mean())
        return np.clip((x0hat[0, 0].cpu().numpy() + 1) / 2, 0, 1), mse

    # ---- the bisection itself ----
    print("\n" + "=" * 96)
    print("BISECTION — noise the REAL volume to t, denoise ONCE, invert to x0.")
    print("eps-MSE ~1.0 means the model predicted nothing; the 3D training loss floor was ~0.005.")
    print("=" * 96)
    print(f"{'t':<16}{'min':>8}{'p1':>8}{'p99':>8}{'max':>8}{'bg mean':>9}{'brain':>9}  eps-MSE")
    recons = []
    for t_int in TS:
        a = ab[t_int]
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
        g, mse = recover(x_t, t_int, c)
        recons.append((t_int, g))
        print(row(f"t={t_int}", stats(g, bg, brain), f"{mse:.4f}"))

    # ---- CONTROL: pure noise, exactly what sampling starts from ----
    print("\n" + "=" * 96)
    print("CONTROL 2 — PURE NOISE at t=T (zero real signal), which is where sampling begins.")
    print("A washed-out x0 here means the chain is already lost at step 1: not a trajectory")
    print("problem, a commitment problem. CONTROL 3 zeroes T1 to ask whether the model is even")
    print("USING its conditioning this deep in noise -- if the two rows match, it is not.")
    print("=" * 96)
    print(f"{'':<16}{'min':>8}{'p1':>8}{'p99':>8}{'max':>8}{'bg mean':>9}{'brain':>9}")
    x_T = eps.clone()
    pure, _ = recover(x_T, T - 1, c)
    print(row("pure noise", stats(pure, bg, brain)))

    c_nocond = c.clone(); c_nocond[:, 0:1] = 0.0                # channel 0 = T1
    nocond, _ = recover(x_T, T - 1, c_nocond)
    print(row("pure, no T1", stats(nocond, bg, brain)))
    delta = float(np.abs(pure - nocond).mean())
    print(f"\nmean |with T1 - without T1| = {delta:.4f}")
    print("  < 0.01  -> the model IGNORES conditioning at t=T; it can only emit the dataset mean")
    print("  > 0.05  -> conditioning IS being read, so the floor is not a commitment failure")

    print("\nHOW TO READ THE WHOLE THING")
    print("  Compare every row's min/bg against CONTROL 1's REAL x0 row -- that is the target.")
    print("  The t where bg mean lifts off ~0 is the crossover. Below it the denoiser works and")
    print("  the sampler is at fault; if there is no such t, the denoiser never worked and every")
    print("  architecture swap so far was testing the wrong component.")

    # ---- figure: one axial mid-slice per t, plus the two pure-noise controls ----
    m = RES // 2
    panels = [("REAL", real01)] + [(f"t={t}", g) for t, g in recons] \
             + [("pure noise", pure), ("pure, no T1", nocond)]
    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(2.1 * n, 2.8))
    for a_, (name, vol) in zip(ax, panels):
        a_.imshow(vol[:, :, m].T, cmap="gray", origin="lower", vmin=0, vmax=1)
        a_.set_title(name, fontsize=8); a_.axis("off")
    fig.suptitle(f"one-step x0 recovery vs noise level — {os.path.basename(CKPT)} "
                 f"(zero-SNR: {zero_snr})", fontsize=10)
    os.makedirs("outputs/bisect", exist_ok=True)
    out = f"outputs/bisect/bisect_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
