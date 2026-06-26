#!/usr/bin/env python3
"""eval_grid.py — generate several samples on one conditioning + report quality metrics.

Metrics reported per generated sample:
  - brain mean intensity  : brightness sanity check vs the real brain
  - bgSNR (background SNR) : mean brain signal / std of the (should-be-black) background
                             -> high = clean background, low = grainy. The professor's "SNR".
  - PSNR (vs real, dB)     : pixel closeness to the real patient's FLAIR over the brain
  - SSIM (vs real, [0,1])  : structural similarity to the real FLAIR (whole image)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDIMScheduler
from src.dataset import SliceDataset
from src.model import build_model

try:                                   # SSIM lives in scikit-image; degrade gracefully if missing
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False

DEVICE, N_SAMPLES = "cuda", 6

def psnr(gen, real, brain, data_range=1.0):
    """Peak signal-to-noise ratio (dB) between gen and real, measured over the brain only.
    Higher = generated pixels are closer to the real ones. inf if identical."""
    mse = np.mean((gen[brain] - real[brain]) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(data_range ** 2 / mse)

def bg_snr(gen, brain, bg):
    """Background SNR of ONE generated image: mean brain signal / std of the background.
    The real image is skull-stripped (background exactly 0 -> infinite SNR), so this scores
    how clean OUR generated background is. Higher = cleaner."""
    noise = gen[bg].std()
    return float("inf") if noise == 0 else gen[brain].mean() / noise

def main():
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load("models/diffusion_ema.pt", map_location=DEVICE))
    model.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(200)

    ds = SliceDataset("val")
    for i in range(len(ds)):                      # find a tumor slice (mask is cond channel 1 now)
        target, cond = ds[i]
        if cond[1].sum() > 100:
            break
    real   = target[0].numpy()
    cond_b = cond.unsqueeze(0).to(DEVICE)

    brain = real > 0.05        # skull-stripped: brain is the bright nonzero region
    bg    = real == 0.0        # everything that should be pure black (air + padding)
    if not HAVE_SSIM:
        print("NOTE: scikit-image not installed -> SSIM skipped (pip install scikit-image)")
    print(f"REAL brain mean intensity = {real[brain].mean():.3f}")

    samples, metrics = [], []
    for s in range(N_SAMPLES):
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            with torch.no_grad():
                npred = model(torch.cat([x, cond_b], dim=1), t).sample
            x = sched.step(npred, t, x).prev_sample
        g = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)
        samples.append(g)

        p    = psnr(g, real, brain)
        snr  = bg_snr(g, brain, bg)
        ssim = _ssim(real, g, data_range=1.0) if HAVE_SSIM else float("nan")
        metrics.append((p, ssim, snr))
        print(f"  gen {s}: mean={g[g>0.05].mean():.3f}  "
              f"PSNR={p:5.2f}dB  SSIM={ssim:.3f}  bgSNR={snr:6.1f}")

    arr = np.array(metrics, dtype=float)
    print(f"AVG over {N_SAMPLES}: PSNR={arr[:,0].mean():.2f}dB  "
          f"SSIM={arr[:,1].mean():.3f}  bgSNR={arr[:,2].mean():.1f}")

    fig, ax = plt.subplots(2, 4, figsize=(16, 8)); ax = ax.ravel()
    ax[0].imshow(real.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax[0].set_title("REAL"); ax[0].axis("off")
    for k, g in enumerate(samples):
        p, ssim, snr = metrics[k]
        ax[k+1].imshow(g.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[k+1].set_title(f"gen {k}\nPSNR {p:.1f}  SSIM {ssim:.2f}  SNR {snr:.0f}")
        ax[k+1].axis("off")
    for a in ax[N_SAMPLES+1:]: a.axis("off")
    out = f"outputs/eval_grid_{datetime.now():%Y%m%d_%H%M%S}.png"   # unique name per run
    plt.tight_layout(); plt.savefig(out, dpi=110); print("saved", out)

if __name__ == "__main__":
    main()
