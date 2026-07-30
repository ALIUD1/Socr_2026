#!/usr/bin/env python3
"""test_focal.py — is the TEXT-placed tumour a focal lesion, or diffuse hemispheric brightening?

test_text_quant proved text controls WHICH SIDE the tumour goes (laterality). It could not tell
us whether the model draws a bounded lesion or just brightens half the brain. This does.

METHOD (mask blanked, so only text places anything; same seed both times):
    gen_tumour  = generate(the slice's real tumour caption)
    gen_healthy = generate("axial FLAIR slice of a brain with no tumor")
    added = gen_tumour - gen_healthy        # exactly what the word "tumour" put there

FOCALITY METRIC = fraction of the positive added-signal that sits in the brightest 5% of brain
pixels. A tight blob (a real tumour is ~1-3% of the brain) piles almost all its signal into that
top 5% -> focality near 1. A diffuse wash spreads evenly -> focality near 0.05 (the top 5% of a
uniform field holds 5% of the signal). So:
    ~0.05  = diffuse (text only brightens a region)
    ~0.5+  = focal   (text draws a bounded lesion)

Usage:  python scripts/test_focal.py [n_slices]     # default 8
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from src.dataset import SliceDataset
from src.model import build_text_model

DEVICE, STEPS, SEED, GUID = "cuda", 200, 0, 3.0
CLIP_ID  = "openai/clip-vit-base-patch32"
CKPT     = "models/text_place_ema.pt"
HEALTHY  = "axial FLAIR slice of a brain with no tumor"
TOPFRAC  = 0.05

def focality(pos, brain):
    """Fraction of positive added-signal held by the brightest TOPFRAC of brain pixels."""
    v = np.sort(pos[brain])[::-1]                  # brain pixels' added signal, high -> low
    if v.sum() <= 0:
        return 0.0
    k = max(1, int(TOPFRAC * len(v)))              # how many pixels the top 5% is
    return float(v[:k].sum() / v.sum())            # 0.05 = diffuse, ~1 = focal

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    txt = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True); sched.set_timesteps(STEPS)
    print(f"checkpoint: {CKPT}   mask BLANKED   guidance {GUID}\n")

    def encode(s):
        t = tok([s], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return txt(**t).last_hidden_state
    enc_null = encode("")

    def generate(prompt, cond_b):
        enc_c = encode(prompt)
        torch.manual_seed(SEED)
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            xin = torch.cat([x, cond_b], dim=1)
            with torch.no_grad():
                e_c = model(xin, t, encoder_hidden_states=enc_c).sample
                e_u = model(xin, t, encoder_hidden_states=enc_null).sample
            x = sched.step(e_u + GUID * (e_c - e_u), t, x).prev_sample
        return np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

    ds = SliceDataset("val", return_caption=True)
    focs, shown = [], []
    for i in range(len(ds)):
        target, cond, caption = ds[i]
        if cond[1].sum() <= 100 or (" left " not in caption and " right " not in caption):
            continue
        cond_b = cond.unsqueeze(0).to(DEVICE)
        cond_b[:, 1:2] = 0.0                                   # BLANK THE MASK
        brain = target[0].numpy() > 0.05
        gt, gh = generate(caption, cond_b), generate(HEALTHY, cond_b)
        added  = gt - gh
        f = focality(np.clip(added, 0, None), brain)
        focs.append(f)
        print(f"  slice {i:5d} | focality={f:.3f}  <- {caption[:55]}")
        if len(shown) < 3:
            shown.append((gt, gh, added, f))
        if len(focs) >= N:
            break

    focs = np.array(focs)
    mean, se = focs.mean(), focs.std(ddof=1) / np.sqrt(len(focs))
    print(f"\nfocality = {mean:.3f} +/- {se:.3f} (SE) over {len(focs)} slices")
    print("  ~0.05 = diffuse hemispheric brightening | ~0.5+ = a bounded focal lesion")

    vmax = max(float(np.abs(a).max()) for _, _, a, _ in shown)
    fig, ax = plt.subplots(len(shown), 3, figsize=(10, 3.3 * len(shown)))
    ax = np.atleast_2d(ax)
    for r, (gt, gh, added, f) in enumerate(shown):
        ax[r, 0].imshow(gt.T, cmap="gray", origin="lower", vmin=0, vmax=1); ax[r, 0].set_title("with tumour prompt", fontsize=9)
        ax[r, 1].imshow(gh.T, cmap="gray", origin="lower", vmin=0, vmax=1); ax[r, 1].set_title("healthy prompt", fontsize=9)
        ax[r, 2].imshow(added.T, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax); ax[r, 2].set_title(f"what the tumour word added  (focality {f:.2f})", fontsize=9)
        for a in ax[r]: a.axis("off")
    out = f"outputs/focal_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110); print("saved", out)

if __name__ == "__main__":
    main()
