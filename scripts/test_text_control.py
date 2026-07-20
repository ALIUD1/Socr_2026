#!/usr/bin/env python3
"""test_text_control.py — the decisive test: does the caption actually change the output?

METHOD (this is the whole point):
  Hold the spatial conditioning (T1 + mask + atlas) FIXED.
  Hold the random seed FIXED, so every run starts from the SAME initial noise.
  Vary ONLY the caption.
Any difference between the outputs is therefore caused by the text alone. If the images
come out identical (mean abs difference ~0), the model is ignoring the prompt.

We compare each variant against the generation from the slice's own TRUE caption.

Usage:
    python scripts/test_text_control.py            # guidance 3.0
    python scripts/test_text_control.py 7.5        # stronger guidance
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

DEVICE  = "cuda"
STEPS   = 200
SEED    = 0                     # FIXED -> identical starting noise for every prompt
CLIP_ID = "openai/clip-vit-base-patch32"
LOBES   = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]

def flip_side(cap):
    """Swap left <-> right so the prompt contradicts the mask's actual side."""
    if " left " in cap:  return cap.replace(" left ", " right ")
    if " right " in cap: return cap.replace(" right ", " left ")
    return cap + " (no side found)"

def swap_lobe(cap):
    """Replace whichever lobe is named with a clearly different one."""
    for l in LOBES:
        if l in cap:
            other = "frontal" if l != "frontal" else "occipital"
            return cap.replace(l, other)
    return cap + " (no lobe found)"

def main():
    guidance = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load("models/text_diffusion_ema.pt", map_location=DEVICE))
    model.eval()
    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()

    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(STEPS)

    # one tumour slice from the held-out split (mask is cond channel 1)
    ds = SliceDataset("val", return_caption=True)
    for i in range(len(ds)):
        target, cond, caption = ds[i]
        if cond[1].sum() > 100:
            break
    real   = target[0].numpy()
    cond_b = cond.unsqueeze(0).to(DEVICE)
    print("TRUE caption:", caption, "\nguidance:", guidance, "\n")

    def encode(text):
        tok = tokenizer([text], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return text_encoder(**tok).last_hidden_state

    enc_null = encode("")           # unconditional branch for classifier-free guidance

    def generate(prompt):
        """Same seed + same spatial conditioning every call -> only `prompt` varies."""
        enc_c = encode(prompt)
        torch.manual_seed(SEED)
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            x_in = torch.cat([x, cond_b], dim=1)
            with torch.no_grad():
                e_c = model(x_in, t, encoder_hidden_states=enc_c).sample
                e_u = model(x_in, t, encoder_hidden_states=enc_null).sample
            eps = e_u + guidance * (e_c - e_u)
            x = sched.step(eps, t, x).prev_sample
        return np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

    variants = [
        ("TRUE caption",   caption),
        ("opposite side",  flip_side(caption)),
        ("different lobe", swap_lobe(caption)),
        ("no tumour",      "axial FLAIR slice of a brain with no tumor"),
        ("null (empty)",   ""),
    ]

    brain = real > 0.05
    outs, stats = [], []
    for label, p in variants:
        g = generate(p)
        outs.append(g)
        # difference vs the TRUE-caption generation (outs[0]) -> how much did the text move it?
        mad_all   = float(np.mean(np.abs(g - outs[0])))
        mad_brain = float(np.mean(np.abs(g - outs[0])[brain]))
        stats.append((mad_all, mad_brain))
        print(f"{label:16s} | MAD(whole)={mad_all:.4f}  MAD(brain)={mad_brain:.4f}  <- {p[:70]}")

    print("\nINTERPRETATION: MAD ~0.000 for every row = the prompt is being IGNORED.")
    print("Clearly non-zero MAD = the text is genuinely steering the generation.")

    n = len(variants) + 1
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 4))
    ax[0].imshow(real.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax[0].set_title("REAL", fontsize=9); ax[0].axis("off")
    for k, ((label, p), g) in enumerate(zip(variants, outs)):
        ax[k+1].imshow(g.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[k+1].set_title(f"{label}\nMAD {stats[k][1]:.4f}", fontsize=9)
        ax[k+1].axis("off")
    fig.suptitle(f"same conditioning + same seed, only the caption changes (guidance {guidance})",
                 fontsize=10)
    out = f"outputs/text_control_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110); print("saved", out)

if __name__ == "__main__":
    main()
