#!/usr/bin/env python3
"""test_text_direction.py — did the prompt move the tumour, or just perturb the image?

test_text_control.py answered "did the image CHANGE?" (yes). This answers the harder
question: "did it change in the DIRECTION the prompt asked for?"

Two instruments:
  1. DIFFERENCE MAPS  |variant - true| as a heatmap. Makes diffuse changes visible and
     shows WHERE they are. Localised near the tumour = semantic. Spread everywhere =
     global perturbation / degradation.
  2. LATERAL ASYMMETRY  mean brain intensity in the high-axis0 half minus the low-axis0
     half. Tumour is hyperintense (bright) in FLAIR, so the hemisphere holding it should
     be brighter. If the prompt genuinely relocates the tumour, flipping left->right
     should FLIP the sign of this number. If the mask wins, it barely moves.

Orientation reminder (verified earlier via nib.aff2axcodes -> ('L','P','S')):
axis 0 increases toward the patient's LEFT, and the midline sits at pixel 128.

Usage:  python scripts/test_text_direction.py [guidance]     # default 3.0
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

DEVICE, STEPS, SEED = "cuda", 200, 0
CLIP_ID  = "openai/clip-vit-base-patch32"
MIDLINE  = 128
LOBES    = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]

def flip_side(cap):
    if " left " in cap:  return cap.replace(" left ", " right ")
    if " right " in cap: return cap.replace(" right ", " left ")
    return cap

def swap_lobe(cap):
    for l in LOBES:
        if l in cap:
            return cap.replace(l, "frontal" if l != "frontal" else "occipital")
    return cap

def main():
    guidance = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load("models/text_diffusion_ema.pt", map_location=DEVICE))
    model.eval()
    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True); sched.set_timesteps(STEPS)

    ds = SliceDataset("val", return_caption=True)
    for i in range(len(ds)):
        target, cond, caption = ds[i]
        if cond[1].sum() > 100:
            break
    real   = target[0].numpy()                 # (256,256) ground-truth FLAIR
    mask   = cond[1].numpy()                   # (256,256) tumour segmentation
    cond_b = cond.unsqueeze(0).to(DEVICE)      # (1,8,256,256)

    # --- where does the MASK actually put the tumour? (ground truth for the test) ---
    a0, _ = np.nonzero(mask > 0.5)             # a0 = axis-0 coords of tumour pixels
    tum_a0 = a0.mean()
    tum_side = "patient LEFT (high axis0)" if tum_a0 >= MIDLINE else "patient RIGHT (low axis0)"
    print(f"TRUE caption : {caption}")
    print(f"mask centroid on axis0 = {tum_a0:.1f}  -> tumour is on the {tum_side}")
    print(f"guidance = {guidance}\n")

    # --- half-brain masks, used for the asymmetry metric ---
    brain = real > 0.05
    idx0  = np.arange(256)[:, None]            # (256,1) axis-0 index, broadcasts to (256,256)
    hi    = brain & (idx0 >= MIDLINE)          # patient-LEFT half of the brain
    lo    = brain & (idx0 <  MIDLINE)          # patient-RIGHT half

    def encode(text):
        tok = tokenizer([text], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return text_encoder(**tok).last_hidden_state
    enc_null = encode("")

    def generate(prompt):
        enc_c = encode(prompt)
        torch.manual_seed(SEED)
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            x_in = torch.cat([x, cond_b], dim=1)
            with torch.no_grad():
                e_c = model(x_in, t, encoder_hidden_states=enc_c).sample
                e_u = model(x_in, t, encoder_hidden_states=enc_null).sample
            x = sched.step(e_u + guidance * (e_c - e_u), t, x).prev_sample
        return np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

    variants = [("TRUE caption",  caption),
                ("opposite side", flip_side(caption)),
                ("different lobe", swap_lobe(caption)),
                ("no tumour",     "axial FLAIR slice of a brain with no tumor")]

    outs, rows = [], []
    for label, p in variants:
        g = generate(p)
        outs.append(g)
        asym = float(g[hi].mean() - g[lo].mean())      # + => patient-LEFT half brighter
        mad  = float(np.mean(np.abs(g - outs[0])[brain]))
        rows.append((label, asym, mad))
        print(f"{label:15s} | lateral asymmetry (L-R) = {asym:+.4f} | MAD(brain) = {mad:.4f}")

    base_asym = rows[0][1]
    print(f"\nREAL image asymmetry (L-R) = {float(real[hi].mean() - real[lo].mean()):+.4f}")
    print("\nHOW TO READ THIS:")
    print("  If the prompt truly relocates the tumour, 'opposite side' should have an")
    print(f"  asymmetry with the OPPOSITE SIGN to 'TRUE caption' ({base_asym:+.4f}).")
    print("  If it merely shifts by a small amount in the SAME direction, the mask is")
    print("  winning and the text is only perturbing global appearance.")

    # ---------------- figure: generations on top, difference maps below ----------------
    diffs = [np.abs(g - outs[0]) for g in outs]
    vmax  = max(float(d.max()) for d in diffs[1:]) if len(diffs) > 1 else 1.0
    n = len(variants)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7.4))
    for k, ((label, p), g, d) in enumerate(zip(variants, outs, diffs)):
        ax[0, k].imshow(g.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[0, k].set_title(f"{label}\nasym {rows[k][1]:+.4f}", fontsize=9)
        ax[0, k].axis("off")
        im = ax[1, k].imshow(d.T, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax[1, k].set_title(f"|diff vs TRUE|  max {d.max():.2f}", fontsize=9)
        ax[1, k].axis("off")
    fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02)
    fig.suptitle(f"top: generations   bottom: where the caption changed things "
                 f"(guidance {guidance}, shared scale 0-{vmax:.2f})", fontsize=10)
    out = f"outputs/text_direction_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); print("\nsaved", out)

if __name__ == "__main__":
    main()
