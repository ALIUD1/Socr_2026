#!/usr/bin/env python3
"""unified_demo.py — ONE model, four conditioning regimes, side by side.

The point: there is no need for separate "spatial" and "text" models. The text architecture
(UNet2DConditionModel) is a strict SUPERSET of the spatial one -- identical 9 input channels
(noisy FLAIR + T1 + mask + 6 atlas) PLUS cross-attention to a caption. Train it with BOTH
condition dropouts, as train_text_place.py does (text 10%, mask 50%), and a single checkpoint
covers every regime:

    mask + text   both given                     (the normal task)
    mask only     caption blanked to ""          (behaves like the spatial-only paper model)
    text only     mask channel zeroed            (caption + atlas must place the pathology)
    neither       both withheld                  (unconditional given the anatomy)

This script runs all four on the SAME slice with the SAME random seed, so the only thing that
varies is which conditioning is supplied. It reports per-regime numbers as well as the figure,
because "one model does everything" is a quantitative claim, not a visual impression.

Usage
    python scripts/unified_demo.py                       # default checkpoint
    python scripts/unified_demo.py 3.0 24                # guidance, val-slice index
Env
    CKPT       checkpoint (default models/text_place_ema.pt; use text_rich_ema.pt once trained)
    STEPS      DDIM steps (default 200)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

try:
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False

from src.dataset import SliceDataset
from src.model import build_text_model

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
CKPT    = os.environ.get("CKPT", "models/text_place_ema.pt")
STEPS   = int(os.environ.get("STEPS", "200"))
SEED    = 0
CLIP_ID = "openai/clip-vit-base-patch32"

def main():
    guidance = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    which    = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    txt = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(STEPS)
    print(f"device {DEVICE} | checkpoint {CKPT} | {STEPS} steps | guidance {guidance}")

    # a held-out slice that actually has a tumour, so every regime has something to place
    ds = SliceDataset("val", return_caption=True)
    seen = 0
    for i in range(len(ds)):
        target, cond, caption = ds[i]
        if cond[1].sum() > 100:
            if seen == which:
                break
            seen += 1
    real  = target[0].numpy()
    mask  = cond[1].numpy()
    brain = real > 0.05
    print(f"val slice {i} | caption: {caption}\n")

    def encode(text):
        t = tok([text], padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return txt(**t).last_hidden_state
    enc_null = encode("")

    def generate(use_mask, use_text):
        """Same seed and same slice every call -- only the SUPPLIED CONDITIONING changes."""
        c = cond.clone().unsqueeze(0).to(DEVICE)          # (1,8,256,256)
        if not use_mask:
            c[:, 1:2] = 0.0                               # channel 1 = tumour mask
        enc_c = encode(caption) if use_text else enc_null
        torch.manual_seed(SEED)
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            x_in = torch.cat([x, c], dim=1)
            with torch.no_grad():
                e_c = model(x_in, t, encoder_hidden_states=enc_c).sample
                e_u = model(x_in, t, encoder_hidden_states=enc_null).sample
            # With no caption, enc_c == enc_null so (e_c - e_u) is zero and guidance correctly
            # has no effect -- the formula degenerates to the unconditional prediction.
            x = sched.step(e_u + guidance * (e_c - e_u), t, x).prev_sample
        return np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

    modes = [("mask + text", True,  True),
             ("mask only",   True,  False),
             ("text only",   False, True),
             ("neither",     False, False)]

    # CONTROL: the same contrast measured on the REAL image. Without it the generated number
    # is uninterpretable -- necrotic core is HYPOintense in FLAIR, so a negative contrast can be
    # entirely correct. What matters is whether the model MATCHES the real image's contrast.
    r_in, r_out = real[mask > 0], real[(mask == 0) & brain]
    real_lc = float(r_in.mean() - r_out.mean()) if r_in.size and r_out.size else float("nan")
    comp = {"necrotic": int((np.rint(mask) == 1).sum()), "edema": int((np.rint(mask) == 2).sum()),
            "enhancing": int((np.rint(mask) == 3).sum())}
    print(f"mask composition (voxels): {comp}")
    print(f"REAL lesion contrast = {real_lc:+.4f}   <- the target every regime should match")
    print()
    print(f"{'regime':<14}{'PSNR':>8}{'SSIM':>8}{'lesion':>10}{'vs REAL':>10}")
    outs, stats = [], []
    for name, um, ut in modes:
        g = generate(um, ut)
        mse = float(((g - real) ** 2)[brain].mean())
        p   = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        s   = _ssim(real, g, data_range=1.0) if HAVE_SSIM else float("nan")
        # lesion contrast: FLAIR pathology is hyperintense, so signal inside the tumour mask
        # should exceed the surrounding healthy brain. This is what says the pathology rendered.
        inside, outside = g[mask > 0], g[(mask == 0) & brain]
        lc = float(inside.mean() - outside.mean()) if inside.size and outside.size else float("nan")
        outs.append(g); stats.append((p, s, lc))
        print(f"{name:<14}{p:8.2f}{s:8.3f}{lc:+10.4f}{lc - real_lc:+10.4f}")

    print("\nHOW TO READ THIS")
    print("  All four rows come from ONE checkpoint -- the model covers every regime because it")
    print("  was trained with both text dropout and mask dropout.")
    print("  'mask only' should look like the spatial-only paper model (that is the same task).")
    print("  'text only' reproducing the REAL lesion contrast means the CAPTION placed the")
    print("  pathology with no mask at all -- a capability a spatial-only model cannot have.")
    print("  Judge lesion contrast by the 'vs REAL' column, NOT its raw sign: necrotic core is")
    print("  hypointense in FLAIR, so a negative contrast is correct when necrosis dominates.")
    print("  If 'mask + text' and 'text only' are IDENTICAL, the mask channel is being ignored")
    print("  -- the caption already encodes the tumour location, so mask dropout can teach the")
    print("  model to rely on text alone. Lower MASK_DROP if the mask must stay load-bearing.")

    # ---- crop coordinates centred on the tumour mask, with margin, clamped to the image ----
    ys, xs = np.nonzero(mask > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = max(int(max(ys.max()-ys.min(), xs.max()-xs.min()) * 1.5) // 2 + 10, 20)
    y0, y1 = max(cy-half, 0), min(cy+half, real.shape[0])
    x0, x1 = max(cx-half, 0), min(cx+half, real.shape[1])

    def crop(img):
        return img[y0:y1, x0:x1]

    fig, ax = plt.subplots(2, 5, figsize=(20, 9))
    # top row: full slice with a red box marking the crop region
    ax[0, 0].imshow(real.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax[0, 0].set_title("REAL", fontsize=10); ax[0, 0].axis("off")
    for a in [ax[0, 0]] + [ax[0, k+1] for k in range(4)]:
        rect = plt.Rectangle((y0, x0), y1-y0, x1-x0, edgecolor="red", facecolor="none", lw=1.5)
        a.add_patch(rect)
    for k, ((name, _, _), g) in enumerate(zip(modes, outs)):
        p, s, lc = stats[k]
        ax[0, k+1].imshow(g.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[0, k+1].set_title(f"{name}\nPSNR {p:.1f}  SSIM {s:.2f}\nlesion {lc:+.3f}", fontsize=9)
        ax[0, k+1].axis("off")

    # bottom row: ZOOMED crop of the red box, with the mask outline overlaid, so the tumour
    # region itself -- not the whole 256x256 slice -- is what you're actually comparing.
    ax[1, 0].imshow(crop(real).T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax[1, 0].contour(crop(mask).T, levels=[0.5], colors="lime", linewidths=1.5)
    ax[1, 0].set_title("REAL (zoomed, mask outline)", fontsize=9); ax[1, 0].axis("off")
    for k, g in enumerate(outs):
        ax[1, k+1].imshow(crop(g).T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[1, k+1].contour(crop(mask).T, levels=[0.5], colors="lime", linewidths=1.5)
        ax[1, k+1].set_title(f"{modes[k][0]} (zoomed)", fontsize=9); ax[1, k+1].axis("off")

    fig.suptitle(f"ONE model, four conditioning regimes — {os.path.basename(CKPT)}\n"
                 f"top: full slice (red box = crop region)   bottom: zoomed on the tumour "
                 f"(green = mask boundary)", fontsize=11)
    os.makedirs("outputs/unified", exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = f"outputs/unified/unified_{stamp}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110)

    # The figure does not show the REAL contrast -- it is printed only. Write the numbers to a
    # sidecar so a run is not lost when the terminal scrollback is.
    txt = f"outputs/unified/unified_{stamp}.txt"
    with open(txt, "w") as f:
        f.write(f"checkpoint {CKPT} | steps {STEPS} | guidance {guidance} | val slice {i}\n"
                f"caption: {caption}\n"
                f"mask composition (voxels): {comp}\n"
                f"REAL lesion contrast = {real_lc:+.4f}\n\n"
                f"{'regime':<14}{'PSNR':>8}{'SSIM':>8}{'lesion':>10}{'vs REAL':>10}\n")
        for (name, _, _), (p, s, lc) in zip(modes, stats):
            f.write(f"{name:<14}{p:8.2f}{s:8.3f}{lc:+10.4f}{lc - real_lc:+10.4f}\n")
    print(f"\nsaved {out}\nsaved {txt}")

if __name__ == "__main__":
    main()
