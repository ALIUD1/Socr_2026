#!/usr/bin/env python3
"""test_text_quant.py — laterality control measured QUANTITATIVELY over many slices.

A single slice's asymmetry is thousandths-scale noise; you cannot conclude anything from it.
This runs a controlled per-slice test and AVERAGES it with a standard error, which is what
turns "it looks a bit different" into "the effect is 0.009 +/- 0.001, so it is real".

PER SLICE (same seed, full spatial conditioning, only the SIDE WORD changes):
    force the caption to say "left"  -> generate -> asym_left
    force the caption to say "right" -> generate -> asym_right
    lat = asym_left - asym_right
Tumour is hyperintense (bright) in FLAIR, so if the word left/right controls which
hemisphere the model brightens, lat > 0. We average lat over N held-out tumour slices.

INTERPRETATION:
    mean(lat) > ~2 * SE  -> the prompt reliably controls laterality (statistically clear)
    compare two checkpoints -> whichever has the larger, tighter mean uses text more.

Usage:
    python scripts/test_text_quant.py [guidance] [checkpoint] [n_slices]
    python scripts/test_text_quant.py 3.0 models/text_diffusion_ema.pt 24
    python scripts/test_text_quant.py 3.0 models/text_dual_ema.pt      24
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from diffusers import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from src.dataset import SliceDataset
from src.model import build_text_model

DEVICE, STEPS, SEED, MIDLINE = "cuda", 100, 0, 128   # 100 steps (not 200): plenty for an intensity stat, 2x faster
CLIP_ID    = "openai/clip-vit-base-patch32"
BLANK_MASK = os.environ.get("BLANK_MASK") == "1"     # BLANK_MASK=1 -> zero the tumour-mask channel at test
MASK_CH    = 1                                        # cond order is [T1, mask, atlas*6]

def force_side(cap, side):
    """Rewrite the caption so its side word is exactly `side`."""
    for s in ("left", "right"):
        if f" {s} " in cap:
            return cap.replace(f" {s} ", f" {side} ")
    return None            # no side word -> caller skips this slice

def main():
    guidance = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    ckpt     = sys.argv[2] if len(sys.argv) > 2 else "models/text_diffusion_ema.pt"
    N        = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True); sched.set_timesteps(STEPS)
    print(f"checkpoint: {ckpt}\nguidance: {guidance}   target slices: {N}   "
          f"mask: {'BLANKED (text must place the tumour)' if BLANK_MASK else 'present'}\n")

    idx0 = np.arange(256)[:, None]      # (256,1) axis-0 index, broadcasts to a full-image mask

    def encode(text):
        tok = tokenizer([text], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return text_encoder(**tok).last_hidden_state
    enc_null = encode("")

    def generate(prompt, cond_b):
        enc_c = encode(prompt)
        torch.manual_seed(SEED)                            # same starting noise every call
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            x_in = torch.cat([x, cond_b], dim=1)
            with torch.no_grad():
                e_c = model(x_in, t, encoder_hidden_states=enc_c).sample
                e_u = model(x_in, t, encoder_hidden_states=enc_null).sample
            x = sched.step(e_u + guidance * (e_c - e_u), t, x).prev_sample
        return np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

    ds = SliceDataset("val", return_caption=True)
    lats = []
    for i in range(len(ds)):
        target, cond, caption = ds[i]
        if cond[1].sum() <= 100:                 # need a real tumour
            continue
        cap_L, cap_R = force_side(caption, "left"), force_side(caption, "right")
        if cap_L is None:                        # caption had no side word -> skip
            continue

        real  = target[0].numpy()
        brain = real > 0.05
        hi, lo = brain & (idx0 >= MIDLINE), brain & (idx0 < MIDLINE)   # L / R half-brain masks
        cond_b = cond.unsqueeze(0).to(DEVICE)
        if BLANK_MASK:
            cond_b[:, MASK_CH:MASK_CH+1] = 0.0       # remove the mask -> only text+atlas can place the tumour

        gL, gR = generate(cap_L, cond_b), generate(cap_R, cond_b)
        asym_L = gL[hi].mean() - gL[lo].mean()
        asym_R = gR[hi].mean() - gR[lo].mean()
        lat = float(asym_L - asym_R)
        lats.append(lat)
        print(f"  slice {i:5d} | asym(left)={asym_L:+.4f}  asym(right)={asym_R:+.4f}  lat={lat:+.4f}")
        if len(lats) >= N:
            break

    lats = np.array(lats)
    mean = lats.mean()
    se   = lats.std(ddof=1) / np.sqrt(len(lats))           # standard error of the mean
    frac = float((lats > 0).mean())                        # fraction of slices with the correct sign
    print(f"\n=== {ckpt} ===")
    print(f"laterality effect = {mean:+.4f} +/- {se:.4f} (SE)  over {len(lats)} slices")
    print(f"slices with correct sign (lat > 0): {frac*100:.0f}%")
    print(f"z = mean/SE = {mean/se:.1f}   (|z| > 2 => clearly non-zero, not noise)")

if __name__ == "__main__":
    main()
