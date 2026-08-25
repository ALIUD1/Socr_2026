#!/usr/bin/env python3
"""augment_text.py — atlas-guided synthetic tumour + MATCHING text prompt, in one pipeline.

This merges the two 2D threads:
  * augment_tumor.py  grows an organic tumour mask that conforms to a chosen atlas lobe
  * the text-conditioned model  steers generation from a natural-language caption

The key design point is CONSISTENCY BY CONSTRUCTION. Rather than writing the caption by hand
(which could disagree with the mask), we derive it from the synthetic mask using the exact same
function that captioned the real training data (04_captions.caption_for). So the caption is
guaranteed to describe the mask, because it is READ OFF the mask -- the same way every training
caption was. Mask and text reinforce each other instead of competing.

Output is a labelled TRIPLE -- generated FLAIR, ground-truth mask, and the caption describing it
-- which is richer than the (image, mask) pairs augment_tumor.py produces, and is exactly what a
text-conditioned downstream model would need.

Usage
    python scripts/augment_text.py                     # random lobe
    python scripts/augment_text.py temporal            # a specific lobe
    python scripts/augment_text.py temporal 5          # ... and 5 samples
Env
    TEXT_CKPT   checkpoint to use (default models/text_place_ema.pt)
    GUIDANCE    classifier-free guidance strength (default 3.0)
    AUG_STEPS   DDIM steps (default 200; lower for a fast CPU test)
"""
import sys, os, glob, random, csv, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import torch, numpy as np
from datetime import datetime
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from augment_tumor import synth_mask, LOBES, ATLAS0, PROB_TH, MIN_BRAIN, MIN_LOBE
from src.model import build_text_model

# 04_captions.py starts with a digit, so a normal `import` is impossible -- load it by path.
# Reusing its caption_for() is what guarantees the caption matches the mask.
_spec = importlib.util.spec_from_file_location("captions", os.path.join(HERE, "04_captions.py"))
_captions = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_captions)
caption_for = _captions.caption_for

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
STEPS    = int(os.environ.get("AUG_STEPS", "200"))
GUIDANCE = float(os.environ.get("GUIDANCE", "3.0"))
CKPT     = os.environ.get("TEXT_CKPT", "models/text_place_ema.pt")
# MASK_MODE=use   -> the synthetic mask is fed as conditioning (mask drives placement)
# MASK_MODE=blank -> the mask channel is ZEROED, so only the caption + atlas can place the
#   tumour. The synthetic mask is still built and still captions the prompt, but it becomes
#   GROUND TRUTH rather than input: measuring lesion contrast inside it then tests whether TEXT
#   put the pathology where the caption said. This only works with a mask-dropout checkpoint
#   (text_place / text_rich), because a mask-always model never learned to cope without one.
MASK_MODE = os.environ.get("MASK_MODE", "use")
CLIP_ID  = "openai/clip-vit-base-patch32"
OUT_DIR  = "outputs/augment_text"

def find_slice(lobe_idx):
    """A tumour-FREE slice with plenty of brain and this lobe well represented, so we are
    cleanly ADDING one controlled tumour to an otherwise healthy slice."""
    for f in sorted(glob.glob("data/processed/slices/val/*.npy")):
        s = np.load(f).astype(np.float32)
        brain = s[1] > 0.05                       # channel 1 = T1
        al = s[ATLAS0 + lobe_idx]
        if (brain.sum() > MIN_BRAIN and (s[2] > 0.5).sum() < 50
                and ((al > PROB_TH) & brain).sum() > MIN_LOBE):
            return s, al, brain
    raise RuntimeError("no suitable tumour-free slice found for this lobe")

def main():
    lobe = sys.argv[1] if len(sys.argv) > 1 else random.choice(LOBES)
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    li   = LOBES.index(lobe)                      # ValueError = lobe name typo
    print(f"device {DEVICE} | {STEPS} DDIM steps | guidance {GUIDANCE} | checkpoint {CKPT}")
    print(f"mask mode: {MASK_MODE}" +
          ("  (mask conditions the model)" if MASK_MODE == "use"
           else "  (mask BLANKED -> text+atlas must place the tumour; mask is ground truth)"))

    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    txt = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(STEPS)

    def encode(text):
        t = tok([text], padding="max_length", max_length=77, truncation=True,
                return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return txt(**t).last_hidden_state          # (1,77,512)
    enc_null = encode("")                              # the unconditional branch for CFG

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, panels = [], []

    for i in range(n):
        rng = np.random.default_rng(i)                 # different tumour shape per sample
        stack, atlas_lobe, brain = find_slice(li)
        m, centre = synth_mask(atlas_lobe, brain, rng)
        if m is None:
            print(f"  sample {i}: could not grow a lesion, skipping"); continue

        # --- the consistency step: read the caption OFF our synthetic mask ---
        fields, caption = caption_for(m, stack[ATLAS0:ATLAS0 + 6])

        # verification: does the derived caption actually name the lobe we asked for?
        ok = (fields["lobe"] == lobe)
        print(f"\n  sample {i}: requested '{lobe}' -> caption says '{fields['lobe']}' "
              f"{'OK' if ok else 'MISMATCH (lesion straddles a boundary)'}")
        print(f"    side={fields['side']}  components={fields['components']}")
        print(f"    caption: {caption}")

        # --- conditioning: swap OUR mask into channel 2, keep the patient T1 + atlas ---
        cond_stack = stack.copy()
        cond_stack[2] = m
        cond = torch.from_numpy(cond_stack[1:9]).unsqueeze(0).to(DEVICE)   # (1,8,256,256)
        if MASK_MODE == "blank":
            cond[:, 1:2] = 0.0        # channel 1 = mask; text + atlas must now do the placing

        enc_c = encode(caption)
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            x_in = torch.cat([x, cond], dim=1)
            with torch.no_grad():
                e_c = model(x_in, t, encoder_hidden_states=enc_c).sample
                e_u = model(x_in, t, encoder_hidden_states=enc_null).sample
            # classifier-free guidance: push g times along the direction the prompt wants
            x = sched.step(e_u + GUIDANCE * (e_c - e_u), t, x).prev_sample
        flair = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)

        # tumour-fidelity check: FLAIR lesions are hyperintense, so signal inside the mask
        # should exceed signal in the surrounding healthy brain. This is the number that says
        # the pathology actually rendered, rather than just "the image looks fine".
        inside  = flair[m > 0]
        outside = flair[(m == 0) & brain]
        contrast = float(inside.mean() - outside.mean()) if inside.size and outside.size else float("nan")
        verdict = ("lesion rendered" if contrast > 0.02 else "WEAK/absent")
        if MASK_MODE == "blank":
            verdict += "  <- placed by TEXT alone" if contrast > 0.02 else "  <- text did not place it"
        print(f"    lesion contrast (inside - outside) = {contrast:+.3f}  {verdict}")

        base = f"aug_{lobe}_{stamp}_{i:02d}"
        np.save(os.path.join(OUT_DIR, base + ".npy"), np.stack([flair, m]))   # (2,256,256)
        rows.append([base + ".npy", lobe, fields["side"], fields["components"],
                     MASK_MODE, f"{contrast:.4f}", caption])
        panels.append((stack[1], m, flair, caption, contrast))

    if not rows:
        raise SystemExit("nothing generated")

    csv_path = os.path.join(OUT_DIR, "captions.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "lobe", "side", "components", "mask_mode",
                        "lesion_contrast", "caption"])
        w.writerows(rows)

    fig, ax = plt.subplots(len(panels), 3, figsize=(12, 4 * len(panels)))
    ax = np.atleast_2d(ax)
    for r, (t1, m, flair, cap, c) in enumerate(panels):
        ax[r, 0].imshow(t1.T,    cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[r, 0].set_title("patient T1 (healthy)", fontsize=9)
        ax[r, 1].imshow(m.T,     cmap="hot",  origin="lower")
        ax[r, 1].set_title("synthetic mask (atlas-fitted)", fontsize=9)
        ax[r, 2].imshow(flair.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[r, 2].set_title(f"GENERATED FLAIR  (lesion contrast {c:+.3f})", fontsize=9)
        for a in ax[r]:
            a.axis("off")
        ax[r, 0].set_xlabel(cap, fontsize=7)
    fig.suptitle(f"text + atlas-guided tumour augmentation — '{lobe}' lobe "
                 f"(mask {'used as conditioning' if MASK_MODE == 'use' else 'BLANKED: text placed it'})",
                 fontsize=12)
    out = os.path.join(OUT_DIR, f"aug_{lobe}_{stamp}.png")
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print(f"\nsaved {len(rows)} labelled triples -> {OUT_DIR}")
    print(f"saved {out}")
    print(f"captions -> {csv_path}")

if __name__ == "__main__":
    main()
