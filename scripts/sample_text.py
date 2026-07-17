#!/usr/bin/env python3
"""sample_text.py — generate from the text-conditioned model with classifier-free guidance.

Shows the payoff of condition dropout: a GUIDANCE STRENGTH knob on the prompt.

Usage:
    python scripts/sample_text.py                      # uses the slice's own true caption
    python scripts/sample_text.py "axial FLAIR slice with enhancing tumor in the left frontal lobe"
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

DEVICE   = "cuda"
STEPS    = 200
CLIP_ID  = "openai/clip-vit-base-patch32"
SCALES   = [1.0, 3.0, 7.5]     # 1.0 = no guidance (prompt used as-is); higher = push harder toward prompt

def main():
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else None

    model = build_text_model().to(DEVICE)
    model.load_state_dict(torch.load("models/text_diffusion_ema.pt", map_location=DEVICE))
    model.eval()

    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE).eval()

    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(STEPS)

    # find a tumour slice in the val split (mask is cond channel 1)
    ds = SliceDataset("val", return_caption=True)
    for i in range(len(ds)):
        target, cond, caption = ds[i]
        if cond[1].sum() > 100:
            break
    prompt = user_prompt if user_prompt else caption
    print("true caption:", caption)
    print("prompt used :", prompt)

    real   = target[0].numpy()
    cond_b = cond.unsqueeze(0).to(DEVICE)

    def encode(text):
        tok = tokenizer([text], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            return text_encoder(**tok).last_hidden_state      # (1,77,512)

    enc_cond = encode(prompt)
    enc_null = encode("")                                     # the "no prompt" embedding

    samples = []
    for g in SCALES:
        torch.manual_seed(0)                                  # same noise -> isolates guidance's effect
        x = torch.randn(1, 1, 256, 256, device=DEVICE)
        for t in sched.timesteps:
            x_in = torch.cat([x, cond_b], dim=1)
            with torch.no_grad():
                e_c = model(x_in, t, encoder_hidden_states=enc_cond).sample   # WITH the prompt
                e_u = model(x_in, t, encoder_hidden_states=enc_null).sample   # WITHOUT it
            # classifier-free guidance: start from the unconditional prediction and push
            # g times further along the direction the prompt wants to move it.
            eps = e_u + g * (e_c - e_u)
            x = sched.step(eps, t, x).prev_sample
        samples.append(np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1))
        print(f"  guidance {g}: done")

    fig, ax = plt.subplots(1, len(SCALES) + 1, figsize=(4 * (len(SCALES) + 1), 4))
    ax[0].imshow(real.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax[0].set_title("REAL"); ax[0].axis("off")
    for k, (g, s) in enumerate(zip(SCALES, samples)):
        ax[k+1].imshow(s.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax[k+1].set_title(f"guidance {g}"); ax[k+1].axis("off")
    fig.suptitle(prompt, fontsize=9)
    out = f"outputs/sample_text_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110); print("saved", out)

if __name__ == "__main__":
    main()
