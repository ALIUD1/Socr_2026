#!/usr/bin/env python3
"""train_text.py — text-conditioned conditional diffusion (spatial channels + CLIP text).

Difference from train.py (the paper model):
  - the U-Net is build_text_model() (has cross-attention blocks)
  - each batch also carries a caption string
  - captions are turned into vectors by a FROZEN CLIP text encoder
  - CONDITION DROPOUT: some captions are blanked to "" during training, which is what
    later enables classifier-free guidance (a strength knob on the prompt) at sampling.

Checkpoints are written to models/text_diffusion_{last,ema}.pt so the paper model's
models/diffusion_ema.pt is never overwritten.

NOTE: the CLIP weights download from HuggingFace. Great Lakes compute nodes usually have
no internet, so run this ONCE on a login node first to warm the cache:
    python -c "from transformers import CLIPTokenizer, CLIPTextModel; \
               CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32'); \
               CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')"
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from transformers import CLIPTokenizer, CLIPTextModel

from src.dataset import SliceDataset
from src.model import build_text_model

DEVICE, T, LR = "cuda", 1000, 1e-4
BATCH   = 16          # smaller than the paper model's 32: cross-attention adds memory. Raise if it fits.
EPOCHS  = 20
DROP_PROB = 0.1       # blank out the caption 10% of the time -> teaches the "no prompt" case
CLIP_ID = "openai/clip-vit-base-patch32"
CKPT_DIR = "models"

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ---- diffusion schedule (identical to the paper model) ----
    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    # ---- data: now returns (target, cond, caption) ----
    loader = DataLoader(SliceDataset("train", return_caption=True),
                        batch_size=BATCH, shuffle=True, num_workers=4)

    # ---- text encoder: FROZEN. We only use it to read captions, never train it. ----
    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE)
    text_encoder.requires_grad_(False)          # no gradients -> its weights never change
    text_encoder.eval()

    model  = build_text_model().to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    ema    = EMAModel(model.parameters(), decay=0.999)

    for epoch in range(EPOCHS):
        running = 0.0
        for target, cond, captions in loader:
            target = target.to(DEVICE) * 2 - 1      # FLAIR [0,1] -> [-1,1]
            cond   = cond.to(DEVICE)                # T1 + mask + 6 atlas, left in [0,1]
            B = target.shape[0]

            # ---- CONDITION DROPOUT ----
            # Replace some captions with the empty string. The model therefore learns BOTH
            # "denoise given this prompt" and "denoise given no prompt at all". Having both
            # in one network is exactly what classifier-free guidance needs at sampling time.
            captions = ["" if random.random() < DROP_PROB else c for c in captions]

            # ---- text -> vectors ----
            tok = tokenizer(list(captions), padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():                              # frozen encoder: no grad needed
                enc = text_encoder(**tok).last_hidden_state    # (B, 77, 512)

            # ---- forward diffusion: noise the FLAIR ----
            t     = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target)
            a     = alpha_bars[t].view(B, 1, 1, 1)
            noisy = a.sqrt() * target + (1 - a).sqrt() * noise

            # spatial conditioning still enters by channel-concat, exactly as before
            x_in = torch.cat([noisy, cond], dim=1)             # (B, 9, 256, 256)

            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                # the ONLY new argument: encoder_hidden_states = the text the U-Net attends to
                pred = model(x_in, t, encoder_hidden_states=enc).sample
                loss = F.mse_loss(pred, noise)                 # still epsilon-prediction
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ema.step(model.parameters())
            running += loss.item()

        print(f"epoch {epoch}: loss {running/len(loader):.4f}", flush=True)

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "text_diffusion_last.pt"))
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "text_diffusion_ema.pt"))
        ema.restore(model.parameters())

if __name__ == "__main__":
    main()
