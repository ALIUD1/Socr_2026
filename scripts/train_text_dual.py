#!/usr/bin/env python3
"""train_text_dual.py — DUAL condition dropout: blank the caption AND the spatial channels.

WHY THIS EXISTS
    train_text.py drops only the caption, so the spatial conditioning (T1 + mask + atlas)
    was present in 100% of training batches. The model therefore never had to learn where
    pathology goes from words -- the mask always told it, pixel-perfectly. The direction
    test confirmed the consequence: text steers global intensity but cannot relocate a
    lesion.

THE FIX
    Independently blank each modality. Now the model sometimes sees batches where TEXT IS
    THE ONLY SIGNAL, and the loss can only go down if it learns to place pathology from the
    caption. With TEXT_DROP=0.10 and SPATIAL_DROP=0.20 the four training regimes appear at
    roughly:
        both present  0.90 * 0.80 = 72%   <- the normal task
        text only     0.90 * 0.20 = 18%   <- forces spatial grounding from words
        image only    0.10 * 0.80 =  8%   <- the unconditional-text branch CFG needs
        neither       0.10 * 0.20 =  2%   <- fully unconditional
    The probabilities are deliberately ASYMMETRIC: we handicap the dominant signal (spatial)
    harder than the weak one (text), because the whole point is to stop the mask from
    monopolising the job.

Checkpoints -> models/text_dual_{last,ema}.pt, so neither the paper model
(models/diffusion_ema.pt) nor the first text model (models/text_diffusion_ema.pt) is touched.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from transformers import CLIPTokenizer, CLIPTextModel

from src.dataset import SliceDataset
from src.model import build_text_model

DEVICE, T, LR = "cuda", 1000, 1e-4
BATCH        = 16
EPOCHS       = 20
TEXT_DROP    = 0.10          # probability a given sample's caption is blanked
SPATIAL_DROP = 0.20          # probability a given sample's T1+mask+atlas is blanked
CLIP_ID      = "openai/clip-vit-base-patch32"
CKPT_DIR     = "models"
TAG          = "text_dual"

# Set to "models/text_diffusion_ema.pt" to WARM-START from the first text model. That trains
# much faster, but it confounds "more total training" with "dual dropout". None = from
# scratch, which is the clean A/B against train_text.py at equal epochs.
INIT_FROM = None

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    loader = DataLoader(SliceDataset("train", return_caption=True),
                        batch_size=BATCH, shuffle=True, num_workers=4)

    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    model = build_text_model().to(DEVICE)
    if INIT_FROM:
        model.load_state_dict(torch.load(INIT_FROM, map_location=DEVICE))
        print(f"warm-started from {INIT_FROM}", flush=True)

    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    ema    = EMAModel(model.parameters(), decay=0.999)

    for epoch in range(EPOCHS):
        running = 0.0
        seen = {"both": 0, "text_only": 0, "image_only": 0, "neither": 0}

        for target, cond, captions in loader:
            target = target.to(DEVICE) * 2 - 1
            cond   = cond.to(DEVICE)                       # (B, 8, 256, 256)
            B = target.shape[0]

            # ---- draw both dropout decisions per SAMPLE, on the GPU ----
            # True  = keep this modality for that sample
            keep_txt = torch.rand(B, device=DEVICE) >= TEXT_DROP      # (B,) bool
            keep_img = torch.rand(B, device=DEVICE) >= SPATIAL_DROP   # (B,) bool

            # ---- blank the SPATIAL conditioning by multiplying it away ----
            # An all-zero conditioning tensor never occurs naturally (a real slice always has
            # nonzero T1/atlas somewhere), so zeros are an unambiguous "nothing given" signal.
            mask_img = keep_img.float().view(B, 1, 1, 1)              # (B,1,1,1) -> broadcasts
            cond = cond * mask_img                                    # dropped rows become all-zero

            # ---- blank the CAPTION by swapping in the empty string ----
            captions = [c if k else "" for c, k in zip(captions, keep_txt.tolist())]

            tok = tokenizer(list(captions), padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                enc = text_encoder(**tok).last_hidden_state           # (B, 77, 512)

            # ---- bookkeeping so we can VERIFY the regimes actually occur ----
            for kt, ki in zip(keep_txt.tolist(), keep_img.tolist()):
                seen["both" if (kt and ki) else
                     "text_only" if kt else
                     "image_only" if ki else "neither"] += 1

            # ---- standard DDPM forward + epsilon-prediction (unchanged) ----
            t     = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target)
            a     = alpha_bars[t].view(B, 1, 1, 1)
            noisy = a.sqrt() * target + (1 - a).sqrt() * noise
            x_in  = torch.cat([noisy, cond], dim=1)                   # (B, 9, 256, 256)

            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                pred = model(x_in, t, encoder_hidden_states=enc).sample
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ema.step(model.parameters())
            running += loss.item()

        tot = max(sum(seen.values()), 1)
        pct = {k: f"{100*v/tot:.0f}%" for k, v in seen.items()}
        print(f"epoch {epoch}: loss {running/len(loader):.4f} | regimes {pct}", flush=True)

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{TAG}_last.pt"))
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{TAG}_ema.pt"))
        ema.restore(model.parameters())

if __name__ == "__main__":
    main()
