#!/usr/bin/env python3
"""train_text_place.py — drop ONLY the tumour mask, keep T1 + atlas, so text must place the tumour.

WHY (from the dual-dropout post-mortem):
    Dual dropout blanked ALL 8 spatial channels, so 18% of batches asked the model to draw a
    whole brain from a 5-word caption -- impossible. That inflated the loss floor (0.0056 vs
    0.0042) and corrupted the text pathway, erasing the laterality control the simple model had.

THE SURGICAL FIX:
    Blank ONLY channel 1 (the tumour mask). Keep channel 0 (T1 = full anatomy) and channels
    2-7 (the atlas = where each lobe is) ALWAYS present. So when the mask is dropped the model
    still knows the brain and the lobe map; the only missing piece is WHERE THE TUMOUR IS -- and
    the caption ("...in the left frontal lobe") plus the atlas together can supply exactly that.
    The task stays solvable, so the pathology-placement signal from text is no longer drowned.

    With MASK_DROP=0.5 / TEXT_DROP=0.1 the regimes are about:
        mask + text        0.5 * 0.9 = 45%   normal task
        text-only-places   0.5 * 0.9 = 45%   <- mask gone, text+atlas must place the tumour
        mask, no text      0.5 * 0.1 =  5%
        neither placer     0.5 * 0.1 =  5%   only ~5% has no placement signal (and it is a
                                             SMALL loss hit: tumour is <1% of pixels, and T1
                                             still makes the rest of the image solvable)

Checkpoints -> models/text_place_{last,ema}.pt (leaves every prior model intact).
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
BATCH     = 16
EPOCHS    = 20
TEXT_DROP = 0.10          # blank the caption 10% of the time (keeps classifier-free guidance)
MASK_DROP = 0.50          # blank ONLY the tumour-mask channel 50% of the time
MASK_CH   = 1             # cond channel order is [T1, mask, atlas*6] -> the mask is index 1
CLIP_ID   = "openai/clip-vit-base-patch32"
CKPT_DIR  = "models"
TAG       = os.environ.get("TAG_TEXT", "text_place")   # keeps parallel runs apart

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    betas      = torch.linspace(1e-4, 0.02, T, device=DEVICE)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    print(f"captions: {os.environ.get('CAPTIONS_CSV','captions.csv')} | tag: {TAG}", flush=True)
    loader = DataLoader(SliceDataset("train", return_caption=True),
                        batch_size=BATCH, shuffle=True, num_workers=4)

    tokenizer    = CLIPTokenizer.from_pretrained(CLIP_ID)
    text_encoder = CLIPTextModel.from_pretrained(CLIP_ID).to(DEVICE)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    model  = build_text_model().to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    ema    = EMAModel(model.parameters(), decay=0.999)

    for epoch in range(EPOCHS):
        running = 0.0
        seen = {"mask+text": 0, "text_places": 0, "mask_only": 0, "neither": 0}

        for target, cond, captions in loader:
            target = target.to(DEVICE) * 2 - 1
            cond   = cond.to(DEVICE)                          # (B, 8, 256, 256): [T1, mask, atlas*6]
            B = target.shape[0]

            keep_txt  = torch.rand(B, device=DEVICE) >= TEXT_DROP    # (B,) keep the caption?
            keep_mask = torch.rand(B, device=DEVICE) >= MASK_DROP    # (B,) keep the mask channel?

            # ---- blank ONLY the mask channel for the dropped samples ----
            # cond[:, 1:2] is (B,1,256,256) = just the mask. Multiply it by a per-sample 0/1.
            # T1 (channel 0) and the atlas (channels 2-7) are untouched -> anatomy always present.
            cond[:, MASK_CH:MASK_CH+1] = cond[:, MASK_CH:MASK_CH+1] * keep_mask.float().view(B, 1, 1, 1)

            captions = [c if k else "" for c, k in zip(captions, keep_txt.tolist())]
            tok = tokenizer(list(captions), padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                enc = text_encoder(**tok).last_hidden_state   # (B, 77, 512)

            for km, kt in zip(keep_mask.tolist(), keep_txt.tolist()):
                seen["mask+text" if (km and kt) else
                     "mask_only" if km else
                     "text_places" if kt else "neither"] += 1

            t     = torch.randint(0, T, (B,), device=DEVICE)
            noise = torch.randn_like(target)
            a     = alpha_bars[t].view(B, 1, 1, 1)
            noisy = a.sqrt() * target + (1 - a).sqrt() * noise
            x_in  = torch.cat([noisy, cond], dim=1)           # (B, 9, 256, 256)

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
