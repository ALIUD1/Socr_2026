#!/usr/bin/env python3
"""probe_3d.py — feasibility probe: does a 3D diffusion U-Net fit on the GPU, and at what batch?

A 3D volume has ~150x the voxels of one 2D slice, so before building any 3D pipeline we measure the
memory envelope. This builds a representative 3D U-Net at a chosen resolution, runs ONE
forward+backward with a dummy batch, and reports parameter count + peak GPU memory. It sweeps batch
sizes until it runs out of memory.

This is a MEMORY test, not a real model -- weights are random and the loss is meaningless. The only
output that matters is "batch B fits in X GB".

Usage:  python scripts/probe_3d.py [resolution]     # default 64  (i.e. 64x64x64 volumes)
Must run on the RTX 6000 (the only GPU your torch build supports).
"""
import sys
import torch, torch.nn as nn

RES    = int(sys.argv[1]) if len(sys.argv) > 1 else 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def blk(ci, co):
    """Two 3D conv layers with GroupNorm + SiLU -- the standard U-Net building block, in 3D."""
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
        nn.Conv3d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())

class UNet3D(nn.Module):
    """A compact 4-level 3D U-Net (channels 32-64-128-256). Enough to gauge memory realistically."""
    def __init__(self, cin=2, cout=1, ch=(32, 64, 128, 256)):
        super().__init__()
        self.d0, self.d1, self.d2 = blk(cin, ch[0]), blk(ch[0], ch[1]), blk(ch[1], ch[2])
        self.mid  = blk(ch[2], ch[3])
        self.pool = nn.MaxPool3d(2)                                  # halve each spatial dim
        self.up2, self.u2 = nn.ConvTranspose3d(ch[3], ch[2], 2, 2), blk(ch[2]*2, ch[2])
        self.up1, self.u1 = nn.ConvTranspose3d(ch[2], ch[1], 2, 2), blk(ch[1]*2, ch[1])
        self.up0, self.u0 = nn.ConvTranspose3d(ch[1], ch[0], 2, 2), blk(ch[0]*2, ch[0])
        self.out = nn.Conv3d(ch[0], cout, 1)

    def forward(self, x):
        e0 = self.d0(x)
        e1 = self.d1(self.pool(e0))
        e2 = self.d2(self.pool(e1))
        m  = self.mid(self.pool(e2))                                 # bottleneck
        d2 = self.u2(torch.cat([self.up2(m),  e2], 1))               # cat = skip connection
        d1 = self.u1(torch.cat([self.up1(d2), e1], 1))
        d0 = self.u0(torch.cat([self.up0(d1), e0], 1))
        return self.out(d0)

def probe(batch):
    torch.cuda.reset_peak_memory_stats()
    model = UNet3D().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x     = torch.randn(batch, 2, RES, RES, RES, device=DEVICE)      # noisy FLAIR + 1 cond channel
    tgt   = torch.randn(batch, 1, RES, RES, RES, device=DEVICE)
    with torch.amp.autocast("cuda"):
        loss = ((model(x) - tgt) ** 2).mean()
    loss.backward(); opt.step()                                      # a full training step
    peak = torch.cuda.max_memory_allocated() / 1e9
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    del model, opt, x, tgt, loss
    torch.cuda.empty_cache()
    return nparams, peak

def main():
    print(f"device={DEVICE}   resolution={RES}^3")
    if DEVICE == "cpu":
        print("no GPU -> cannot measure GPU memory. Run this on the RTX 6000.")
        return
    print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB\n")
    for b in [1, 2, 4, 8, 16, 32]:
        try:
            n, peak = probe(b)
            print(f"batch {b:2d}: {n:5.1f}M params, peak GPU {peak:5.1f} GB   -> FITS")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"batch {b:2d}: OUT OF MEMORY")
                torch.cuda.empty_cache()
                break
            raise

if __name__ == "__main__":
    main()
