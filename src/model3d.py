"""model3d.py — the 3D diffusion denoiser, defined once and shared by the 3D scripts.

The 2D model came from diffusers (UNet2DModel). For 3D there is no drop-in equivalent that fits
our use (diffusers' 3D U-Nets are video-oriented), so this is a compact 3D U-Net written directly
in PyTorch. It is the same architecture family as the 2D one, with every 2D op swapped for its 3D
counterpart: Conv2d -> Conv3d (a 3x3x3 kernel sliding through a volume instead of a 3x3 square
sliding over an image).

Same conditioning contract as 2D: 9 input channels = noisy FLAIR + 8 conditioning
(T1 + tumour mask + 6 atlas lobes), 1 output channel = the predicted noise.

Resolution ladder at 64^3:  64 -> 32 -> 16 -> 8, channels 32 -> 64 -> 128 -> 256.
Self-attention runs ONLY at the 8^3 bottleneck (512 tokens = cheap). At 64^3 it would be
262,144 tokens attending to each other, which is impossible -- the same constraint that kept
attention at the deepest level in the 2D model.
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim):
    """Map integer timesteps -> smooth vectors the network can condition on.

    t: (B,) int tensor of timesteps.  Returns (B, dim).
    Uses the standard sinusoidal encoding: a bank of sine/cosine waves at geometrically spaced
    frequencies. Nearby timesteps get similar vectors (so the model can interpolate), while the
    many different frequencies keep every timestep uniquely identifiable.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]          # (B, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)

class ResBlock3D(nn.Module):
    """Conv-norm-SiLU twice, with the timestep embedding added in between, plus a residual skip."""
    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.emb   = nn.Linear(temb_dim, cout)        # projects the time vector to this block's width
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        # if the channel count changes, the residual needs a 1x1x1 conv to match shapes
        self.skip  = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))                       # (B,cout,D,H,W)
        # broadcast the per-sample time vector over every voxel: (B,cout) -> (B,cout,1,1,1)
        h = h + self.emb(F.silu(temb))[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class ResStack(nn.Module):
    """`n` ResBlock3D in series. Only the first changes the channel count.

    DEPTH is what the first 3D build was missing: it had ONE ResBlock per resolution level
    (8 total) while the working 2D model has layers_per_block=2 (~22 total). Width gives you
    more features per step; DEPTH gives you more refinement steps, which is what produces fine
    detail. A wide-but-shallow net learns coarse shape well and stalls on contrast -- exactly
    the failure we observed.
    """
    def __init__(self, cin, cout, temb_dim, n):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock3D(cin if i == 0 else cout, cout, temb_dim) for i in range(n)])

    def forward(self, x, temb):
        for b in self.blocks:
            x = b(x, temb)
        return x

class Attn3D(nn.Module):
    """Self-attention over the volume — every voxel attends to every other voxel.
    Only affordable at the 8^3 bottleneck (512 tokens -> a 512x512 attention matrix)."""
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv  = nn.Conv3d(c, c * 3, 1)            # one 1x1x1 conv produces Q, K and V at once
        self.proj = nn.Conv3d(c, c, 1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W                                              # number of tokens (voxels)
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, N).unbind(1)   # each (B,C,N)
        att = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)  # (B,N,N)
        h = (v @ att.transpose(1, 2)).reshape(B, C, D, H, W)
        return x + self.proj(h)                                    # residual

class UNet3D(nn.Module):
    def __init__(self, in_ch=9, out_ch=1, ch=(32, 64, 128, 256), temb_dim=128, blocks=2):
        super().__init__()
        self.temb_dim = temb_dim
        emb = temb_dim * 4
        # a small MLP gives the raw sinusoidal encoding room to become a useful representation
        self.temb = nn.Sequential(nn.Linear(temb_dim, emb), nn.SiLU(), nn.Linear(emb, emb))

        self.stem = nn.Conv3d(in_ch, ch[0], 3, padding=1)          # 9 -> 32 channels @64^3
        self.d0, self.d1, self.d2 = (ResStack(ch[0], ch[0], emb, blocks),
                                     ResStack(ch[0], ch[1], emb, blocks),
                                     ResStack(ch[1], ch[2], emb, blocks))
        # stride-2 convs halve each spatial dimension: 64 -> 32 -> 16 -> 8
        self.down0 = nn.Conv3d(ch[0], ch[0], 3, stride=2, padding=1)
        self.down1 = nn.Conv3d(ch[1], ch[1], 3, stride=2, padding=1)
        self.down2 = nn.Conv3d(ch[2], ch[2], 3, stride=2, padding=1)

        self.mid1, self.attn, self.mid2 = (ResBlock3D(ch[2], ch[3], emb),
                                           Attn3D(ch[3]),
                                           ResBlock3D(ch[3], ch[3], emb))

        # after upsampling we concatenate the matching skip, so the input width is up+skip
        self.u2 = ResStack(ch[3] + ch[2], ch[2], emb, blocks)
        self.u1 = ResStack(ch[2] + ch[1], ch[1], emb, blocks)
        self.u0 = ResStack(ch[1] + ch[0], ch[0], emb, blocks)

        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_conv = nn.Conv3d(ch[0], out_ch, 1)

    @staticmethod
    def up(x):
        """Nearest-neighbour upsample by 2 (avoids the checkerboard artefacts of transposed conv)."""
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x, t):
        """x: (B,9,64,64,64) noisy FLAIR + conditioning.  t: (B,) timesteps.  -> (B,1,64,64,64)."""
        temb = self.temb(timestep_embedding(t, self.temb_dim))     # (B, 512)

        h  = self.stem(x)                       # (B, 32, 64,64,64)
        s0 = self.d0(h, temb)                   # (B, 32, 64,64,64)  <- skip 0
        s1 = self.d1(self.down0(s0), temb)      # (B, 64, 32,32,32)  <- skip 1
        s2 = self.d2(self.down1(s1), temb)      # (B,128, 16,16,16)  <- skip 2

        m = self.mid2(self.attn(self.mid1(self.down2(s2), temb)), temb)   # (B,256, 8,8,8)

        d = self.u2(torch.cat([self.up(m), s2], 1), temb)   # (B,128, 16,16,16)
        d = self.u1(torch.cat([self.up(d), s1], 1), temb)   # (B, 64, 32,32,32)
        d = self.u0(torch.cat([self.up(d), s0], 1), temb)   # (B, 32, 64,64,64)
        return self.out_conv(F.silu(self.out_norm(d)))      # (B,  1, 64,64,64)

def build_model_monai():
    """MONAI's DiffusionModelUNet (spatial_dims=3) — a PROVEN 3D diffusion backbone.

    Why this exists: every other piece of the 3D pipeline (schedule, epsilon-MSE objective, DDIM
    sampler, conditioning layout, EMA) is shared with the 2D pipeline that demonstrably works.
    The ONLY unvalidated component is the hand-written UNet3D above. Swapping in a library
    implementation that many medical-imaging papers rely on tests that component directly instead
    of debugging it. Same interface: forward(x, timesteps) -> predicted noise.

    Needs: pip install monai
    """
    from monai.networks.nets import DiffusionModelUNet      # imported lazily so monai stays optional
    w  = int(os.environ.get("WIDTH_3D", "64"))
    ch = (w, w * 2, w * 4, w * 8)                           # 64 -> 32 -> 16 -> 8, same ladder as ours
    kw = dict(spatial_dims=3, in_channels=9, out_channels=1,
              attention_levels=(False, False, False, True), # attention only at 8^3, same constraint
              num_res_blocks=int(os.environ.get("BLOCKS_3D", "2")),
              num_head_channels=w)
    try:                                                    # arg was renamed between MONAI versions
        return DiffusionModelUNet(num_channels=ch, **kw)
    except TypeError:
        return DiffusionModelUNet(channels=ch, **kw)

def build_3d(arch=None):
    """Pick the architecture: ARCH_3D=custom (our UNet3D) or ARCH_3D=monai."""
    a = (arch or os.environ.get("ARCH_3D", "custom")).lower()
    return build_model_monai() if a == "monai" else build_model_3d()

def load_state_compat(model, path, device):
    """Load a checkpoint, tolerating ones saved BEFORE the ResStack wrapper existed.

    Wrapping the per-level blocks in ResStack inserted a level into every parameter path:
        old  d0.norm1.weight        (ResBlock3D directly on the attribute)
        new  d0.blocks.0.norm1.weight
    The tensors are identical, only the names differ, so for a 1-block model we can just
    reinsert ".blocks.0". The mid blocks are still plain ResBlock3D, so they never changed.
    """
    sd = torch.load(path, map_location=device)
    try:
        model.load_state_dict(sd)
        return "direct"
    except RuntimeError:
        stacks = ("d0.", "d1.", "d2.", "u0.", "u1.", "u2.")
        remap = {}
        for k, v in sd.items():
            if k.startswith(stacks) and k.split(".")[1] != "blocks":
                head, rest = k.split(".", 1)
                k = f"{head}.blocks.0.{rest}"
            remap[k] = v
        model.load_state_dict(remap)
        return "remapped (pre-ResStack checkpoint)"

def build_model_3d(width=None):
    """Conditional 3D denoiser: 9 in (noisy FLAIR + T1 + mask + 6 atlas) -> 1 out (noise).

    `width` is the base channel count; the four levels are (w, 2w, 4w, 8w).
      w=32 ->  11M params (the first feasibility build, too small for good detail)
      w=64 ->  45M params (recommended: uses the RTX 6000's headroom)
      w=96 -> 100M params (comparable to the 2D model's 85M)
    Override without editing code:  WIDTH_3D=64 python src/model3d.py
    Any w must be divisible by 8, because every GroupNorm uses 8 groups.
    """
    w = int(width if width is not None else os.environ.get("WIDTH_3D", "64"))
    b = int(os.environ.get("BLOCKS_3D", "2"))     # ResBlocks per level; 2 matches the 2D model
    assert w % 8 == 0, "width must be divisible by 8 (GroupNorm(8, C))"
    return UNet3D(in_ch=9, out_ch=1, ch=(w, w * 2, w * 4, w * 8), blocks=b)

if __name__ == "__main__":
    # shape + memory self-test: run `python src/model3d.py` before launching any training
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_3d().to(dev)
    w = int(os.environ.get("WIDTH_3D", "64"))
    b = int(os.environ.get("BLOCKS_3D", "2"))
    print(f"arch {os.environ.get('ARCH_3D','custom')} | device {dev} | width {w} -> channels {(w, w*2, w*4, w*8)} | {b} blocks/level | "
          f"params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    for B in (1, 2, 4, 8):
        try:
            x = torch.randn(B, 9, 64, 64, 64, device=dev)
            t = torch.randint(0, 1000, (B,), device=dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                y = m(x, t)
                loss = y.pow(2).mean()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
            print(f"batch {B}: in {tuple(x.shape)} -> out {tuple(y.shape)}  peak {peak:.1f} GB  FITS")
            m.zero_grad(set_to_none=True)
            del x, t, y, loss
            if dev == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"batch {B}: OUT OF MEMORY"); break
            raise
