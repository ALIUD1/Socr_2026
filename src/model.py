"""model.py — the diffusion denoiser architecture, defined once and shared by all scripts."""
from diffusers import UNet2DModel

def build_model():
    """Conditional denoiser U-Net: 9 input channels (noisy FLAIR + 8 conditioning: T1 + mask + 6 atlas) -> 1 (noise)."""
    return UNet2DModel(
        sample_size=256,
        in_channels=9,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),   # wider than before (was 64,128,256,256) -> ~4x capacity
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )