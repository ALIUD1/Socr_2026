"""model.py — the diffusion denoiser architectures, defined once and shared by all scripts.

Two builders live here on purpose:

  build_model()      -> the SPATIAL-ONLY model (UNet2DModel). This is the model in the
                        paper (braingen_CondDiffuser_BraTS_v1). DO NOT change it, or the
                        deployed checkpoint models/diffusion_ema.pt will no longer load.

  build_text_model() -> the NEW text-conditioned model (UNet2DConditionModel). Same 9 input
                        channels, but its attention blocks can also read text embeddings.
"""
from diffusers import UNet2DModel, UNet2DConditionModel

# CLIP ViT-B/32's text encoder outputs 512-dim vectors. The U-Net's cross-attention
# layers must be built to expect exactly this width, so the two numbers must match.
TEXT_EMBED_DIM = 512

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

def build_text_model():
    """Text-conditioned denoiser: same 9 spatial input channels PLUS cross-attention to text.

    Spatial conditioning (T1 + mask + atlas) still enters through the input channels exactly
    as before. Text enters through the CrossAttn* blocks, which attend to the caption
    embeddings passed as `encoder_hidden_states` at forward time.

    Why cross-attention only at the two DEEPEST levels: a CrossAttn block runs BOTH
    self-attention (every pixel attends to every other pixel -> cost grows with the SQUARE of
    the pixel count) and cross-attention (pixels attend to ~77 text tokens -> cheap/linear).
    The self-attention is the killer: at 256x256 that's 65,536 tokens attending to each other,
    which will not fit in any GPU. So attention lives at the 64x64 and 32x32 levels only --
    the same place the spatial-only model put its attention.
    """
    return UNet2DConditionModel(
        sample_size=256,
        in_channels=9,                              # noisy FLAIR + T1 + mask + 6 atlas (unchanged)
        out_channels=1,                             # predicted noise
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),    # resolutions: 256 -> 128 -> 64 -> 32
        down_block_types=(
            "DownBlock2D",           # @256 - no attention (too big)
            "DownBlock2D",           # @128 - no attention (too big)
            "CrossAttnDownBlock2D",  # @64  - reads text
            "CrossAttnDownBlock2D",  # @32  - reads text
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",    # @32  - reads text
            "CrossAttnUpBlock2D",    # @64  - reads text
            "UpBlock2D",             # @128
            "UpBlock2D",             # @256
        ),
        cross_attention_dim=TEXT_EMBED_DIM,         # must equal the text encoder's output width
    )
