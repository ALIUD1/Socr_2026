# Paper Materials — Conditional Diffusion for Synthetic Brain MRI

*A single reference document containing everything needed to write the paper: motivation,
methods, hyperparameters, data, results, figures, and open questions. Numbers are pulled
from the actual training logs and evaluation runs of this project.*

---

## 0. Working titles

- *Anatomy-Conditioned Diffusion for Synthetic Brain FLAIR MRI with Preserved Pathology*
- *Breaking the Informational Floor: Patient-Anatomy Conditioning in Diffusion-Based MRI Synthesis*
- *Toward Multimodal (Image + Text) Conditional Generation of Brain Tumor MRI*

---

## 1. Abstract (draft)

We present a conditional 2-D denoising diffusion model that synthesizes brain FLAIR MRI
slices conditioned on a patient's own anatomy and pathology. Each generation is conditioned
on (i) the patient's T1-weighted scan, (ii) a tumor segmentation mask, and (iii) a
registered probabilistic lobe atlas, concatenated as input channels to a U-Net denoiser
that predicts noise (ε-prediction). Trained on ~1,250 patients from BraTS 2023, the model
produces anatomically faithful, patient-matched FLAIR slices that preserve tumor structure
without pixel-copying the source. Our central empirical finding is that **the informativeness
of the conditioning sets a floor on the achievable denoising loss**: conditioning on tumor
mask + atlas alone plateaus at MSE ≈ 0.006 regardless of model size or training length,
whereas adding the patient T1 lowers the floor to ≈ 0.0042 (~30% reduction) and produces
sharp, clean-background samples. We report PSNR, SSIM, and background-SNR over held-out
patients, discuss the tension between full-reference fidelity metrics and generative
diversity, and outline an extension to multimodal (text-prompt) conditioning via
cross-attention and classifier-free guidance.

---

## 2. Contributions

1. A fully working conditional 2-D diffusion pipeline for brain FLAIR synthesis on BraTS,
   from atlas registration → preprocessing → training → sampling.
2. **The "informational floor" observation**: identifying that the conditioning's ability
   to *determine* the target upper-bounds how low the ε-prediction MSE can go, and showing
   that stronger conditioning (patient T1) demonstrably lowers that floor.
3. A reusable, atlas-derived **automatic captioning** scheme that converts segmentation +
   probabilistic atlas into per-slice text — enabling text conditioning on a dataset with
   no radiology reports.
4. An evaluation discussion on why full-reference metrics (PSNR/SSIM) under-credit a
   *generative* (non-copying) model, motivating distribution- and utility-based metrics.

---

## 3. Background / related work (to expand + cite)

- **Denoising diffusion probabilistic models (DDPM)** — Ho et al. 2020; ε-prediction objective.
- **DDIM** — Song et al. 2021 — deterministic, few-step sampling.
- **Classifier-free guidance** — Ho & Salimans 2022 — condition dropout enabling optional
  modalities and guidance-strength control.
- **Latent/So text-to-image diffusion (cross-attention conditioning)** — Rombach et al. 2022
  (Stable Diffusion) — the `UNet2DConditionModel` architecture we adopt for text.
- **Medical image synthesis / cross-modality translation** — GAN- and diffusion-based MRI
  synthesis; note these are often *deterministic* translation (high SSIM) vs. our stochastic
  generation.
- **BraTS challenge** — Menze et al. 2015; Baid et al. 2021 (BraTS 2021/2023 GLI).
- **Atlases / spatial priors** — ICBM452, SRI24 (Rohlfing et al. 2010).
- **Lab context** — the group's existing U-Net regressor (deterministic, tends to blur) is
  the comparison baseline; a separate 3-D wavelet diffusion project (wdm-3d, x0-prediction)
  exists. Our model is 2-D, ε-prediction, and adds anatomy+atlas conditioning.

---

## 4. Data

### 4.1 BraTS 2023 (adult glioma, GLI)
- Volumes: **240 × 240 × 155**, **1 mm isotropic**, already **skull-stripped** (background
  is *exactly* 0) and co-registered to **SRI24** space.
- Modalities (suffixes): `t1n` (T1), `t1c` (T1ce), `t2w` (T2), **`t2f` (FLAIR, our target)**,
  `seg` (segmentation).
- Segmentation labels: **1 = necrotic core (NCR)**, **2 = edema (ED)**, **3 = enhancing
  tumor (ET)**; tumor is < 1% of brain volume.
- **~1,252 patients** used (train/val/test split by patient, 70/15/15, to avoid leakage from
  near-duplicate adjacent slices).
- Non-standard on-disk layout handled: some modalities are stored as a *folder* named
  `<patient>-<suffix>.nii/` with the image inside; others are flat files.

### 4.2 Atlas: ICBM452 probabilistic lobular
- 6 lobe probability maps: frontal, parietal, temporal, occipital, cerebellum, insula.
- Native format Analyze (`.hdr/.img`), ~149 × 188 × 148, different space/size from BraTS.

### 4.3 Processed dataset
- **163,974 slices** total after filtering (slices with ≥ 1,000 brain voxels kept).
- **79,001 tumor slices / 84,973 tumor-free slices** (tumor slices include edema, which
  spreads well beyond the core).
- Stored on lab storage (Turbo) as per-slice `.npy` of shape **(9, 256, 256)**, `float16`.

---

## 5. Methods

### 5.1 Atlas registration (once, reused for all patients)
Because BraTS patients are already mutually co-registered in SRI24, the atlas is registered
**once** into SRI24 and reused (not per patient). Pipeline (`02_register_atlas.py`):
1. Convert the ICBM Analyze files → NIfTI with **nibabel** (`as_closest_canonical`) *before*
   ANTs. **Key bug solved:** ANTs/ITK misread the Analyze orientation (anterior–posterior
   flip) while nibabel read it correctly; converting first fixes it.
2. Register ICBM452 T1 → SRI24 template (`spgr.nii.gz`, skull-stripped T1) with **ANTs SyN**.
3. Apply the transform to all 6 lobe probability maps (linear interpolation).
- **Validation:** overlaying each warped lobe on real patient FLAIR shows correct anatomy
  (frontal = anterior, occipital = posterior, temporal = lateral). See Fig. 2.

### 5.2 Preprocessing (`03_preprocess.py`)
- **Intensity normalization:** percentile [1, 99] over brain voxels → clip to [0, 1];
  background forced to 0.
- **Padding:** zero-pad 240 → **256** (a power of two for the U-Net's repeated halving);
  *padding, not resizing*, so anatomy is never resampled/distorted.
- **Slice stack (9 channels):** `[FLAIR, T1, mask, 6 atlas lobes]`; saved as `float16`.

### 5.3 Conditioning design (the key lever)
- **Target:** channel 0 = FLAIR.
- **Conditioning (8 channels):** channel 1 = **patient T1**, channel 2 = tumor mask,
  channels 3–8 = 6 atlas lobe maps.
- Rationale: mask + atlas specify *where* and *what kind* of pathology, but **do not
  determine the specific patient's brain** → irreducible prediction error (see §6.1).
  Adding the co-registered **patient T1** supplies real per-patient anatomy.
- Conditioning enters the network by **channel concatenation** with the noisy FLAIR:
  `x_in = cat([noisy_flair, cond], dim=1)` → first conv sees all 9 channels at every pixel.

### 5.4 Diffusion setup (DDPM, ε-prediction)
- Schedule: `betas = linspace(1e-4, 0.02, T=1000)`; `alpha_bar = cumprod(1 − betas)`.
- Forward: `noisy = sqrt(ā)·x + sqrt(1−ā)·ε`, `ε ~ N(0, I)`.
- Target FLAIR scaled **[0,1] → [−1,1]** (`·2 − 1`); conditioning channels **not** rescaled.
- Objective: **MSE(predicted noise, true noise)** — a consistent unit-variance target,
  more stable than predicting the image.

### 5.5 Model
- `diffusers.UNet2DModel`, `sample_size=256`, `in_channels=9`, `out_channels=1`,
  `layers_per_block=2`, `block_out_channels=(128, 256, 384, 512)`, **attention at the
  deepest level** (`AttnDownBlock2D` / `AttnUpBlock2D`).
- Defined once in `src/model.py` and imported by every script (single source of truth →
  no checkpoint/shape mismatches).

### 5.6 Training
- Optimizer **AdamW**, lr **1e-4**, batch **32** (batch 64 OOMs even at 96 GB with this model).
- **Mixed precision** (`torch.amp` autocast + `GradScaler`) and **EMA** (decay 0.999); the
  EMA weights are what we sample from. Checkpoints saved every epoch.

### 5.7 Sampling
- **DDIM** (`DDIMScheduler`, `clip_sample=True`), **200 steps**, deterministic.
- Chosen after plain 1000-step DDPM sampling proved unstable (same conditioning produced
  wildly varying brightness — see §6.2).

### 5.8 Text conditioning (in progress)
- **Captions (`04_captions.py`):** BraTS has no reports, so a per-slice caption is *derived*
  from the mask + atlas: presence/among {enhancing, necrotic, edema}, the lobe of maximum
  atlas overlap under the tumor, and laterality (tumor centroid vs. midline). Example:
  *"axial FLAIR slice with enhancing tumor, necrotic core and surrounding edema in the left
  occipital lobe."* 163,974 captions generated.
  - **Orientation note (for reviewers):** BraTS/SRI24 is stored **LPS**; laterality was
    verified via the NIfTI affine (`aff2axcodes → ('L','P','S')`).
- **Planned model change:** swap `UNet2DModel → UNet2DConditionModel` (adds cross-attention),
  encode captions with a (frozen) text encoder, feed embeddings via cross-attention while
  keeping the spatial channels.
- **Planned training change:** **condition dropout** (randomly blank text and/or image
  conditioning) → **classifier-free guidance**, enabling image-only / text-only / both and a
  guidance-strength knob.

---

## 6. Results

### 6.1 The informational floor (central finding)
Every **8-channel** (mask + atlas only) configuration — small model, large model, 150-patient
subset, full data — plateaued at training MSE **≈ 0.006**, indicating an *informational*
limit rather than a capacity or optimization limit. Adding the **patient T1 (9-channel)**
lowered the plateau to **≈ 0.0042** (~30% reduction). See **Fig. 4** (`training_loss_curve.svg`).

Representative 9-channel loss trajectory (run `52090423`, per-epoch MSE): 0.0266 (ep 0) →
0.0072 (ep 10) → 0.0055 (ep 40) → 0.0044 (ep 60) → **0.0042 (ep 79)**. Full series in §9.

### 6.2 Sample stability (DDPM → DDIM)
A 6-sample grid on one conditioning under plain DDPM showed brain-mean intensities of
0.17, 0.17, 0.29, 0.59, 0.63, 0.72 (vs. real 0.58) — high-variance sampling instability.
Switching to deterministic DDIM tightened this dramatically: on a held-out slice (real mean
**0.413**) the six samples read **0.477, 0.584, 0.426, 0.474, 0.415, 0.491**.

### 6.3 Background cleanliness
Grainy backgrounds in early/undertrained models resolved with full training: the fully
trained model renders **true black** backgrounds (real background is exactly 0). Quantified
by background-SNR below.

### 6.4 Quantitative metrics (100 held-out val slices, 1 sample each, 200 DDIM steps)
| Metric | Value | Units | Meaning |
|---|---|---|---|
| **PSNR** | **17.09 ± 2.84** | dB (log scale) | pixel closeness to the real slice (over brain) |
| **SSIM** | **0.402 ± 0.127** | unitless [0,1] | structural similarity to the real slice |
| **background-SNR** | **66.6 ± 35.5** | unitless ratio | mean brain signal ÷ background noise std |

Per-sample metrics are also stamped on the qualitative grid (Fig. 3): e.g. PSNR up to 20.0 dB,
SSIM up to 0.55, SNR up to 74 on individual samples.

### 6.5 Qualitative
Generated FLAIR slices reproduce patient-specific anatomy (ventricles, gyri, gray/white
contrast) and tumor location, with per-sample texture diversity (not pixel copies). See
Figs. 3 and 5.

---

## 7. Discussion

- **Why the floor exists:** ε-prediction MSE is lower-bounded by the conditional variance of
  the target given the conditioning. Weak conditioning ⇒ high residual variance ⇒ high floor.
  This reframes "the loss won't go below X" from a training failure to an *information*
  statement, and predicts (correctly) that adding informative conditioning lowers it.
- **Metrics vs. the generative goal:** PSNR/SSIM measure fidelity to *one* reference and thus
  **reward copying and penalize diversity** — the opposite of the project's aim ("preserve
  pathology *without* pixel-copying"). Moderate PSNR (17 dB) / SSIM (0.40) are therefore
  partly *by design*. The proper metrics are distribution-level (FID) and downstream utility
  (does synthetic data improve a segmentation model?).
- **A degenerate-background caveat:** whole-image SSIM is *not* inflated by the matching black
  background here — the real background has zero variance, which actually depresses SSIM's
  contrast term. A brain-masked SSIM is the honest tissue-structure number (planned).

---

## 8. Limitations & future work

- **Limitations:** 2-D slices only (no 3-D continuity); full-reference metrics under-credit
  generation; captions are templated (limited vocabulary); laterality convention required
  manual verification; tumor-fidelity not yet quantified.
- **Future work:** (1) text conditioning + classifier-free guidance (in progress); (2)
  tumor/pathology-fidelity metric (does generated FLAIR show abnormal signal where the mask
  says?); (3) FID and downstream-utility evaluation; (4) brain-masked SSIM; (5) extension to
  3-D.

---

## 9. Figures (files in the repo root)

| # | File | Caption |
|---|---|---|
| 1 | `one_example_channels.png` | **Conditioning inputs.** Target FLAIR + tumor mask + 6 registered atlas lobe maps for one slice (T1 channel added later; shown here is the spatial-prior stack). |
| 2 | `atlas_on_patient.png` | **Atlas registration validation.** Frontal-lobe probability map (warped ICBM452→SRI24) overlaid on a real patient FLAIR — correct anterior alignment. |
| 3 | `eval_grid_20260626_133819.png` | **Main qualitative + quantitative result.** Real FLAIR vs. 6 generated samples on the same conditioning, each annotated with PSNR / SSIM / background-SNR. |
| 4 | `training_loss_curve.svg` | **The informational floor.** Per-epoch ε-prediction MSE for the 9-channel (+T1) model; dashed line marks the ~0.006 plateau of every mask+atlas-only (8-channel) run. |
| 5 | `eval_grid_20260625_145850.png` | **Full-dataset samples.** Real vs. 6 generations — clean black backgrounds, patient-matched anatomy. |
| 6 | `noise_schedule.png` | **Forward diffusion.** Progressive noising of a tumor FLAIR slice at signal levels ā = 1.0, 0.9, 0.6, 0.2. |

Supplementary figures also available in the repo root: `brats_overlay.png`,
`warpedT1_vs_sri24.png`, `normalization_hist.png`, `forward_noise.png`,
`generated_sample_v3_ema.png`.

---

## 10. Reproducibility appendix

### 10.1 Hyperparameters
| Item | Value |
|---|---|
| Image size | 256 × 256 (240 zero-padded), 1 mm iso, axial 2-D |
| Input / output channels | 9 in (FLAIR + T1 + mask + 6 atlas) / 1 out (noise) |
| Diffusion steps T | 1000 |
| β schedule | linear 1e-4 → 0.02 |
| Target scaling | FLAIR [0,1] → [−1,1]; conditioning unscaled |
| Model | `UNet2DModel`, blocks (128, 256, 384, 512), attn at deepest, 2 layers/block |
| Optimizer / lr | AdamW / 1e-4 |
| Batch size | 32 (mixed precision) |
| EMA decay | 0.999 |
| Sampler | DDIM, `clip_sample=True`, 200 steps |
| Split | 70 / 15 / 15 by patient |

### 10.2 Full 9-channel loss series (run 52090423, per-epoch MSE)
```
0.0266 0.0100 0.0081 0.0074 0.0070 0.0074 0.0069 0.0064 0.0062 0.0062
0.0061 0.0059 0.0061 0.0057 0.0060 0.0057 0.0061 0.0056 0.0056 0.0054
0.0061 0.0054 0.0054 0.0055 0.0055 0.0053 0.0053 0.0054 0.0051 0.0054
0.0050 0.0055 0.0053 0.0049 0.0051 0.0051 0.0051 0.0050 0.0050 0.0051
0.0048 0.0048 0.0048 0.0048 0.0048 0.0050 0.0047 0.0047 0.0047 0.0046
0.0048 0.0047 0.0049 0.0046 0.0046 0.0044 0.0044 0.0046 0.0046 0.0046
0.0045 0.0046 0.0045 0.0043 0.0046 0.0045 0.0045 0.0042 0.0044 0.0045
0.0043 0.0042 0.0047 0.0042 0.0042 0.0043 0.0043 0.0042 0.0044 0.0042
```

### 10.3 Hardware / environment
- U-Michigan Great Lakes HPC; **NVIDIA RTX PRO 6000 Blackwell, 96 GB** (compute sm_120),
  requiring **PyTorch cu128** (torch 2.6/cu124 lacked sm_120 kernels).
- Python venv (`brainmri`); batch 32 uses ~89 GB.

### 10.4 Code map
| File | Role |
|---|---|
| `scripts/02_register_atlas.py` | ICBM452 → SRI24 registration (ANTs SyN) |
| `scripts/03_preprocess.py` | build 9-channel `.npy` slices |
| `scripts/04_captions.py` | derive per-slice text captions (seg + atlas) |
| `src/model.py` | `build_model()` — the U-Net denoiser (single source of truth) |
| `src/dataset.py` | `SliceDataset` (optional caption return) |
| `scripts/train.py` (+`train.sbatch`) | training (mixed precision + EMA) |
| `scripts/sample.py`, `eval_grid.py`, `eval_metrics.py` | sampling + evaluation |
| `scripts/plot_loss.py` | training-loss figure |

---

*Generated as a working reference; verify all numbers against the current logs before
submission, and expand §3 with full citations.*
