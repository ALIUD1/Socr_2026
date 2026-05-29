# Synthetic Brain MRI Generation

Conditional generative pipeline that takes a real patient scan + tumor mask +
anatomical atlas prior and produces a **synthetic** FLAIR brain MRI that
preserves anatomy and pathology without being a pixel copy of the source patient.

- **Patient data:** BraTS 2023 (adult glioma). FLAIR = the `t2f` modality.
- **Anatomical prior:** ICBM452 probabilistic lobular atlas (6 lobes).
- **Plan:** 2D first (easier to debug, less GPU) → 3D later.
- **Model:** conditional U-Net regression baseline first, then conditional diffusion.

## Key data facts (verified)

| Thing | Value |
|-------|-------|
| BraTS image space | SRI24, 240×240×155, 1 mm isotropic, skull-stripped, co-registered |
| BraTS modalities | `t1n` (T1), `t1c` (T1ce), `t2w` (T2), **`t2f` (FLAIR)**, `seg` (mask) |
| BraTS labels | 1 = necrotic core (NCR), 2 = edema (ED), 3 = enhancing tumor (ET) |
| Atlas | ICBM452, Analyze `.hdr/.img`, 149×188×148, 1 mm, values 0–32640 (scale to 0–1) |

Because BraTS patients are **already** skull-stripped and co-registered to SRI24,
we do NOT register patients one by one. We register the **atlas → SRI24 once** and
reuse that transform for every patient.

## Folder layout

```
Summer2026/
├── data/
│   ├── raw/brats/        # BraTS 2023 (lives on Great Lakes; gitignored)
│   ├── raw/icbm/         # ICBM452 atlas (gitignored)
│   └── processed/        # aligned + normalized data ready for training (gitignored)
├── scripts/              # runnable steps, numbered in order
├── src/                  # reusable modules (datasets, models) — added later
├── configs/              # experiment settings — added later
├── models/               # saved checkpoints (gitignored)
├── outputs/              # figures, generated samples (gitignored)
└── environment.yml       # conda environment definition
```

Code is tracked in git and pushed to GitHub. Data and model files are **not**
committed (they are large and/or licensed) — they live on Great Lakes only.

## Environment setup (on Great Lakes)

```bash
module load python                 # or load Miniconda, per the cluster docs
conda env create -f environment.yml
conda activate brainmri

# PyTorch is installed separately to match the cluster's CUDA version.
# Check the CUDA version first:  module avail cuda   (or  nvidia-smi  on a GPU node)
# Then install the matching wheel, e.g. for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Workflow (run order)

1. `scripts/01_inspect_brats.py` — confirm one patient loads, shapes match, labels are 1/2/3.
2. *(next)* inspect the ICBM atlas the same way.
3. *(next)* register atlas → SRI24 (one time, with ANTs).
4. *(next)* preprocessing: normalize, crop, save aligned FLAIR/mask/atlas.
5. *(next)* extract 2D slices, build a patient-level train/val/test split.
6. *(next)* train the U-Net baseline, then the diffusion model.

## Running on Great Lakes (SLURM crash course)

Great Lakes is a shared cluster. You never run heavy jobs on the login node;
you ask the **SLURM** scheduler for a node. Two ways:

**Interactive** (for debugging the inspection script):
```bash
salloc --account=YOUR_ALLOCATION --partition=gpu --gres=gpu:1 \
       --cpus-per-task=4 --mem=16G --time=1:00:00
# you land on a compute node; now run:
python scripts/01_inspect_brats.py data/raw/brats/BraTS-GLI-00000-000
```

**Batch** (for real training — submit and walk away): see `scripts/*.sbatch`
job files (added when we start training).
