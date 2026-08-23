#!/bin/bash
#SBATCH --job-name=3d-monai
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/train_3d_monai_%j.log

# EXPERIMENT A — same data, PROVEN architecture.
# Every other part of the 3D pipeline is shared with the 2D pipeline that works; the only
# unvalidated component is our hand-written UNet3D. This swaps in MONAI's DiffusionModelUNet
# and changes NOTHING else, so the result attributes cleanly to the architecture.
export ARCH_3D=monai
export SPLIT_3D=train
export WIDTH_3D=64
export BLOCKS_3D=2
export BATCH_3D=8
export EPOCHS_3D=800
export LR_3D=1e-4
export TAG_3D=diffusion3d_monai

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/train_3d.py
