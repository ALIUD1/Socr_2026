#!/bin/bash
#SBATCH --job-name=3d-zerosnr
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/train_3d_zerosnr_%j.log

# EXPERIMENT C — zero terminal SNR.
# Three configs (custom 52M, MONAI 165M, +21% data) ALL floored at a minimum of ~0.22 and
# could not produce black. A failure that reproducible across architecture and data size is
# not the model -- it is something they share. The standard linear schedule leaves
# sqrt(alpha_bar_T) ~= 0.0066, i.e. 0.66% of the image still present at the noisiest step.
# Across 262k redundant voxels that hint is recoverable, so the model leans on it for global
# brightness; at sampling we start from pure noise, the hint is gone, and it reverts to the
# mean. ZERO_SNR=1 rescales so alpha_bar_T is exactly 0, removing the train/sample mismatch.
# NOTE: ZERO_SNR must also be set when sampling this checkpoint.
export ARCH_3D=custom
export SPLIT_3D=train
export ZERO_SNR=1
export WIDTH_3D=64
export BLOCKS_3D=2
export BATCH_3D=12
export EPOCHS_3D=800
export LR_3D=1e-4
export TAG_3D=diffusion3d_zerosnr

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/train_3d.py
