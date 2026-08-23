#!/bin/bash
#SBATCH --job-name=3d-moredata
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/train_3d_moredata_%j.log

# EXPERIMENT B — same architecture, MORE data.
# Folds the validation volumes into training (875 -> ~1062, +21%). TEST stays untouched as a
# clean hold-out, so final evaluation is still honest; sample_3d.py reads from val, which the
# model has now seen, so judge this run on TEST volumes rather than val.
export ARCH_3D=custom
export SPLIT_3D=train+val
export WIDTH_3D=64
export BLOCKS_3D=2
export BATCH_3D=12
export EPOCHS_3D=800
export LR_3D=1e-4
export TAG_3D=diffusion3d_moredata

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/train_3d.py
