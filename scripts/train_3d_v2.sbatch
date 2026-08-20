#!/bin/bash
#SBATCH --job-name=brainmri-3dv2
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/train_3d_v2_%j.log

# Run 2 of the 3D model. Run 1 (width 96, ONE ResBlock per level, no augmentation) learned
# coarse anatomy but stalled on contrast and never produced the tumour. Two changes:
#   DEPTH:  BLOCKS_3D=2 -> ~16 ResBlocks instead of 8, matching the 2D model that works.
#           Traded width 96 -> 64 to keep memory in budget; depth matters more than width
#           for fine detail. Params land ~80M, still comparable to the 2D model's 85M.
#   AUG:    AUG_3D=1 -> left-right mirror + small shifts. With only ~875 volumes this is the
#           difference between seeing 875 examples and seeing many poses of them.
# This is a FRESH run: the architecture changed, so the old checkpoint cannot be resumed.
export WIDTH_3D=64
export BLOCKS_3D=2
export AUG_3D=1
export BATCH_3D=12
export EPOCHS_3D=800
export LR_3D=1e-4

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/train_3d.py
