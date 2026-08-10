#!/bin/bash
#SBATCH --job-name=brainmri-3d
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=outputs/train_3d_%j.log

# sized from the src/model3d.py self-test: width 96 = 95M params,
# batch 16 ~= 45 GB of the 96 GB card. ~875 volumes -> 55 steps/epoch.
export WIDTH_3D=96
export BATCH_3D=16
export EPOCHS_3D=800

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/train_3d.py
