#!/bin/bash
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/train_%j.log

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python scripts/train.py