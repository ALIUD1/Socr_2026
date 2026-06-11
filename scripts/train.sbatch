#!/bin/bash
#SBATCH --job-name=brainmri-train
#SBATCH --account=dinov99
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/train_%j.log

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python scripts/train.py