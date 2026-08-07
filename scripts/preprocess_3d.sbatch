#!/bin/bash
#SBATCH --job-name=brainmri-prep3d
#SBATCH --account=dinov99
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/preprocess3d_%j.log

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/05_preprocess_3d.py
