#!/bin/bash
#SBATCH --job-name=brainmri-prep
#SBATCH --account=dinov99
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/preprocess_%j.log

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python scripts/03_preprocess.py