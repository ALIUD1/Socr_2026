#!/bin/bash
#SBATCH --job-name=text-rich
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/train_text_rich_%j.log

# Text conditioning v2: RICHER CAPTIONS + the mask-dropout recipe that already worked.
#
# v1 captions carried only lobe + side, so laterality was the only thing text could ever
# control (proven: +0.025, z=9.6, 100% of held-out slices). v2 adds size, slice level and the
# dominant component, giving the prompt three more attributes to steer.
#
# Mask-only dropout (MASK_DROP=0.5) is kept because it is what made text load-bearing: with the
# mask gone, the caption is the ONLY thing that can place pathology, so the model is forced to
# use it. Dual dropout is deliberately NOT used -- it blanked the anatomy too, creating
# impossible whole-brain-from-text batches that destroyed the laterality control.
export CAPTIONS_CSV=captions_v2.csv
export TAG_TEXT=text_rich

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/train_text_place.py
