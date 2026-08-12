#!/bin/bash
#SBATCH --job-name=brainmri-3dres
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/train_3d_resume_%j.log

# Continue the plateaued 800-epoch run at a 5x LOWER learning rate.
# The first run flattened at ~0.005 and fluctuated, which says the LR was too
# large to settle; decaying it is the standard fix and usually sharpens fine
# detail (which is exactly what is missing: tumour contrast).
# Resume from _last (raw training weights), NOT _ema — EMA rebuilds from these.
export WIDTH_3D=96
export BATCH_3D=16
export EPOCHS_3D=800
export LR_3D=2e-5
export RESUME_3D=models/diffusion3d_last.pt

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
# keep a copy of the current model so a bad resume cannot destroy it
cp models/diffusion3d_ema.pt  models/diffusion3d_ema_run1.pt
cp models/diffusion3d_last.pt models/diffusion3d_last_run1.pt
python -u scripts/train_3d.py
