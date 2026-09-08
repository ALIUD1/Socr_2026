#!/bin/bash
#SBATCH --job-name=conddiff-gallery
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/render_gallery_%j.log

# =============================================================================================
# PRE-RENDER THE WEBSITE'S GALLERY.
#
# WHAT THIS JOB PRODUCES
#   conddiff_gallery/manifest.json  +  conddiff_gallery/<cell_id>/vNN_flair.png (and _seg.png)
#   -- one image per (Tumour x Lobe x Slice Location x Tumour Size) combination the site
#   exposes, K seed variants each, rendered at the paper's full 200 DDIM steps.
#
# WHY IT EXISTS AT ALL
#   The deployed backend runs on a Lightsail "micro" instance: 0.25 vCPU, 1 GB RAM, and a
#   health check that replaces the container after ~65 s of not answering. One DDIM sample from
#   this 85.3M-parameter U-Net is 50-200 forward passes of ~950 GFLOP each, and the checkpoint
#   alone is ~341 MB. Live CPU diffusion there is not slow, it is impossible -- the container
#   is killed mid-request, every time. But the user-visible parameter space is FINITE (57 real
#   cells, because Lobe and Size are meaningless without a tumour), so we render it here, once,
#   on a GPU, and the server just serves PNGs. See scripts/render_gallery.py's docstring.
#
# WHY gpu-rtx6000 SPECIFICALLY, AND NOT ANY OTHER GPU PARTITION
#   The environment's torch is the cu128 build, whose bundled kernels cover compute capability
#   7.5 through 12.0. The RTX 6000 (Turing, CC 7.5) is inside that range. The V100 (Volta,
#   CC 7.0) is NOT: every CUDA kernel launch on it fails with
#       RuntimeError: CUDA error: no kernel image is available for execution on the device
#   which surfaces only once the first forward pass runs -- i.e. after the queue wait, the
#   checkpoint load and the calibration, several minutes into a job that was never going to
#   work. Do not "helpfully" widen this to --partition=gpu.
#
# WHY --time=06:00:00
#   57 cells x 4 variants x 200 steps = 45,600 U-Net forward passes. At the ~0.18 s/forward this
#   model achieves on an RTX 6000 that is about 2h20m, plus a few minutes for the checkpoint
#   load and the per-image mask synthesis. Six hours is roughly 2.5x headroom, which covers a
#   busy/shared GPU without parking a large reservation in the owned partition.
#   Run `python scripts/render_gallery.py --dry-run` on a LOGIN NODE first: it prints the exact
#   forward-pass count and time estimate for whatever K and step count you set below, without
#   importing torch or needing a GPU. Size this --time from that number.
#
# WHY ONLY 4 CPUS AND 32G
#   Unlike the training jobs, there is no DataLoader here and no multi-worker augmentation
#   pipeline: the loop loads one 1 MiB .npy at a time and everything expensive happens on the
#   GPU. The peak host-side allocation is the checkpoint (~341 MB) plus a handful of 256x256
#   float32 arrays. Asking for 8 cpus / 64G like train_text.sbatch would just make this job
#   queue longer for resources it would never touch.
#
# RESUMING AFTER A WALL-CLOCK KILL
#   Just `sbatch scripts/render_gallery.sbatch` again. The renderer rewrites manifest.json
#   after every cell (atomically), and on startup it skips any variant whose PNGs AND manifest
#   record both already exist. Seeds are derived as sha256(base_seed | cell_id | variant), so a
#   resumed variant is rendered with exactly the seed it would have had in the first run -- the
#   gallery does not become a patchwork of two different random draws.
# =============================================================================================

# --- knobs. The script reads these from the environment and prints every one in its banner. ---
# GALLERY_K       seed variants per cell. 4 x 57 = 228 images, ~12 MB of PNG. This is what
#                 stops repeated clicks on the same settings from returning the identical
#                 picture, which would make the site look like a lookup table.
# GALLERY_STEPS   DDIM steps. 200 is the paper's value (sample.py:22, augment_tumor.py:121).
#                 There is no reason to degrade it here -- this is the GPU, and the whole
#                 argument for pre-rendering is that the gallery can afford quality the live
#                 CPU path never could.
# GALLERY_BASE_SEED  salt for the per-image seeds. Changing it re-rolls the ENTIRE gallery;
#                 leave it alone unless you intend exactly that.
# GALLERY_BUNDLE  the conditioning bank from scripts/export_for_website.py. Build it first:
#                     python scripts/export_for_website.py --out web_bundle
#                 (login node is fine, no GPU needed). Without it this job exits immediately
#                 with that instruction rather than guessing at conditioning.
# GALLERY_OUT     output folder. Keep the name conddiff_gallery: the backend hard-codes
#                 inference/conddiff_gallery/manifest.json and will not find a renamed one.
export GALLERY_K=4
export GALLERY_STEPS=200
export GALLERY_BASE_SEED=20260101
export GALLERY_BUNDLE=web_bundle
export GALLERY_OUT=conddiff_gallery

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026

# -u = unbuffered stdout. Without it Python block-buffers when stdout is a file, so the
# per-image progress lines and the ETA would appear in the .log in 8 KB bursts -- and if the
# job is killed by the wall clock, the last (most interesting) buffer is lost entirely.
python -u scripts/render_gallery.py

# After it finishes, the script prints the sftp block that pulls conddiff_gallery/ down to the
# laptop. Use sftp, NOT scp: the U-M login banner is written into the channel scp uses for its
# own protocol and the transfer dies with "Received message too long 1349874536".
