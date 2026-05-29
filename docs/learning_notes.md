# Learning Notes

A running explainer for the concepts behind this project. Reread anytime.
Each section answers "what is this and why do I care."

---

## 1. What a brain MRI actually IS (the mental model)

A brain MRI is **not** a photo. It's a **3D grid of numbers**.

- The grid is made of tiny cubes called **voxels** (= "volume pixels", the 3D
  version of a pixel).
- Each voxel holds **one number = the signal intensity** at that point in the
  brain. Brighter = higher number.
- A BraTS FLAIR volume is **240 × 240 × 155** voxels. That's 240 wide, 240 tall,
  155 deep — about 8.9 million numbers describing one brain.
- In Python (numpy) this is just a 3D array: `volume[x, y, z]` gives the intensity
  at one voxel.

A **slice** is one 2D plane pulled out of that cube, e.g. `volume[:, :, 75]` is the
75th axial ("from the top") slice — a 240×240 image. Our 2D model trains on slices.

The three viewing directions:
- **Axial** = horizontal slices (top-down) → `volume[:, :, z]`
- **Coronal** = front-to-back slices → `volume[:, y, :]`
- **Sagittal** = side slices (left-right) → `volume[x, :, :]`

## 2. The tumor mask (segmentation) is a DIFFERENT kind of volume

Same 240×240×155 grid, but the numbers mean **categories**, not brightness:
- 0 = background / normal tissue
- 1 = necrotic tumor core (NCR)
- 2 = peritumoral edema (ED)
- 3 = enhancing tumor (ET)   ← BraTS 2023 uses 3 here (older versions used 4)

This is why a mask must use **nearest-neighbor interpolation** if it's ever
resized: averaging would create nonsense like "label 1.5", but there is no tissue
type 1.5. The label must stay a clean integer.

## 3. File formats

- **NIfTI** (`.nii`) — the standard brain-imaging file. Stores the voxel grid PLUS
  a header (dimensions, voxel size in mm, orientation).
- **`.nii.gz`** — the exact same thing, gzip-compressed. Smaller on disk, tiny bit
  slower to read. `nibabel` reads both identically. (Our BraTS files are plain `.nii`.)
- **Analyze 7.5** (`.hdr` + `.img` pair) — an older format; the header and data are
  in two separate files. The ICBM atlas uses this.

We read all of these with one library: **`nibabel`**.
```python
img    = nib.load("scan.nii")    # opens header (lazy — no voxels loaded yet)
volume = img.get_fdata()         # NOW pulls the voxels into a numpy array
```

## 4. Voxel size, affines, and why "registration" exists

The header stores **voxel size in millimeters** (BraTS = 1×1×1 mm) so software knows
the real-world scale, and an **affine matrix** — a 4×4 table that maps array indices
`[i,j,k]` to real-world coordinates `[x,y,z] in mm`. The affine encodes position,
orientation, and spacing.

Two volumes "line up" only if they describe the **same physical space**. BraTS
patients are all pre-aligned to a template called **SRI24** (240×240×155, 1mm), so
they line up with each other automatically. The **ICBM atlas is in a different space**
(149×188×148, its own affine), so it does NOT line up yet.

**Registration** = computing the transform that warps one volume into another's
space so they overlap voxel-for-voxel. We'll register the **atlas → SRI24 once** and
reuse it for every patient (because patients are already in SRI24).

## 5. The inspection script, concept by concept

File: `scripts/01_inspect_brats.py`. What each idea teaches you:

- **`argparse`** — lets the script take an argument from the command line
  (`python script.py SOME_PATH`). Standard way to make a reusable script.
- **`glob`** — find files by a wildcard pattern. `*-t2f.nii*` means "anything ending
  in -t2f.nii or -t2f.nii.gz". This is how we locate the FLAIR file without
  hardcoding the patient ID.
- **`matplotlib.use("Agg")`** — a compute node has NO screen. "Agg" tells matplotlib
  to draw straight to an image file instead of trying to pop open a window (which
  would crash). Always do this for headless/cluster plotting.
- **`nib.load(...).get_fdata()`** — load voxels into numpy (see section 3).
- **`.astype(int)`** on the mask — force labels to integers, since they're categories.
- **`flair.shape == seg.shape`** — the single most important check. If image and mask
  aren't the same grid, the mask doesn't describe the image and everything downstream
  is wrong.
- **`np.unique(seg, return_counts=True)`** — lists every distinct value in the mask
  and how many voxels have it. Confirms we see exactly {0,1,2,3} and shows tumor size.
- **`(seg > 0).sum(axis=(0,1))`** — `seg > 0` makes a True/False tumor mask; summing
  over the x and y axes counts tumor voxels per z-slice. `.argmax()` finds the slice
  with the most tumor, so the overlay is actually informative.
- **`.T` and `origin="lower"`** — display conventions so the brain is oriented the way
  you expect on screen. They do NOT change the data, only how it's drawn.
- **`np.ma.masked_where(seg==0, ...)`** — hide background so only tumor voxels get
  colored in the overlay.

## 6. numpy skills you're using (worth practicing)

numpy is THE array library for scientific Python. The moves that matter here:
- **shape**: `arr.shape` → the dimensions, e.g. `(240, 240, 155)`.
- **indexing/slicing**: `arr[:, :, 75]` → one slice; `:` means "all of this axis".
- **boolean masks**: `arr > 0` → array of True/False, used to select/count.
- **reductions over axes**: `arr.sum(axis=(0,1))` → collapse some dimensions.
- **dtype**: the kind of number (float vs int). Masks are int; images are float.

## 7. The skills roadmap for the whole project

Where we are = step 2. The skills each step will teach you:

1. ✅ HPC basics — SSH, storage tiers (home/Turbo/scratch), SLURM, symlinks, git workflow.
2. ⬅ **Data inspection** — nibabel, numpy, sanity-checking volumes (this script).
3. Atlas inspection + **image registration** (ANTs) — coordinate spaces, transforms,
   interpolation (linear for images, nearest-neighbor for masks).
4. **Preprocessing** — intensity normalization, cropping/resizing, saving tensors.
5. **PyTorch Dataset / DataLoader** — how data is fed to a model in batches; the
   crucial **patient-level** train/val/test split (never split slices of one patient
   across sets).
6. **Conditional U-Net** (baseline) — convolutional networks, encoder/decoder,
   multi-channel input (FLAIR + mask + atlas), loss functions (L1/L2), training loop
   (forward → loss → backprop → update), validation metrics (SSIM, PSNR).
7. **Diffusion models** — the real goal: how they add and learn to remove noise,
   conditioning, why they produce sharp realistic samples and avoid copying.
8. **Evaluation** — realism, tumor/Dice preservation, anatomical consistency, and a
   privacy / non-copy (nearest-neighbor similarity) check.
9. **3D extension** — patches / 2.5D / latent diffusion to fit GPU memory.

## 8. Glossary (terms you'll hear constantly)

- **voxel** — one cube in a 3D image (3D pixel).
- **modality** — a type of MRI scan/contrast (T1, T1ce, T2, FLAIR).
- **FLAIR / t2f** — the modality where edema/tumor is easy to see; our first target.
- **segmentation / mask** — a label map marking which voxels are tumor.
- **affine** — 4×4 matrix mapping array indices to real-world mm coordinates.
- **registration** — warping one image into another's coordinate space.
- **interpolation** — estimating values between grid points when resampling
  (linear for images, nearest-neighbor for masks).
- **SLURM** — the cluster job scheduler you ask for compute nodes.
- **SRI24** — the template space BraTS patients are aligned to.
- **conditioning** — extra inputs that steer a generative model (here: mask + atlas).
