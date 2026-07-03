#!/usr/bin/env python3
"""04_captions.py — auto-generate a text caption for every preprocessed slice.

WHY: text conditioning needs a text string paired with each training image, but
BraTS ships no radiology reports. So we DERIVE a caption from data we already have
inside each slice's .npy stack (shape (9,256,256)):

    channel 2    = the tumor segmentation mask (labels 1=necrotic, 2=edema, 3=enhancing)
    channels 3-8 = the 6 atlas lobe probability maps, in LOBES order below
                   -> tells us WHICH lobe any voxel belongs to

For each slice we work out four things — is there tumor? which components? which
lobe? which side? — then assemble a caption and write it to captions.csv, keyed by
the slice's filename so the dataset can look it up later.

Run:  python scripts/04_captions.py
"""
import os, glob, csv
import numpy as np

SLICES_DIR = "data/processed/slices"
SPLITS     = ["train", "val", "test"]
LOBES      = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]

# label codes inside the segmentation mask (channel 2)
NECROTIC, EDEMA, ENHANCING = 1, 2, 3

# thresholds (in voxels) so we ignore tiny specks that are noise, not real structure
MIN_TUMOR_VOXELS = 20     # a slice needs at least this much tumor to count as "a tumor slice"
MIN_LABEL_VOXELS = 10     # a single component must exceed this to be named

# --- laterality convention (VERIFY VISUALLY, then flip if wrong) ---------------------
# In the displayed images the horizontal (left-right) axis is array axis 0, and the
# anatomical midline sits near pixel 128 (the 240-wide brain zero-padded to 256 -> +8
# border each side). WHICH physical side the image-left is (patient's left vs right)
# depends on the volume's orientation convention (radiological vs neurological). We
# DEFAULT to radiological (image-left = patient's RIGHT). Confirm on a slice whose tumor
# side you can see, and flip IMAGE_LEFT_IS_PATIENT if the caption comes out mirrored.
MIDLINE_AXIS0         = 128
IMAGE_LEFT_IS_PATIENT = "right"      # radiological default; set "left" if verification disagrees

def region_phrase(lobe, side):
    """Turn (lobe, side) into natural English, e.g. 'the right temporal lobe'."""
    if lobe == "cerebellum":
        return f"the {side} cerebellum"       # cerebellum/insula aren't called "lobes"
    if lobe == "insula":
        return f"the {side} insula"
    return f"the {side} {lobe} lobe"

def caption_for(mask, atlas):
    """One slice in -> (fields dict, caption string) out.
    mask  : (256,256) segmentation channel
    atlas : (6,256,256) lobe probability channels
    """
    m = np.rint(mask).astype(np.int8)          # float16 -> clean integer labels 0,1,2,3
    tumor = m > 0                              # boolean map: any tumor voxel

    # --- 1. tumor-free slice? most slices are, since tumor touches only some z-levels ---
    if tumor.sum() < MIN_TUMOR_VOXELS:
        return dict(has_tumor=0, side="", lobe="", components=""), \
               "axial FLAIR slice of a brain with no tumor"

    # --- 2. which components are present (each with enough voxels to be real) ---
    present = []
    if (m == ENHANCING).sum() > MIN_LABEL_VOXELS: present.append("enhancing tumor")
    if (m == NECROTIC ).sum() > MIN_LABEL_VOXELS: present.append("necrotic core")
    if (m == EDEMA    ).sum() > MIN_LABEL_VOXELS: present.append("surrounding edema")
    if not present:                            # tumor exists but every type is tiny
        present = ["tumor"]

    # --- 3. which lobe: the atlas lobe holding the most probability mass under the tumor ---
    # For each lobe map, sum its probabilities over just the tumor pixels; biggest wins.
    overlap = [atlas[k][tumor].sum(dtype=np.float32) for k in range(6)]
    lobe = LOBES[int(np.argmax(overlap))]

    # --- 4. which side: tumor centroid along the horizontal axis vs the midline ---
    cx = np.argwhere(tumor)[:, 0].mean()       # mean position of tumor pixels along axis 0
    image_side = "left" if cx < MIDLINE_AXIS0 else "right"
    if IMAGE_LEFT_IS_PATIENT == "right":       # map image side -> patient side
        side = {"left": "right", "right": "left"}[image_side]
    else:
        side = image_side

    # --- 5. assemble the caption from the fields ---
    if len(present) == 1:
        desc = present[0]
    else:
        desc = ", ".join(present[:-1]) + " and " + present[-1]   # "a, b and c"
    caption = f"axial FLAIR slice with {desc} in {region_phrase(lobe, side)}"

    return dict(has_tumor=1, side=side, lobe=lobe, components="|".join(present)), caption

def main():
    rows = []
    for split in SPLITS:
        files = sorted(glob.glob(os.path.join(SLICES_DIR, split, "*.npy")))
        for n, path in enumerate(files):
            stack = np.load(path)              # (9,256,256) float16
            fields, caption = caption_for(stack[2], stack[3:9])   # ch2 = mask, ch3-8 = atlas
            rows.append((os.path.basename(path), split, fields["has_tumor"],
                         fields["side"], fields["lobe"], fields["components"], caption))
            if (n + 1) % 5000 == 0:
                print(f"  {split}: {n+1}/{len(files)}")
        print(f"{split}: {len(files)} slices captioned")

    out = os.path.join(SLICES_DIR, "captions.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "has_tumor", "side", "lobe", "components", "caption"])
        w.writerows(rows)
    n_tumor = sum(r[2] for r in rows)
    print(f"\nsaved {len(rows)} captions to {out}  "
          f"({n_tumor} tumor / {len(rows) - n_tumor} no-tumor)")

if __name__ == "__main__":
    main()
