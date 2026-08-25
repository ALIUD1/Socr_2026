#!/usr/bin/env python3
"""06_captions_v2.py — RICHER captions, so the text prompt has more it can actually control.

WHY v2 EXISTS
    The v1 captions (04_captions.py) said only lobe + side + components:
        "axial FLAIR slice with enhancing tumor in the left frontal lobe"
    A model can only learn to control what appears in its captions, so text could steer
    laterality (proven: +0.025, z=9.6, 100% of held-out slices) and nothing else. Size,
    slice level and severity were never words, so they were never controllable.

    v2 adds three attributes, chosen because each is (a) derivable from data we already have,
    (b) NOT already given by the mask when the mask is dropped, and (c) a control someone would
    actually want:
        SIZE         small / moderate / large      <- from the lesion's area
        SLICE LEVEL  inferior / middle / superior  <- from the z index in the filename
        DOMINANCE    "predominantly edema" etc.    <- from which component is largest

    Example v2 caption:
        "axial FLAIR slice through the superior brain with a large enhancing tumor and
         surrounding edema, predominantly edema, in the left frontal lobe"

    Writes captions_v2.csv alongside the v1 captions.csv, so nothing is overwritten and the two
    can be compared head-to-head by retraining with each.

Run:  python scripts/06_captions_v2.py
"""
import os, glob, csv, re
import numpy as np

SLICES_DIR = "data/processed/slices"
SPLITS     = ["train", "val", "test"]
LOBES      = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]
OUT_NAME   = "captions_v2.csv"

NECROTIC, EDEMA, ENHANCING = 1, 2, 3
MIN_TUMOR_VOXELS = 20
MIN_LABEL_VOXELS = 10

# Area thresholds in pixels; 1 px = 1 mm^2 at BraTS's 1 mm resolution, so these are ~mm^2.
# 500 mm^2 ~ a 25 mm-diameter lesion, 2000 mm^2 ~ a 50 mm one. The script prints the realised
# distribution at the end so you can retune if the buckets come out lopsided.
SMALL_MAX, LARGE_MIN = 500, 2000

# z index of the slice within the 155-slice volume, parsed from the filename "<patient>_z###.npy".
INFERIOR_MAX, SUPERIOR_MIN = 60, 100

MIDLINE_AXIS0         = 128
IMAGE_LEFT_IS_PATIENT = "right"     # radiological; verified LPS via nib.aff2axcodes

def region_phrase(lobe, side):
    if lobe == "cerebellum":
        return f"the {side} cerebellum"
    if lobe == "insula":
        return f"the {side} insula"
    return f"the {side} {lobe} lobe"

def size_word(n):
    return "small" if n < SMALL_MAX else ("large" if n > LARGE_MIN else "moderate")

def level_word(z):
    if z is None:
        return None
    return "inferior" if z < INFERIOR_MAX else ("superior" if z > SUPERIOR_MIN else "middle")

def caption_for(mask, atlas, z=None):
    """One slice in -> (fields dict, caption string). Same contract as v1's caption_for, plus
    the extra fields, so downstream code can swap v1 for v2 without changes."""
    m = np.rint(mask).astype(np.int8)
    tumor = m > 0
    lvl = level_word(z)
    lvl_phrase = f" through the {lvl} brain" if lvl else ""

    if tumor.sum() < MIN_TUMOR_VOXELS:
        return (dict(has_tumor=0, side="", lobe="", components="", size="", level=lvl or "",
                     area=0),
                f"axial FLAIR slice{lvl_phrase} of a brain with no tumor")

    counts = {"enhancing tumor": int((m == ENHANCING).sum()),
              "necrotic core":   int((m == NECROTIC).sum()),
              "surrounding edema": int((m == EDEMA).sum())}
    present = [k for k, v in counts.items() if v > MIN_LABEL_VOXELS]
    if not present:
        present, counts = ["tumor"], {"tumor": int(tumor.sum())}

    # which component dominates -- a severity-ish cue the mask gives away only when present
    dominant = max(present, key=lambda k: counts[k])
    dom_short = {"enhancing tumor": "enhancing tissue", "necrotic core": "necrosis",
                 "surrounding edema": "edema", "tumor": "tumor"}[dominant]

    overlap = [atlas[k][tumor].sum(dtype=np.float32) for k in range(6)]
    lobe = LOBES[int(np.argmax(overlap))]

    cx = np.argwhere(tumor)[:, 0].mean()
    image_side = "left" if cx < MIDLINE_AXIS0 else "right"
    side = ({"left": "right", "right": "left"}[image_side]
            if IMAGE_LEFT_IS_PATIENT == "right" else image_side)

    area = int(tumor.sum())
    size = size_word(area)

    desc = present[0] if len(present) == 1 else ", ".join(present[:-1]) + " and " + present[-1]
    caption = (f"axial FLAIR slice{lvl_phrase} with a {size} {desc}, "
               f"predominantly {dom_short}, in {region_phrase(lobe, side)}")

    return (dict(has_tumor=1, side=side, lobe=lobe, components="|".join(present),
                 size=size, level=lvl or "", area=area),
            caption)

def main():
    rows, sizes, levels = [], [], []
    for split in SPLITS:
        files = sorted(glob.glob(os.path.join(SLICES_DIR, split, "*.npy")))
        for n, path in enumerate(files):
            base = os.path.basename(path)
            mz = re.search(r"_z(\d+)\.npy$", base)          # pull z out of "<patient>_z075.npy"
            z = int(mz.group(1)) if mz else None
            stack = np.load(path)
            fields, caption = caption_for(stack[2], stack[3:9], z)
            rows.append((base, split, fields["has_tumor"], fields["side"], fields["lobe"],
                         fields["components"], fields["size"], fields["level"],
                         fields["area"], caption))
            if fields["has_tumor"]:
                sizes.append(fields["size"])
            levels.append(fields["level"])
            if (n + 1) % 5000 == 0:
                print(f"  {split}: {n+1}/{len(files)}")
        print(f"{split}: {len(files)} slices captioned")

    out = os.path.join(SLICES_DIR, OUT_NAME)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "has_tumor", "side", "lobe", "components",
                    "size", "level", "area", "caption"])
        w.writerows(rows)

    n_t = sum(r[2] for r in rows)
    print(f"\nsaved {len(rows)} captions to {out}  ({n_t} tumour / {len(rows)-n_t} no-tumour)")
    # Verify the buckets are usable: a bucket holding <5% of cases gives the model almost no
    # examples to learn that word from, so retune the thresholds if one is starved.
    for name, vals in (("size (tumour slices)", sizes), ("level (all slices)", levels)):
        uniq, cnt = np.unique([v for v in vals if v], return_counts=True)
        dist = "  ".join(f"{u}={c} ({100*c/max(cnt.sum(),1):.0f}%)" for u, c in zip(uniq, cnt))
        print(f"  {name}: {dist}")

if __name__ == "__main__":
    main()
