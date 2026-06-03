#!/usr/bin/env python3
"""02_register_atlas.py — register the ICBM atlas into BraTS/SRI24 space (all 6 lobes)."""
import os
import numpy as np
import nibabel as nib
import ants

FIXED      = "data/raw/templates/sri24/inst/extdata/spgr.nii.gz"
T1_ANALYZE = "data/raw/templates/icbm452_atlas_air12.zipfile/icbm452_atlas_air12_sinc.hdr"
PROB_TMPL  = "data/raw/icbm/icbm452_atlas_probability_{lobe}.zipfile/icbm452_atlas_probability_{lobe}.hdr"
LOBES      = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]
NIFTI_DIR  = "data/processed/icbm_nifti"      # converted (Analyze -> NIfTI) inputs
OUT_DIR    = "data/processed/atlas_in_brats"  # final warped maps in BraTS space


def analyze_to_nifti(in_path, out_path):
    """Read an Analyze file with nibabel (correct orientation) and re-save as NIfTI
    so ANTs can't misread the orientation. This is the fix for the flip bug."""
    img  = nib.load(in_path)
    data = np.squeeze(img.get_fdata())                       # drop singleton 4th dim
    img3 = nib.as_closest_canonical(nib.Nifti1Image(data, img.affine))
    nib.save(img3, out_path)
    return out_path


def main():
    os.makedirs(NIFTI_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- TODO 1: convert the ICBM T1 to NIfTI, then load fixed + moving into ANTs ----
    # hints:
    #   - call analyze_to_nifti(T1_ANALYZE, <NIFTI_DIR>/"icbm452_T1.nii.gz")
    #   - fixed  = ants.image_read(FIXED)
    #   - moving = ants.image_read(<the converted T1 path>)
    t1_nii = analyze_to_nifti(T1_ANALYZE, os.path.join(NIFTI_DIR, "icbm452_T1.nii.gz"))
    fixed = ants.image_read(FIXED)
    moving = ants.image_read(t1_nii)

    # ---- TODO 2: register moving -> fixed with SyN, and save the warped T1 ----
    # hints:
    #   - reg = ants.registration(fixed=..., moving=..., type_of_transform="SyN")
    #   - ants.image_write(reg["warpedmovout"], os.path.join(OUT_DIR, "icbm452_T1_in_sri24.nii.gz"))
    reg = ants.registration(fixed = fixed, moving = moving, type_of_transform = "SyN")
    ants.image_write(reg["warpedmovout"], os.path.join(OUT_DIR, "icbm452_T1_in_sri24.nii.gz"))
    # ---- TODO 3: for each lobe: convert its prob map -> NIfTI, apply the SAME transform,
    #              and save the warped map. Use interpolator="linear" (smooth probabilities). ----
    # hints:
    #   for lobe in LOBES:
    #       in_path  = PROB_TMPL.format(lobe=lobe)
    #       nii_path = analyze_to_nifti(in_path, os.path.join(NIFTI_DIR, f"{lobe}_prob.nii.gz"))
    #       prob   = ants.image_read(nii_path)
    #       warped = ants.apply_transforms(fixed=fixed, moving=prob,
    #                                      transformlist=reg["fwdtransforms"], interpolator="linear")
    #       ants.image_write(warped, os.path.join(OUT_DIR, f"atlas_{lobe}_in_sri24.nii.gz"))
    #       print("warped", lobe)
    for lobe in LOBES:
        in_path = PROB_TMPL.format(lobe = lobe)
        nii_path = analyze_to_nifti(in_path, os.path.join(NIFTI_DIR, f"{lobe}_prob.nii.gz"))
        prob = ants.image_read(nii_path)
        warped = ants.apply_transforms(fixed = fixed, moving = prob, transformlist = reg["fwdtransforms"], interpolator = "linear")
        ants.image_write(warped, os.path.join(OUT_DIR, f"atlas_{lobe}_in_sri24.nii.gz"))
        print("warped", lobe)
        
    print("done — 6 lobe maps in", OUT_DIR)


if __name__ == "__main__":
    main()