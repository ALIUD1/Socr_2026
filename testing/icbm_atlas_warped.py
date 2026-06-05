import nibabel as nib, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sri = np.squeeze(nib.load("data/raw/templates/sri24/inst/extdata/spgr.nii.gz").get_fdata())
for lobe in ["frontal", "occipital", "temporal"]:
    m = np.squeeze(nib.load(f"data/processed/atlas_in_brats/atlas_{lobe}_in_sri24.nii.gz").get_fdata())
    z = sri.shape[2] // 2
    ms = m[:, :, z].T
    plt.figure(figsize=(6, 6))
    plt.imshow(sri[:, :, z].T, cmap="gray", origin="lower")
    plt.imshow(np.ma.masked_where(ms < ms.max()*0.1, ms), cmap="hot", origin="lower", alpha=0.5)
    plt.title(f"{lobe} on SRI24"); plt.axis("off")
    plt.savefig(f"outputs/check_{lobe}.png", dpi=120)
print("saved checks")