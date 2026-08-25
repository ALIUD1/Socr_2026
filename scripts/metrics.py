#!/usr/bin/env python3
"""metrics.py — manuscript-grade metrics for ANY generative model's output.

Deliberately model-agnostic: it evaluates IMAGE SETS, not models. Point it at a folder of
generated images and a folder of real ones and it reports the four metrics the manuscript
already uses (FID, SSIM, MSE, PSNR) plus background SNR, each with bootstrap 95% CIs. The same
script therefore scores our diffusion models AND the SOCR GAN / wavelet / 3D models as soon as
their weights are available -- no per-model code needed.

METRICS AND WHAT EACH IS FOR
  FID    Distribution-level realism. Compares the STATISTICS of two image sets in Inception-v3
         feature space, so it needs no pairing -- the right metric for a generator, because it
         does not punish a model for producing a different-but-plausible image. Lower better.
  SSIM   Perceptual structural similarity to a PAIRED real image, in [0,1]. Higher better.
  PSNR   Pixel fidelity in dB (log scale: +10 dB = 10x smaller MSE). Higher better.
  MSE    Raw mean squared pixel error. Lower better.
  bgSNR  No-reference noise measure: mean brain signal / std of the should-be-black background.
         Higher = cleaner. Needs no real counterpart, only a background definition.

PAIRING: SSIM/PSNR/MSE need a real counterpart per generated image. If the two folders share
filenames they are paired by name; otherwise those three are skipped and only FID + bgSNR are
reported -- which is the honest thing to do for an unconditional model.

BOOTSTRAP: a mean is itself an estimate. Resampling the per-image values with replacement 1000
times and taking the 2.5/97.5 percentiles gives a 95% CI, which is what makes a number citable
rather than anecdotal. The manuscript already states 1,000 bootstrap resamples, so this matches.

Usage
  python scripts/metrics.py --gen outputs/eval/mymodel --real data/real_slices --name mymodel
Accepts .npy (2D array or 3D volume) and .png/.jpg. A 3D volume contributes several axial slices.
"""
import sys, os, csv, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    HAVE_SSIM = True
except ImportError:
    HAVE_SSIM = False

try:
    import torch, torchvision
    from scipy import linalg
    HAVE_FID = True
except ImportError:
    HAVE_FID = False

N_BOOT = 1000
RNG = np.random.default_rng(0)          # seeded -> the CIs are reproducible

def load_images(folder):
    """Return {name: 2D float array in [0,1]}. A 3D .npy contributes 3 evenly spaced axial slices."""
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        ext = os.path.splitext(path)[1].lower()
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            if ext == ".npy":
                a = np.load(path).astype(np.float32)
                if a.ndim == 4:
                    a = a[0]                                   # (C,D,H,W) -> first channel
                if a.ndim == 3 and min(a.shape) > 8:           # a volume, not a small stack
                    for k, frac in enumerate((0.35, 0.5, 0.65)):
                        out[f"{stem}_z{k}"] = a[:, :, int(frac * a.shape[2])]
                    continue
                if a.ndim == 3:
                    a = a[0]                                   # (C,H,W) -> first channel
                out[stem] = a
            elif ext in (".png", ".jpg", ".jpeg"):
                from PIL import Image
                out[stem] = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
    for k, v in list(out.items()):
        if v.max() > 1.5:                                      # e.g. uint8 that slipped through
            out[k] = v / 255.0
    return out

def psnr_mse(gen, real, mask=None, data_range=1.0):
    """PSNR in dB and the MSE it came from, optionally restricted to a mask (the brain)."""
    d = (gen - real) ** 2
    mse = float(d[mask].mean() if mask is not None else d.mean())
    p = float("inf") if mse == 0 else 10 * np.log10(data_range ** 2 / mse)
    return p, mse

def bg_snr(gen, real):
    """Mean brain signal / std of background. The REAL image defines where background is, since
    it is skull-stripped to exactly 0 -- so this scores how cleanly the model renders empty space."""
    brain, bg = real > 0.05, real == 0.0
    if bg.sum() < 10 or brain.sum() < 10:
        return np.nan
    s = gen[bg].std()
    return float("inf") if s == 0 else float(gen[brain].mean() / s)

def inception_features(images, device="cpu", batch=32):
    """2048-d Inception-v3 pool features. Grayscale is replicated to 3 channels and resized to
    299x299, which is what the pretrained network expects."""
    weights = torchvision.models.Inception_V3_Weights.DEFAULT
    net = torchvision.models.inception_v3(weights=weights)
    net.fc = torch.nn.Identity()                               # drop the classifier -> features
    net.eval().to(device)
    feats = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.stack(images[i:i + batch]))[:, None]      # (B,1,H,W)
        x = x.repeat(1, 3, 1, 1)                                          # -> RGB
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear",
                                            align_corners=False)
        x = (x - 0.5) / 0.5                                               # -> [-1,1]
        with torch.no_grad():
            feats.append(net(x.to(device)).cpu().numpy())
    return np.concatenate(feats, 0)

def fid_from_feats(fr, fg):
    """FID = ||mu_r - mu_g||^2 + Tr(Sr + Sg - 2*sqrt(Sr Sg)). Matrix sqrt via scipy."""
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    Sr, Sg = np.cov(fr, rowvar=False), np.cov(fg, rowvar=False)
    covmean, _ = linalg.sqrtm(Sr.dot(Sg), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real                     # tiny imaginary parts are numerical noise
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(Sr + Sg - 2 * covmean))

def boot_ci(vals, n=N_BOOT):
    """Mean plus a 95% CI, by resampling the per-image values with replacement."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    means = [RNG.choice(v, size=v.size, replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def fmt(m, lo, hi, d=3):
    return "n/a" if not np.isfinite(m) else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="folder of generated images")
    ap.add_argument("--real", required=True, help="folder of real images")
    ap.add_argument("--name", required=True, help="model name for the results table")
    ap.add_argument("--out", default="outputs/metrics/all_models.csv")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    gen, real = load_images(a.gen), load_images(a.real)
    if not gen or not real:
        raise SystemExit(f"no images loaded (gen={len(gen)}, real={len(real)})")
    shared = sorted(set(gen) & set(real))
    print(f"{a.name}: {len(gen)} generated | {len(real)} real | {len(shared)} paired by filename")

    ssims, psnrs, mses, snrs = [], [], [], []
    for k in shared:
        g, r = gen[k], real[k]
        if g.shape != r.shape:
            continue
        p, m = psnr_mse(g, r, r > 0.05)
        psnrs.append(p); mses.append(m)
        if HAVE_SSIM:
            ssims.append(_ssim(r, g, data_range=1.0))
    if not HAVE_SSIM:
        print("  scikit-image missing -> SSIM skipped (pip install scikit-image)")

    ref = real[shared[0]] if shared else next(iter(real.values()))
    for k, g in gen.items():
        r = real.get(k, ref)
        if g.shape == r.shape:
            snrs.append(bg_snr(g, r))

    fid = np.nan
    if HAVE_FID:
        try:
            gg = [v for v in gen.values() if v.ndim == 2]
            rr = [v for v in real.values() if v.ndim == 2]
            n = min(len(gg), len(rr))
            if n >= 10:
                fid = fid_from_feats(inception_features(rr[:n], a.device),
                                     inception_features(gg[:n], a.device))
            else:
                print("  FID needs >=10 images per set; skipped")
        except Exception as e:
            print(f"  FID failed: {e}")
    else:
        print("  torchvision/scipy missing -> FID skipped (pip install torchvision scipy)")

    rows = {"SSIM": boot_ci(ssims), "PSNR_dB": boot_ci(psnrs),
            "MSE": boot_ci(mses), "bgSNR": boot_ci(snrs)}
    print(f"\n=== {a.name} ===")
    print("  FID      : " + (f"{fid:.3f}" if np.isfinite(fid) else "n/a"))
    for k, (m, lo, hi) in rows.items():
        print(f"  {k:9s}: {fmt(m, lo, hi, 4 if k == 'MSE' else 3)}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "n_gen", "n_real", "n_paired", "FID",
                        "SSIM", "SSIM_lo", "SSIM_hi", "PSNR_dB", "PSNR_lo", "PSNR_hi",
                        "MSE", "MSE_lo", "MSE_hi", "bgSNR", "bgSNR_lo", "bgSNR_hi"])
        w.writerow([a.name, len(gen), len(real), len(shared), f"{fid:.4f}"] +
                   [f"{x:.6f}" for k in ("SSIM", "PSNR_dB", "MSE", "bgSNR") for x in rows[k]])
    print(f"\nappended to {a.out}")

    s, p, m2, b = rows["SSIM"], rows["PSNR_dB"], rows["MSE"], rows["bgSNR"]
    latex = "{} & {:.2f} & {:.3f} & {:.4f} & {:.1f} & {:.1f}".format(
        a.name.replace("_", r"\_"), fid, s[0], m2[0], p[0], b[0]) + r" \\"
    print("\nLaTeX row (model & FID & SSIM & MSE & PSNR & bgSNR):")
    print("  " + latex)

if __name__ == "__main__":
    main()
