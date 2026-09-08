#!/usr/bin/env python3
"""export_for_website.py — build the self-contained bundle the Brain-Image-Generator site needs.

WHY THIS SCRIPT EXISTS
    The paper's model (braingen_CondDiffuser_BraTS_v1) is CONDITIONAL. Its U-Net takes 9 input
    channels, and only ONE of them is the thing being generated:

        ch0   = the noisy FLAIR being denoised   <- the model's OUTPUT
        ch1   = a real patient's T1              <- anatomy
        ch2   = the tumour mask (labels 0/1/2/3) <- what/where the pathology is
        ch3-8 = 6 ICBM452 lobe probability maps warped to SRI24  <- WHERE the lobes are

    Channels 1-8 are CONDITIONING. The website has no concept of any of them, and it cannot
    manufacture them either:

      * the T1 is a real held-out patient volume percentile-normalized per-VOLUME
        (03_preprocess.py:22-28) — you cannot recreate one from a 2D slice;
      * the 6 atlas maps come from a ONE-TIME ANTs SyN registration (02_register_atlas.py) and
        are then divided by a SINGLE GLOBAL SCALAR across all 6 lobes and all 155 z-slices
        (03_preprocess.py:55, `atlas = atlas / atlas.max()`). PROB_TH = 0.4 in augment_tumor.py
        is calibrated against exactly that scaling. Re-deriving the atlas anywhere else — even
        "correctly" — silently changes every lesion size the site produces.

    So the conditioning must be SHIPPED, verbatim, as preprocessed 03_preprocess.py output.
    That is what this script packages: a small "conditioning bank" of held-out validation
    slices plus the trained checkpoint, in one folder you sftp to the laptop and drop into the
    web backend.

WHAT THE WEBSITE DOES WITH IT
    The user picks Tumour / Lobe / Slice Location / Tumour Size. The backend reads
    manifest.json (a few KB), finds a slice whose LEVEL matches Slice Location and in which the
    requested LOBE is viable, loads that ONE .npy, runs augment_tumor.py's synth_mask() to draw
    a tumour in that lobe, swaps it into the mask channel, and runs the DDIM loop. The manifest
    is the whole point of Phase B below: the backend must never open 70 .npy files to answer
    one request.

WHAT IS DELIBERATELY *NOT* SHIPPED
    Channel 0 — the real patient's FLAIR. The model never reads it (src/dataset.py:40 slices
    `cond = stack[1:9]`), it is 1/9th of the bytes, and shipping real patient FLAIR scans to a
    public web server is a data-use question nobody needs to have. Pass --keep-flair if you
    specifically want the real FLAIR for a side-by-side "reference" panel and have cleared it.

    Also not shipped: BraTS itself, the ICBM .hdr/.img atlas, the SRI24 template,
    manifest.csv, captions*.csv. None of them are on the inference path.

RUN IT ON GREAT LAKES
    This script needs data/processed/slices/val/ and models/diffusion_ema.pt, neither of which
    exists on the laptop. A login node is enough — Phase A is pure numpy I/O and Phase C only
    builds the U-Net on CPU to check the weights fit; no GPU, no sampling.

    ACTIVATE THE brainmri VENV FIRST, even for --dry-run. The threshold constants are imported
    from augment_tumor (see the block below), and that module imports torch, diffusers, scipy
    and matplotlib at ITS top — so a bare `python scripts/export_for_website.py --dry-run`
    outside the venv dies with ModuleNotFoundError before it prints anything.

    python scripts/export_for_website.py                     # full export -> web_bundle/
    python scripts/export_for_website.py --stride 10         # fast scan of every 10th slice
    python scripts/export_for_website.py --dry-run           # select + report, write nothing
    python scripts/export_for_website.py --out /scratch/wb --per-cell 6
"""
import sys, os, re, glob, json, shutil, hashlib, argparse, subprocess
from datetime import datetime, timezone

# Make the repo root importable so `from src.model import build_model` works no matter what
# directory you launched from. Same two lines every script in scripts/ opens with.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ...and scripts/ itself, so we can import augment_tumor as a sibling module. check_atlas_fit.py
# does exactly this for exactly the same reason (check_atlas_fit.py:23-24).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ---------------------------------------------------------------------------------------------
# CONSTANTS ARE IMPORTED, NEVER RETYPED — AND THE SOURCE OF TRUTH IS src/conddiff_core.py.
#
# Every threshold below decides which slices the website can offer. If the module that SELECTS
# the bank and the module that USES the bank ever disagree by even 0.05, the site will happily
# select a slice that synth_mask() then refuses to grow a tumour in — and the user gets a blank
# image with a green "Success" toast, because the web backend swallows exceptions.
#
# This block used to import from augment_tumor.py, which was correct when augment_tumor was the
# only implementation. It no longer is. src/conddiff_core.py now holds the shared logic and
# exists BYTE-IDENTICALLY in both repos (Brain-Image-Generator/backend/inference/conddiff_core.py),
# and both the cluster renderer (render_gallery.py) and the web backend (conddiff_inference.py)
# import their constants from it. If THIS script kept reading augment_tumor.py, the bank would be
# selected under one set of thresholds and consumed under another — exactly the divergence the
# shared core was created to make impossible. So we import from core too.
#
# Second, smaller benefit: augment_tumor.py imports torch, diffusers and scipy at module scope,
# so importing it just to read six numbers dragged the whole research stack into a script that
# only needs numpy. conddiff_core.py imports numpy and nothing else.
#
# What we get:
#   LOBES     ["frontal","parietal","temporal","occipital","cerebellum","insula"] — ORDER MATTERS,
#             it is what fixes atlas lobe k to stack channel 3+k (03_preprocess.py:10).
#   ATLAS0    3      — the channel offset of the first atlas map.
#   PROB_TH   0.4    — "this pixel is inside that lobe" iff atlas probability > 0.4.
#   MIN_LOBE  500    — px of the target lobe required; rejects "the occipital lobe is technically
#                      present as a 60-px sliver" slices.
# ---------------------------------------------------------------------------------------------
from src.conddiff_core import LOBES, ATLAS0, PROB_TH, MIN_LOBE

# ...and the SHAPE constants synth_mask() uses. This script never CALLS synth_mask -- the renderer
# and the web backend do. Importing them here lets the manifest CARRY them, which is how the
# backend proves its copy has not drifted: a silently different AREA_FRAC changes the size of
# every lesion the site draws while the checkpoint hash, the channel contract and the coverage
# table all still check out.
#
# Core names it AREA_FRAC_DEFAULT (because core also carries the per-size bands in
# AREA_FRAC_BY_SIZE, and an unqualified "AREA_FRAC" would be ambiguous between the two). It is
# the same (0.15, 0.35) tuple augment_tumor.py calls AREA_FRAC, so we alias on import and the
# rest of this file reads unchanged.
from src.conddiff_core import (AREA_FRAC_DEFAULT as AREA_FRAC,
                               CORE_FRAC, MIN_AREA, COMPACT, NOISE_SIGMA, NOISE_AMP)

# MIN_BRAIN is deliberately NOT in conddiff_core. It is the one gate that is a DEPLOYMENT choice
# rather than science: augment_tumor.py comments it as "require a full mid-axial slice, not a tiny
# inferior sliver", i.e. it is cosmetic, and some (lobe, level) cells cannot be filled without
# lowering it — the cerebellum only exists in the inferior slices that 15000 was written to
# reject. The backend re-verifies every candidate against this same threshold and reads its own
# value from CONDDIFF_MIN_BRAIN, so the two must MATCH, not be shared. Whatever value this script
# exports the bank with is written into manifest.json's `selection.min_brain`, and the deployment
# sets CONDDIFF_MIN_BRAIN to it. 15000 is augment_tumor.py:31's value, unchanged.
MIN_BRAIN = 15000

# The z-index bands that define the website's Slice Location control also come from core, which
# took them from 06_captions_v2.py — the paper's caption vocabulary — so the website's "Superior"
# means exactly what the paper's captions mean by "superior".
#
# This too used to be an importlib.import_module("06_captions_v2") dance (`import 06_captions_v2`
# is a SyntaxError, since a Python identifier cannot start with a digit) with a hardcoded
# fallback. Sourcing it from core instead removes the fragile string-import AND closes the same
# divergence hole as above: render_gallery.py buckets each slice's level with core.level_word, so
# if this script bucketed with a different function the bank's `level` and the gallery's cell
# `level` could disagree and a whole Slice Location option would come back empty.
from src.conddiff_core import INFERIOR_MAX, SUPERIOR_MIN, level_word, LEVELS
_LEVEL_SRC = "imported from src/conddiff_core.py"

# The three-way admissibility test needs a T1 "is this pixel brain?" threshold. 0.05 on the
# percentile-normalized T1 is the brain/background cut used everywhere in this repo
# (augment_tumor.py:128, check_atlas_fit.py:56) — background is forced to EXACTLY 0 by
# normalize_flair()'s `out[vol == 0] = 0`, so anything above a hair of noise is tissue.
BRAIN_TH = 0.05
# "tumour-free" means fewer than this many pixels carry ANY segmentation label. `> 0.5` catches
# label >= 1 given the labels are stored as float16 1.0/2.0/3.0 (augment_tumor.py:130).
MAX_TUMOUR_PX = 50

SLICE_SUBDIR = "cond_slices"                          # where the .npy files land inside the bundle
CKPT_NAME    = "diffusion_ema.pt"
MANIFEST     = "manifest.json"


# =============================================================================================
# small helpers
# =============================================================================================

def human(nbytes):
    """Bytes -> a human string. Purely cosmetic; the manifest always stores raw integers."""
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024.0


def sha256_of(path, chunk=1 << 20):
    """Streaming SHA-256 of a file. Streaming (1 MiB at a time) because the checkpoint is ~341 MB
    and `open(path,'rb').read()` would pull all of it into RAM for no reason."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain Python types.

    WHY THIS EXISTS, and it is not paranoia: every count below (`int(mask.sum())`, an argmax, a
    mean) is a numpy scalar, and `json.dump` raises
        TypeError: Object of type int64 is not JSON serializable
    on the first one it meets. Worse, the SAME dict eventually gets json.dumps'd by the web
    backend into Supabase's parameters_used column inside a try/except that PRINTS and returns
    None — so a stray np.int64 there becomes a blank image with a success toast. Coerce once,
    here, at the boundary."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def git_state():
    """Record which commit produced this bundle. Wrapped in try/except because the export must
    still work from a tarball with no .git directory."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL).decode().strip())
        return commit, dirty
    except Exception:
        return None, None


def dir_size(path):
    """Total bytes of every file under `path`, recursively (os.walk, so nested dirs count)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


# =============================================================================================
# PHASE A — scan the validation split and score every slice
# =============================================================================================

def scan_slices(slices_dir, stride, min_brain, verbose_every=2000):
    """Walk data/processed/slices/val/*.npy and return the ADMISSIBLE ones with their stats.

    The three tests are augment_tumor.py:130 verbatim — the same three that check_atlas_fit.py:56
    repeats, which is the confirmation they are *the* criteria and not incidental:

      1. enough brain      (s[1] > 0.05).sum() > MIN_BRAIN          — a real mid-axial slice
      2. tumour-FREE       (s[2] > 0.5).sum()  < 50                 — see below
      3. lobe present      ((s[3+k] > 0.4) & brain).sum() > MIN_LOBE — per lobe k

    Why test 2 matters most: we want to ADD one controlled tumour to an otherwise-healthy brain.
    If the slice already has a real BraTS tumour, the generated image contains that lesion PLUS
    ours, the website's "Lobe" control is no longer the only thing determining where pathology
    appears, and the mask we hand back is no longer ground truth for the whole image.

    Returns (candidates, stats) where candidates is a list of dicts:
        {path, file, patient, z, level, brain_px, tumour_px, lobes: {lobe_name: px, ...}}
    `lobes` contains ONLY the lobes that passed test 3, so `lobe in cand["lobes"]` is the
    viability question and `cand["lobes"][lobe]` is how prominent it is.
    """
    files = sorted(glob.glob(os.path.join(slices_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no slices found in {slices_dir} - did 03_preprocess.py run? "
                         f"(this script must be run on Great Lakes, not the laptop)")
    if stride > 1:
        files = files[::stride]                       # coarse subsample for a fast --dry-run

    # Filenames are "<patient>_z###.npy" (03_preprocess.py:89). The patient id can itself contain
    # underscores and digits, so anchor on the LAST "_z<digits>.npy" with $.
    name_re = re.compile(r"^(?P<patient>.+)_z(?P<z>\d+)\.npy$")

    cands = []
    # `unparsed` and `bad_shape` are counted SEPARATELY on purpose: they are different bugs with
    # different fixes. A high `unparsed` means the filename convention changed (03_preprocess.py:89
    # writes "<patient>_z###.npy"); a high `bad_shape` means 03_preprocess.py was re-run with a
    # different channel set and this whole bundle's channel contract is wrong. Rolling them into
    # one number would leave you unable to tell which.
    stats = dict(scanned=0, unparsed=0, bad_shape=0, failed_brain=0, failed_tumour=0,
                 failed_all_lobes=0, admitted=0)

    for n, path in enumerate(files):
        stats["scanned"] += 1
        m = name_re.match(os.path.basename(path))
        if not m:
            stats["unparsed"] += 1
            continue
        patient, z = m.group("patient"), int(m.group("z"))

        # mmap_mode="r" maps the file into the address space instead of reading it. Indexing one
        # channel then touches only that channel's ~131 KB of pages. Across ~24,000 val slices
        # this is the difference between reading ~27 GB and reading a few GB, because most slices
        # fail test 1 and we never touch channels 2-8 for those.
        stack = np.load(path, mmap_mode="r")
        if stack.shape != (9, 256, 256):
            stats["bad_shape"] += 1
            continue

        # --- test 1: enough brain ---------------------------------------------------------
        # np.asarray() materializes just this channel out of the mmap; .astype(np.float32) is the
        # non-negotiable cast — the file is float16 (03_preprocess.py:88) and half-precision
        # arithmetic here would be both slow and needlessly imprecise.
        t1 = np.asarray(stack[1], dtype=np.float32)   # (256,256) patient T1 in [0,1]
        brain = t1 > BRAIN_TH                          # (256,256) bool
        brain_px = int(brain.sum())
        if brain_px <= min_brain:
            stats["failed_brain"] += 1
            continue

        # --- test 2: tumour-free ----------------------------------------------------------
        # Channel 2 is the ONLY channel not in [0,1]: 03_preprocess.py:72 puts the raw BraTS
        # segmentation in unnormalized, so it carries the literal labels 0/1/2/3.
        seg = np.asarray(stack[2], dtype=np.float32)  # (256,256) labels {0,1,2,3}
        tumour_px = int((seg > 0.5).sum())
        if tumour_px >= MAX_TUMOUR_PX:
            stats["failed_tumour"] += 1
            continue

        # --- test 3: which lobes are genuinely present in THIS slice ----------------------
        lobes = {}
        for k, lobe in enumerate(LOBES):
            al = np.asarray(stack[ATLAS0 + k], dtype=np.float32)   # (256,256) probabilities [0,1]
            px = int(((al > PROB_TH) & brain).sum())               # lobe footprint in this slice
            if px > MIN_LOBE:
                lobes[lobe] = px
        if not lobes:
            stats["failed_all_lobes"] += 1
            continue

        stats["admitted"] += 1
        cands.append(dict(path=path, file=os.path.basename(path), patient=patient, z=z,
                          level=level_word(z), brain_px=brain_px, tumour_px=tumour_px,
                          lobes=lobes))

        if (n + 1) % verbose_every == 0:
            print(f"  scanned {n+1}/{len(files)}  admitted {stats['admitted']}")

    return cands, stats


# =============================================================================================
# PHASE B — choose which admissible slices to actually ship
# =============================================================================================

def select_slices(cands, per_cell):
    """Fill an 18-cell (6 lobes x 3 levels) table, preferring DISTINCT PATIENTS.

    The website's parameter space is Lobe x Slice Location = 6 x 3 = 18 cells, and every cell
    needs at least one slice or that combination of dropdowns has nothing to generate from.
    Two competing goals:

      * VARIETY. If one patient supplies every "frontal + middle" slice, clicking Generate twice
        with the same settings shows the same skull twice and the site looks like a lookup table.
        So within a cell we take at most one slice per patient.
      * QUALITY. Within a cell, a slice where the target lobe occupies 2000 px gives synth_mask()
        far more room than one where it barely clears MIN_LOBE = 500. So we sort each cell's
        candidates by that lobe's pixel area, descending, and take the top `per_cell` distinct
        patients.

    A single slice can legitimately serve several cells (one mid-axial slice usually shows
    frontal + parietal + temporal + insula at once), so we dedupe by path at the end — that is
    why the shipped file count is much smaller than 18 x per_cell.

    Returns (selected, coverage, cell_members):
        selected     list of candidate dicts, deduped, in a stable order
        coverage     {lobe: {level: count}}       — the 6x3 table we print and ship
        cell_members {"lobe_level": [filenames]}  — what the backend indexes by
    """
    by_level = {lvl: [c for c in cands if c["level"] == lvl] for lvl in LEVELS}

    selected = {}                                       # path -> candidate  (dict preserves order)
    coverage = {lobe: {lvl: 0 for lvl in LEVELS} for lobe in LOBES}
    cell_members = {}

    for lobe in LOBES:
        for lvl in LEVELS:
            # Only candidates at this level in which THIS lobe passed test 3, best-first.
            pool = [c for c in by_level[lvl] if lobe in c["lobes"]]
            pool.sort(key=lambda c: c["lobes"][lobe], reverse=True)

            used_patients, members = set(), []
            for c in pool:
                if len(members) >= per_cell:
                    break
                if c["patient"] in used_patients:       # one slice per patient per cell
                    continue
                used_patients.add(c["patient"])
                members.append(c["file"])
                selected.setdefault(c["path"], c)       # dedupe: a slice may serve many cells

            coverage[lobe][lvl] = len(members)
            cell_members[f"{lobe}_{lvl}"] = members

    return list(selected.values()), coverage, cell_members


def print_coverage(coverage, per_cell):
    """Print the 6x3 table and return the list of empty (lobe, level) cells.

    THIS IS THE MOST USEFUL OUTPUT OF THE WHOLE SCRIPT. An empty cell here is a dropdown
    combination the website cannot honour. Finding that out on the cluster, with a human
    reading a table, is the whole reason this print exists — the alternative is finding out in
    production, where the backend swallows the exception and returns HTTP 200 with no image.

    Some empty cells are ANATOMY, not a bad threshold: the cerebellum simply does not exist at
    z > 100, so (cerebellum, superior) can never be filled and no --min-brain value will help.
    Others are the threshold: MIN_BRAIN = 15000 exists specifically to reject "tiny inferior
    slivers" (augment_tumor.py:31), and the cerebellum lives in exactly those inferior slices,
    so (cerebellum, inferior) may be recoverable with --min-brain 9000.
    """
    print(f"\n  coverage - slices found per (lobe, level), target {per_cell} each")
    print(f"    {'lobe':<12}" + "".join(f"{lvl:>11}" for lvl in LEVELS))
    empty = []
    for lobe in LOBES:
        row = f"    {lobe:<12}"
        for lvl in LEVELS:
            n = coverage[lobe][lvl]
            row += f"{n:>10}" + ("!" if n == 0 else " ")   # '!' flags an unfillable dropdown combo
            if n == 0:
                empty.append(f"{lobe}_{lvl}")
        print(row)
    if empty:
        print(f"\n  WARNING: {len(empty)} of 18 cells are EMPTY: {', '.join(empty)}")
        print("    The website must DISABLE these Lobe x Slice Location combinations - they are")
        print("    recorded in manifest.json under 'empty_cells' so the backend can do that.")
        print("    If a cell looks recoverable (an inferior lobe rejected by the BRAIN-AREA")
        print("    gate) re-run with e.g. --min-brain 9000, then re-run check_atlas_fit.py")
        print("    and confirm containment is still 100% and coverage still lands in 15-35%.")
        print("    But if the cell is empty because the LOBE never clears MIN_LOBE = %d px"
              % MIN_LOBE)
        print("    (the insula is the likely one: 03_preprocess.py:55 scales all six atlas")
        print("    maps by ONE global maximum, so a small structure's peak probability may")
        print("    never exceed PROB_TH = %s anywhere), --min-brain cannot help and there is"
              % PROB_TH)
        print("    deliberately no --min-lobe flag: synth_mask itself re-tests")
        print("    `n_lobe < MIN_LOBE -> return None`, so exporting a slice below that")
        print("    threshold would ship a cell the backend is GUARANTEED to fail on. The only")
        print("    real lever is PROB_TH in augment_tumor.py, and lowering it changes the")
        print("    shape of every lesion the model draws - re-run check_atlas_fit.py if you do.")
    return empty


# =============================================================================================
# PHASE C — verify the checkpoint before we promise anything about it
# =============================================================================================

def verify_checkpoint(ckpt_path):
    """Load models/diffusion_ema.pt into build_model() and confirm it fits EXACTLY.

    Run BEFORE the 341 MB copy and before we print "success", because the one failure mode that
    matters is silent. Every other loader in the deployed web repo uses
    `load_state_dict(..., strict=False)`, which on a key-name mismatch loads NOTHING, raises
    NOTHING, and leaves a randomly-initialized network that still produces brain-shaped noise.
    We use strict=True (the default) and let it raise.

    Returns a dict of facts for the manifest, including the sha256 of the sorted state_dict KEY
    NAMES. That hash is the cheap cross-environment check: diffusers has historically renamed
    attention parameters (AttentionBlock's query/key/value -> Attention's to_q/to_k/to_v), and
    AttnDownBlock2D/AttnUpBlock2D are exactly the blocks affected. The web backend can recompute
    this hash under ITS diffusers version and compare — turning a silent-noise failure into a
    one-line mismatch report.
    """
    import torch                                        # deferred: --dry-run never reaches Phase C
    from src.model import build_model

    print(f"\n[C] verifying checkpoint: {ckpt_path}")
    size = os.path.getsize(ckpt_path)
    print(f"    file size: {human(size)}  ({size} bytes)")

    # train.py:57-60 does `ema.copy_to(...)` then `torch.save(model.state_dict(), ...)`, so this
    # file is a BARE state_dict — a flat {name: tensor} mapping. There is no ["model"] or
    # ["state_dict"] or ["epoch"] wrapper to unwrap, and if there is one, something is wrong.
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise SystemExit(f"expected a state_dict (a dict), got {type(sd)}")
    wrapper_keys = [k for k in ("model", "state_dict", "ema", "module") if k in sd]
    if wrapper_keys:
        raise SystemExit(f"checkpoint looks WRAPPED (found key {wrapper_keys[0]!r}). train.py "
                         f"saves a bare state_dict; unwrap it before exporting.")

    model = build_model()                               # the exact architecture; see src/model.py
    model.load_state_dict(sd)                           # strict=True is the DEFAULT — let it raise
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    key_hash = hashlib.sha256(",".join(sorted(sd.keys())).encode()).hexdigest()

    print(f"    loaded into build_model() with strict=True - OK")
    print(f"    parameters: {n_params:,}  (paper reports ~85.3M)")
    print(f"    state_dict entries: {len(sd)}   key-name hash: {key_hash[:16]}...")
    if not (84e6 < n_params < 87e6):
        print(f"    WARNING: parameter count {n_params:,} is outside the expected ~85.3M - the "
              f"architecture in src/model.py may have been edited.")

    import diffusers
    return dict(filename=os.path.basename(ckpt_path), bytes=size, param_count=n_params,
                n_state_dict_entries=len(sd), state_dict_key_sha256=key_hash,
                loaded_strict=True, diffusers_version=diffusers.__version__,
                torch_version=torch.__version__)


# =============================================================================================
# main
# =============================================================================================

def main():
    ap = argparse.ArgumentParser(description="Export the website's conditioning bundle.")
    ap.add_argument("--out", default="web_bundle", help="output folder (default: web_bundle)")
    ap.add_argument("--slices", default="data/processed/slices/val",
                    help="source slices. MUST be val or test - never train (see below)")
    ap.add_argument("--ckpt", default="models/diffusion_ema.pt", help="checkpoint to bundle")
    ap.add_argument("--per-cell", type=int, default=4,
                    help="distinct patients per (lobe, level) cell (default 4)")
    ap.add_argument("--min-brain", type=int, default=MIN_BRAIN,
                    help=f"override MIN_BRAIN (default {MIN_BRAIN}); the escape hatch for an "
                         f"empty inferior cell")
    ap.add_argument("--stride", type=int, default=1,
                    help="scan every Nth slice - use ~10 with --dry-run for a fast preview")
    ap.add_argument("--keep-flair", action="store_true",
                    help="ship all 9 channels including the REAL patient FLAIR (default: 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan, select and report; write nothing and never open the checkpoint. "
                         "NOTE: this still needs the brainmri venv active - importing "
                         "augment_tumor for the thresholds pulls in torch/diffusers/scipy at the "
                         "top of this file, long before argparse runs")
    ap.add_argument("--require-full-coverage", action="store_true",
                    help="exit non-zero if any of the 18 (lobe, level) cells is empty")
    args = ap.parse_args()

    # --per-cell 0 would leave every cell empty, `selected` empty, and the run would die 100
    # lines later with a bare IndexError on selected[0]. Reject it here, where the message is
    # about the flag the user actually typed.
    if args.per_cell < 1:
        raise SystemExit(f"--per-cell must be >= 1 (got {args.per_cell})")
    if args.stride < 1:
        raise SystemExit(f"--stride must be >= 1 (got {args.stride})")

    started = datetime.now(timezone.utc)
    print("=" * 92)
    print("export_for_website.py - conditioning bundle for braingen_CondDiffuser_BraTS_v1 (2D)")
    print("=" * 92)
    print(f"  slices     : {args.slices}")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  out        : {args.out}")
    print(f"  thresholds : PROB_TH={PROB_TH}  MIN_BRAIN={args.min_brain}"
          f"{' (OVERRIDDEN)' if args.min_brain != MIN_BRAIN else ''}  MIN_LOBE={MIN_LOBE}")
    print(f"  z bands    : inferior<{INFERIOR_MAX} <= middle <= {SUPERIOR_MIN}<superior   "
          f"[{_LEVEL_SRC}]")

    # ---- preflight -------------------------------------------------------------------------
    # The bank MUST come from held-out patients. 03_preprocess.py:15-19 splits (0.7,0.15,0.15)
    # BY PATIENT with random.seed(42), so val/ and test/ contain brains the model never saw.
    # Shipping train-split anatomy to a public site means the site displays scans the model
    # memorized, which quietly undercuts the paper's held-out claim.
    split_name = os.path.basename(os.path.normpath(args.slices))
    if split_name == "train":
        raise SystemExit("REFUSING to build a bundle from the TRAIN split. Use val/ or test/ - "
                         "the website must only ever show held-out anatomy.")
    if not os.path.isdir(args.slices):
        raise SystemExit(f"missing slices dir: {args.slices}  (run this on Great Lakes)")
    if not args.dry_run and not os.path.isfile(args.ckpt):
        raise SystemExit(f"missing checkpoint: {args.ckpt}  (it lives on Great Lakes only)")

    # Verify the weights FIRST. A 15-minute scan followed by "your checkpoint doesn't load" is a
    # waste of everyone's afternoon; fail fast on the cheap check.
    ckpt_info = None
    if not args.dry_run:
        ckpt_info = verify_checkpoint(args.ckpt)

    # ---- PHASE A: scan ----------------------------------------------------------------------
    print(f"\n[A] scanning {args.slices}" + (f" (stride {args.stride})" if args.stride > 1 else ""))
    cands, scan_stats = scan_slices(args.slices, args.stride, args.min_brain)
    print(f"    scanned {scan_stats['scanned']} slices")
    # These two must be printed, not just buried in the manifest: they are the only signal that
    # the inputs themselves are wrong rather than merely unsuitable, and a --dry-run never writes
    # a manifest. Anything but 0 here needs investigating before you trust the coverage table.
    print(f"      SKIPPED   {scan_stats['unparsed']:>6}  filename did not match <patient>_z###.npy"
          + ("   <-- INVESTIGATE" if scan_stats["unparsed"] else ""))
    print(f"      SKIPPED   {scan_stats['bad_shape']:>6}  array was not (9,256,256)"
          + ("   <-- INVESTIGATE: channel contract changed" if scan_stats["bad_shape"] else ""))
    print(f"      rejected  {scan_stats['failed_brain']:>6}  too little brain (<= {args.min_brain} px)")
    print(f"      rejected  {scan_stats['failed_tumour']:>6}  already has a real tumour (>= {MAX_TUMOUR_PX} px)")
    print(f"      rejected  {scan_stats['failed_all_lobes']:>6}  no lobe above {MIN_LOBE} px")
    print(f"      admitted  {scan_stats['admitted']:>6}  usable conditioning slices")
    if not cands:
        raise SystemExit("no admissible slices at all - loosen --min-brain, or check that the "
                         "atlas channels are non-zero in these files.")
    print(f"    from {len({c['patient'] for c in cands})} distinct patients")

    # ---- PHASE B: select --------------------------------------------------------------------
    print(f"\n[B] selecting up to {args.per_cell} distinct patients per (lobe, level) cell")
    selected, coverage, cell_members = select_slices(cands, args.per_cell)
    empty_cells = print_coverage(coverage, args.per_cell)
    supported = [f"{lo}_{lv}" for lo in LOBES for lv in LEVELS if coverage[lo][lv] > 0]
    print(f"\n    shipping {len(selected)} distinct slices "
          f"from {len({c['patient'] for c in selected})} patients "
          f"(one slice usually serves several cells, which is why this is well under "
          f"{18 * args.per_cell})")
    if empty_cells and args.require_full_coverage:
        raise SystemExit(f"--require-full-coverage: {len(empty_cells)} empty cells, aborting.")

    if args.dry_run:
        print("\n--dry-run: nothing written. Re-run without --dry-run (and without --stride) "
              "to build the bundle.")
        return

    # ---- write the slices --------------------------------------------------------------------
    # CHANNEL LAYOUT DECISION (this is the one the web backend must agree with):
    #   default   -> stack[1:9], 8 channels: [T1, mask, atlas x6]
    #   --keep-flair -> stack[0:9], 9 channels: [FLAIR, T1, mask, atlas x6]
    # The 8-channel form is EXACTLY `cond` as src/dataset.py:40 builds it, so the backend can do
    # torch.from_numpy(arr).unsqueeze(0) and feed it straight in. Note the off-by-one this
    # creates and that it is the single most error-prone fact in this codebase: inside an
    # 8-channel array, index 0 = T1, index 1 = MASK, index 2+k = atlas lobe k. That is why
    # sample.py:29 and eval_grid.py:51 read the mask as cond[1] and not cond[2].
    ch_lo = 0 if args.keep_flair else 1
    ch_names = (["flair"] if args.keep_flair else []) + ["t1", "mask"] + [f"atlas_{l}" for l in LOBES]
    n_ch = 9 - ch_lo

    slice_dir = os.path.join(args.out, SLICE_SUBDIR)
    os.makedirs(slice_dir, exist_ok=True)

    # np.save does not remove a previous run's .npy files, and they are not in the manifest this
    # run writes. The backend would ignore them (it indexes the manifest, never the directory) --
    # but sftp would still carry them to a public web host, so they would be real held-out patient
    # anatomy sitting on that host with nothing recording that it is there. That is exactly the
    # exposure the 8-channel default exists to limit, so say it out loud rather than assume the
    # backend's indifference makes it harmless.
    stale = [f for f in os.listdir(slice_dir)
             if f.endswith(".npy") and f not in {c["file"] for c in selected}]
    if stale:
        print(f"    WARNING: {len(stale)} .npy file(s) already in {slice_dir}/ are NOT part of "
              f"this export and are NOT in the manifest. Delete them before transferring or you "
              f"will ship unrecorded patient slices. e.g. {', '.join(sorted(stale)[:3])}")

    # --stride subsamples the SCAN, so the coverage table was measured on a fraction of the split.
    # A bundle built that way can report a cell as empty when a full scan would have filled it,
    # and picks a worse slice per cell than a full scan would. Fine to preview, not to ship.
    if args.stride > 1:
        print(f"    WARNING: built from a --stride {args.stride} SUBSAMPLE (~1/{args.stride} of "
              f"the split). Re-run without --stride before transferring this bundle.")

    print(f"\n[D] writing {len(selected)} slices to {slice_dir}/  ({n_ch} channels, float16)")

    # DTYPE JUSTIFICATION — float16, and this is not a lossy compromise:
    #   * 03_preprocess.py:88 ALREADY stores these arrays as float16. Writing float16 is therefore
    #     a bit-exact copy of the values the model was trained on. There is nothing to lose.
    #   * float32 would double the file to 2.25 MiB/slice and add exactly zero information — the
    #     extra mantissa bits would all be zeros re-derived from the float16 source.
    #   * The cast back to float32 at LOAD time is mandatory and must be the very first thing the
    #     backend does: `np.load(p).astype(np.float32)`. The U-Net is fp32, and (worse) assigning
    #     a float32 synthetic mask INTO a float16 array silently downcasts it. That is recorded in
    #     the manifest under channels.loader so nobody has to remember it.
    # Per-slice size: 8 x 256 x 256 x 2 bytes = 1,048,576 B = exactly 1.00 MiB (+128 B npy header).
    records = []
    for c in selected:
        stack = np.load(c["path"])                      # (9,256,256) float16 straight off disk
        out_arr = np.ascontiguousarray(stack[ch_lo:9])  # (8,256,256) or (9,256,256), still float16
        assert out_arr.dtype == np.float16 and out_arr.shape == (n_ch, 256, 256), \
            f"unexpected array {out_arr.dtype} {out_arr.shape} from {c['path']}"
        np.save(os.path.join(slice_dir, c["file"]), out_arr)
        records.append(dict(file=c["file"], patient=c["patient"], z=c["z"], level=c["level"],
                            brain_px=c["brain_px"], tumour_px=c["tumour_px"], lobes=c["lobes"]))
    per_slice = os.path.getsize(os.path.join(slice_dir, selected[0]["file"]))
    print(f"    {human(per_slice)} per slice  ->  {human(per_slice * len(selected))} total")

    # ---- copy + re-verify the checkpoint -----------------------------------------------------
    print(f"\n[E] copying checkpoint into the bundle")
    dst = os.path.join(args.out, CKPT_NAME)
    src_hash = sha256_of(args.ckpt)
    shutil.copy2(args.ckpt, dst)                        # copy2 preserves mtime, which is nice for provenance
    dst_hash = sha256_of(dst)
    # A 341 MB copy across cluster filesystems can silently truncate on a full quota. Comparing
    # hashes costs two disk reads and turns "the website generates noise" into "the copy failed".
    if src_hash != dst_hash:
        raise SystemExit(f"COPY CORRUPTED: {args.ckpt} sha256 {src_hash[:16]} != "
                         f"{dst} sha256 {dst_hash[:16]}")
    print(f"    {human(os.path.getsize(dst))}  sha256 {dst_hash[:16]}...  (verified byte-identical)")
    ckpt_info["sha256"] = dst_hash

    # ---- PHASE F: the manifest ---------------------------------------------------------------
    # This is what the backend reads on every request. It is a few KB, so it can be parsed once at
    # startup and held in memory — the backend must NEVER open 70 .npy files to answer one click.
    commit, dirty = git_state()
    import torch, diffusers, scipy
    manifest = jsonable(dict(
        bundle_version=1,
        model_name="braingen_CondDiffuser_BraTS_v1 (2D)",
        created_utc=started.isoformat(timespec="seconds"),
        generator="scripts/export_for_website.py",
        git=dict(commit=commit, dirty=dirty),
        versions=dict(python=sys.version.split()[0], numpy=np.__version__,
                      torch=torch.__version__, diffusers=diffusers.__version__,
                      scipy=scipy.__version__),
        source=dict(slices_dir=args.slices, split=split_name,
                    note="held-out split; 03_preprocess.py splits (0.7,0.15,0.15) BY PATIENT "
                         "with random.seed(42)"),
        # Everything the backend needs to interpret one .npy without guessing.
        channels=dict(count=n_ch, order=ch_names, dtype="float16", shape=[n_ch, 256, 256],
                      note=("stack[1:9] of 03_preprocess.py's 9-channel layout; channel 0, the "
                            "REAL patient FLAIR, is deliberately NOT shipped"
                            if not args.keep_flair else
                            "the full 9-channel stack INCLUDING the real patient FLAIR at index 0"),
                      index_note=("inside this array: 0=T1, 1=tumour mask, 2+k=atlas lobe k "
                                  "(LOBES order). The mask is at index 1, NOT 2."
                                  if not args.keep_flair else
                                  "inside this array: 0=FLAIR, 1=T1, 2=mask, 3+k=atlas lobe k"),
                      loader="np.load(path).astype(np.float32)   # the .astype is MANDATORY",
                      value_ranges=dict(t1="[0,1]", mask="raw BraTS labels {0,1,2,3} - NOT "
                                                          "normalized, never rescale it",
                                        atlas="[0,1] under ONE global scalar across all 6 lobes")),
        # The selection rules, so a future reader can reproduce this exact bank.
        selection=dict(prob_th=PROB_TH, min_brain=args.min_brain, min_brain_default=MIN_BRAIN,
                       min_lobe=MIN_LOBE, brain_th=BRAIN_TH, max_tumour_px=MAX_TUMOUR_PX,
                       inferior_max=INFERIOR_MAX, superior_min=SUPERIOR_MIN,
                       per_cell=args.per_cell, stride=args.stride, level_source=_LEVEL_SRC,
                       scan=scan_stats),
        # The SHAPE parameters of conddiff_core.synth_mask(). Unused by this script; shipped so the
        # backend can assert ITS copy of synth_mask matches the cluster's before it draws anything.
        #
        # `source` names src/conddiff_core.py, NOT augment_tumor.py. augment_tumor.py still holds
        # the original scipy implementation and is still the scientific reference, but it is no
        # longer what runs: the renderer and the backend both call conddiff_core.synth_mask, and
        # this script now imports its constants from there too. Recording the wrong module here
        # would send a future reader to a file that merely LOOKS the same -- scripts/
        # verify_conddiff_port.py exists precisely because the two are only equal by proof, not by
        # construction.
        #
        # Note labels_emitted: synth_mask writes RAW BraTS labels into the one channel that is NOT
        # in [0,1], and it never emits 1 (necrotic) -- so a real BraTS mask and a synthetic one are
        # not interchangeable, and neither may ever be rescaled.
        mask_synthesis=dict(source="src/conddiff_core.py:synth_mask",
                            reference_impl="scripts/augment_tumor.py:synth_mask (scipy original)",
                            area_frac=list(AREA_FRAC), core_frac=CORE_FRAC, min_area=MIN_AREA,
                            compact=COMPACT, noise_sigma=NOISE_SIGMA, noise_amp=NOISE_AMP,
                            labels_emitted=[0, 2, 3],
                            labels_note="2=edema, 3=enhancing core; label 1 (necrotic) is never "
                                        "synthesized, unlike a real BraTS mask",
                            # synth_mask draws n_les = uniform(*area_frac) * n_lobe and returns
                            # (None, None) if n_les < min_area. So a lobe needs at least
                            # ceil(min_area / area_frac[0]) px or the draw can fail. Under the
                            # paper's (0.15, 0.35) that floor is 400 px, comfortably below
                            # MIN_LOBE = 500 -- so every cell in this bundle is safe AS SHIPPED.
                            # If the website narrows area_frac for a "Small" control (e.g. 0.05)
                            # the floor jumps to 1200 px and MIN_LOBE guarantees nothing: the
                            # backend must then filter on slices[].lobes[lobe] itself, or
                            # synth_mask returns None and the user gets a blank image.
                            # np.ceil, not the -(-a//b) integer trick: 0.15 is not exactly
                            # representable, so -(-60 // 0.15) floors to -401 and answers 401.
                            min_lobe_px_for_default_area_frac=int(np.ceil(MIN_AREA / AREA_FRAC[0])),
                            min_lobe_formula="ceil(min_area / area_frac[0])"),
        lobes=LOBES, levels=LEVELS,
        # The website's dropdowns should be driven by THESE two lists, not by a hardcoded 6x3.
        coverage=coverage, supported_cells=supported, empty_cells=empty_cells,
        # cell -> filenames. One lookup, no file I/O, to answer "which slice for frontal+middle?"
        cells=cell_members,
        checkpoint=ckpt_info,
        slices=records,
    ))
    with open(os.path.join(args.out, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[F] wrote {os.path.join(args.out, MANIFEST)}  "
          f"({human(os.path.getsize(os.path.join(args.out, MANIFEST)))}, "
          f"{len(records)} slice records, {len(supported)}/18 cells supported)")

    # ---- PHASE G: size + transfer checklist ---------------------------------------------------
    total = dir_size(args.out)
    slices_bytes = dir_size(slice_dir)
    print("\n" + "=" * 92)
    print("BUNDLE COMPLETE")
    print("=" * 92)
    print(f"  {args.out}/")
    print(f"    {CKPT_NAME:<22} {human(ckpt_info['bytes']):>10}   the trained EMA weights")
    print(f"    {SLICE_SUBDIR + '/':<22} {human(slices_bytes):>10}   {len(records)} conditioning slices")
    print(f"    {MANIFEST:<22} {human(os.path.getsize(os.path.join(args.out, MANIFEST))):>10}   the index the backend reads")
    print(f"    {'TOTAL':<22} {human(total):>10}")

    remote = os.path.abspath(args.out).replace("\\", "/")
    print(f"""
TRANSFER CHECKLIST  (sftp, NOT scp)
  Use sftp. scp on Great Lakes is unreliable because the U-M login banner is printed into the
  channel scp uses for its own protocol, which corrupts the transfer. sftp negotiates a proper
  subsystem after login, so the banner is harmless.

  1. On the LAPTOP, protect the shared repo BEFORE you drop anything into it. The lab's
     .gitignore only covers backend/inference/model/*.safetensors - .pt and .npy are NOT
     ignored, and a `git add -A` would try to stage a {human(ckpt_info['bytes'])} file into
     github.com/SOCR/Brain-Image-Generator, which GitHub rejects at 100 MB per file and which
     leaves a broken local history behind:

       cd C:/Users/alexl/code/Research/socr/Brain-Image-Generator
       printf 'backend/inference/conddiff_bundle/\\nbackend/inference/model/*.pt\\nbackend/inference/model/*.pth\\n' >> .git/info/exclude

     (.git/info/exclude is LOCAL-ONLY - it changes no tracked file, so it cannot be committed
     to the shared repo by accident.)

  2. Pull the bundle down. Fetch the FOLDER (not its contents): `get -r .` is not a portable
     sftp idiom, and pulling the contents into backend/inference/ would scatter manifest.json,
     {CKPT_NAME} and {SLICE_SUBDIR}/ next to the .py files instead of inside the bundle folder
     the backend looks for in step 3.

       sftp <uniqname>@greatlakes.arc-ts.umich.edu
       sftp> lcd C:/Users/alexl/code/Research/socr/Brain-Image-Generator/backend/inference
       sftp> cd {os.path.dirname(remote)}
       sftp> get -r {os.path.basename(remote)}
       sftp> bye

     That leaves backend/inference/{os.path.basename(remote)}/. Rename it to the name the
     backend expects (this is the directory .git/info/exclude was told about in step 1):

       cd C:/Users/alexl/code/Research/socr/Brain-Image-Generator/backend/inference
       mv {os.path.basename(remote)} conddiff_bundle

  3. Put it where the backend expects it and confirm git still sees nothing:

       backend/inference/conddiff_bundle/{MANIFEST}
       backend/inference/conddiff_bundle/{CKPT_NAME}
       backend/inference/conddiff_bundle/{SLICE_SUBDIR}/*.npy

       git status --porcelain        # must print NOTHING about the bundle

  4. Sanity-check the transfer on the laptop (sizes must match this run exactly):

       python -c "import json;m=json.load(open('backend/inference/conddiff_bundle/{MANIFEST}'));print(len(m['slices']),'slices',len(m['supported_cells']),'cells')"
       python -c "import glob;print(len(glob.glob('backend/inference/conddiff_bundle/{SLICE_SUBDIR}/*.npy')),'npy files actually on disk')"
       # BOTH counts must read {len(records)}. The first only reads the manifest against itself,
       # so on its own it cannot notice a .npy that sftp silently dropped.
       python -c "import hashlib;h=hashlib.sha256();f=open('backend/inference/conddiff_bundle/{CKPT_NAME}','rb');[h.update(b) for b in iter(lambda:f.read(1<<20),b'')];print(h.hexdigest())"
       # must equal: {dst_hash}
""")
    if empty_cells:
        print(f"  REMINDER: {len(empty_cells)} dropdown combinations are unsupported "
              f"({', '.join(empty_cells)}).\n"
              f"  Read them from manifest.json['empty_cells'] and DISABLE them in the UI - do not\n"
              f"  silently substitute a different slice level, or the site will label an image\n"
              f"  with a location it did not generate.\n")


if __name__ == "__main__":
    main()
