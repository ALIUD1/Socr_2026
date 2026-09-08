#!/usr/bin/env python3
"""render_gallery.py -- pre-render the FINITE gallery of images the website serves.

WHY THIS SCRIPT EXISTS (the arithmetic that forced it)
=====================================================
The deployed web backend runs on AWS Lightsail at "power": "micro" = 0.25 vCPU / 1 GB RAM,
with a health check of timeoutSeconds 5 / intervalSeconds 30 / unhealthyThreshold 2 -- i.e.
the container is torn down and replaced after roughly 65 s of not answering /health. The
model in this repo (braingen_CondDiffuser_BraTS_v1) is a 85.3M-parameter U-Net doing about
950 GFLOP per forward pass at 256x256, and a DDIM sample is 50-200 of those. The checkpoint
alone is ~341 MB of fp32 weights. And backend/api.py declares its route `async def` while
calling synchronous code, so one long request blocks uvicorn's event loop and /health stops
answering *for everyone*.

Live CPU diffusion there is not slow. It is impossible: the container is killed mid-sample,
every time, forever.

The way out is that the user-visible parameter space is FINITE. The website exposes four
dropdowns, and the conditioning (a real held-out patient's T1 and six atlas lobe maps) comes
from a fixed shipped bank, not from the user. So the only things a visitor can vary are:

    Tumour(2) x Lobe(6) x Slice Location(3) x Tumour Size(3) = 108 nominal combinations

Lobe and Size are meaningless when Tumour = "Without Tumor" -- there is no lesion to place and
nothing to size, and src/conddiff_core.py:params_to_spec returns None for both -- so those 54
nominal combinations collapse into 3 real ones, one per level:

    6*3*3 = 54 with-tumour cells  +  3 without-tumour cells  =  57 cells

At K seed variants per cell that is 57*K images of 256x256 grayscale PNG, a few megabytes in
total. Render them ONCE here, on the RTX 6000, at the paper's full 200 DDIM steps, ship the
PNGs, and the backend answers a request with a dict lookup and a file read: no torch, no
diffusers, no 341 MB checkpoint in a 1 GB container, no multi-minute request holding the event
loop. And the images are BETTER than live CPU could ever produce, because live CPU would have
had to drop to ~50 steps to have any hope at all.

THE CORRECTNESS REQUIREMENT THAT SHAPES THIS FILE
=================================================
A pre-rendered gallery is only honest if THIS script and the live path compute the same
picture from the same four choices. If the cluster used scipy's gaussian_filter while the
server used a numpy reimplementation, or if the two disagreed about PROB_TH by 0.05, then the
site would display "Temporal, Large" over an image that the live path would have drawn
somewhere else, at a different size -- and no exception would ever be raised. That is a silent
correctness failure, and it is the exact failure this whole design exists to prevent.

The defence: this script does not contain any image maths. It imports every bit of it from

    src/conddiff_core.py

which exists BYTE-IDENTICALLY at that path and at

    Brain-Image-Generator/backend/inference/conddiff_core.py

There is deliberately no local copy of synth_mask, no local params_to_spec, no local cell
naming, no local PNG encoder here -- importing the identical code is the entire point. To make
the duplication checkable rather than merely claimed, this script stamps the sha256 of the
core file it actually imported into the gallery manifest (top level AND on every image
record), and the backend prints/compares that hash at startup. A divergence between cluster
and server therefore surfaces as a one-line mismatch at boot instead of as a wrong picture
months later.

WHAT THIS SCRIPT READS
======================
The conditioning bank built by scripts/export_for_website.py:

    <bundle>/manifest.json        the index: which slices, which patient, which z, which lobes
    <bundle>/cond_slices/*.npy    (8,256,256) float16 conditioning  [t1, mask, atlas x6]
    <bundle>/diffusion_ema.pt     the trained EMA weights (bare state_dict)

If that has not been run, this script says so and exits -- it does NOT re-scan the split
itself. Two programs deciding independently which slices are admissible is the same class of
divergence as two copies of synth_mask; export_for_website.py owns that decision, records it,
and this script obeys the record.

WHAT IT WRITES
==============
    <out>/manifest.json                     schema "conddiff_gallery/1"
    <out>/<cell_id>/v00_flair.png           the generated FLAIR
    <out>/<cell_id>/v00_seg.png             the mask we drew (with-tumour cells only)
    ...

NOTE ON THE MANIFEST FILENAME. The workflow brief called this file gallery_manifest.json; it
is written as manifest.json because the backend hard-codes

    GALLERY_MANIFEST = os.path.join(GALLERY_DIR, "manifest.json")

(conddiff_inference.py, SECTION 4) and refuses to start in gallery mode without it. Renaming
it here would produce a bundle the server cannot load, so the server's name wins. Everything
the brief asked that file to contain is in it.

RUN IT ON GREAT LAKES
=====================
It needs a GPU, the checkpoint and the bank -- none of which exist on the laptop.

    python scripts/render_gallery.py --dry-run          # plan + time estimate, NO GPU needed
    sbatch scripts/render_gallery.sbatch                # the real run (see that file)
    python scripts/render_gallery.py --steps 50 -k 1    # a fast smoke test

--dry-run is deliberately importable on a LOGIN NODE: it never imports torch and never opens
the checkpoint, so you can size the SLURM job before you queue for a GPU.
"""
import os
import sys
import json
import time
import glob
import shutil
import hashlib
import argparse
from datetime import datetime, timezone

# Make the repo root importable so `import src.conddiff_core` works no matter which directory
# you launched from (sbatch cds to ~/Summer2026, an interactive test might not). Every script
# in scripts/ opens with this same line.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# =============================================================================================
# THE SHARED CORE -- THE ONLY PLACE IMAGE MATHS MAY COME FROM
#
# Imported as a module (rather than `from ... import *`) so that every call site below reads
# `core.synth_mask_sized(...)`, `core.build_cond(...)`. That is not a style preference: when
# somebody skims this file to check the "no reimplementation" claim, the `core.` prefix makes
# every piece of shared maths visible at a glance, and a locally-defined helper with the same
# name would stand out instead of blending in.
#
# The import is wrapped because the failure it catches is confusing on its own: if
# src/conddiff_core.py is missing, `import src.conddiff_core` raises ModuleNotFoundError naming
# `src.conddiff_core`, which reads like a packaging problem rather than "the file this whole
# design depends on is not there".
# =============================================================================================
try:
    import src.conddiff_core as core
except Exception as _e:                                     # pragma: no cover - setup error only
    raise SystemExit(
        "cannot import src/conddiff_core.py -- this script renders NOTHING without it.\n"
        f"  underlying error: {_e}\n"
        "  conddiff_core.py holds the mask synthesis, the DDIM loop and the PNG encoding that\n"
        "  the web backend also imports (from its own byte-identical copy). Rendering a\n"
        "  gallery with a local reimplementation would defeat the entire point of this script."
    )


# =============================================================================================
# SECTION 1 -- CONSTANTS THAT ARE THIS SCRIPT'S OWN BUSINESS
#
# Everything that changes THE PICTURE lives in conddiff_core.py. What is left here is
# deployment: where files are, how many variants, how many steps, how we name things. Those are
# allowed to differ between the cluster and the server, which is exactly why they are not in
# the shared file (conddiff_core.py rule 3: no paths, no env, no deployment decisions).
# =============================================================================================

# The manifest schema string the backend understands. conddiff_inference.py:
#     GALLERY_SCHEMA = "conddiff_gallery/1"
# and _load_gallery() REFUSES any other value rather than half-reading a layout that changed
# underneath it. If you change the shape of the manifest, bump this AND teach the backend.
GALLERY_SCHEMA = "conddiff_gallery/1"

# The model name string as registered in the web app. It appears identically in
# backend/config.py, backend/image_generation.py, backend/api.py and the frontend's
# ParameterControlPanel.tsx. Recorded here purely as provenance, so an image can be traced to
# the model that made it.
MODEL_NAME = "braingen_CondDiffuser_BraTS_v1 (2D)"

# Names inside the bundle produced by scripts/export_for_website.py. Retyped from that file's
# SLICE_SUBDIR / CKPT_NAME / MANIFEST constants -- this script cannot import them because
# export_for_website.py imports augment_tumor at module scope, which pulls in torch, diffusers,
# scipy and matplotlib. --dry-run must run on a login node without any of that.
BUNDLE_SLICE_SUBDIR = "cond_slices"
BUNDLE_CKPT_NAME    = "diffusion_ema.pt"
BUNDLE_MANIFEST     = "manifest.json"

# The output manifest's name. Fixed by the backend; see the module docstring.
OUT_MANIFEST = "manifest.json"

# The brain/background cut on the percentile-normalised T1. 0.05 is the value used everywhere
# in this repo (augment_tumor.py:128, check_atlas_fit.py:56, export_for_website.py:BRAIN_TH)
# because 03_preprocess.py forces true background to EXACTLY 0, so anything above a hair of
# noise is tissue. It is NOT in conddiff_core because the core takes `brain` as an argument --
# the caller decides what counts as brain, and both callers decide identically.
BRAIN_TH = 0.05

# "This slice already contains a real tumour" gate. We are adding exactly ONE controlled lesion
# to an otherwise-healthy slice; if the slice already has BraTS pathology, the generated image
# contains that lesion PLUS ours and the Lobe control is no longer the only thing determining
# where pathology appears. Same value as export_for_website.py:MAX_TUMOUR_PX and the backend's
# _slice_ok.
MAX_TUMOUR_PX = 50

# Fallback if the bundle manifest does not record the MIN_BRAIN it was built with. MIN_BRAIN is
# the one threshold the cluster and the server are ALLOWED to disagree on (it is cosmetic --
# "is this a real mid-axial slice or a tiny sliver" -- and it is what has to be lowered to fill
# the cerebellum cells), which is precisely why it is not in the shared core.
MIN_BRAIN_FALLBACK = 15000

# A-priori throughput used for the --dry-run estimate, before any GPU exists to measure.
# Derived from the model's ~950 GFLOP per 256x256 forward pass against the Quadro RTX 6000's
# ~16 TFLOP/s fp32 peak at a realistic ~35% utilisation for this shape. It is a GUESS and is
# labelled as one everywhere it is printed; the real run replaces it with a measurement from
# an actual timed forward pass before it commits to the full loop.
SEC_PER_FORWARD_GUESS = 0.18

# How many forward passes to time for the on-GPU calibration. Small enough to be free
# (~2 s at the guess above), large enough that the per-step average is not dominated by one
# scheduling hiccup.
CALIBRATION_FORWARDS = 12


# =============================================================================================
# SECTION 2 -- SMALL HELPERS (formatting, hashing, sizing)
# =============================================================================================

def human(nbytes):
    """Bytes -> a human string like '11.8 MB'. Cosmetic only; the manifest stores raw ints."""
    n = float(nbytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return "%d B" % int(n) if unit == "B" else "%.1f %s" % (n, unit)
        n /= 1024.0


def hms(seconds):
    """Seconds -> 'H:MM:SS'. Used for both the estimate and the final wall clock, so that the
    two are trivially comparable at the end of the log -- if the estimate was badly wrong you
    want to see that immediately, not have to convert units in your head."""
    seconds = int(max(0, round(seconds)))
    return "%d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def sha256_of(path, chunk=1 << 20):
    """Streaming SHA-256 of a file, 1 MiB at a time.

    Streaming rather than read()-it-all because this is also pointed at the 341 MB checkpoint,
    and there is no reason to pull that into RAM to hash it.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dir_size(path):
    """Total bytes of every file under `path`, recursively. os.walk, so per-cell subdirs count."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return total


def jsonable(obj):
    """Recursively convert numpy scalars/arrays into plain Python types.

    NOT paranoia. Every count in the stats dicts below is a numpy scalar (`int(...)` in
    conddiff_core covers most of them, but `coverage` is a rounded float and `size_in_band` is
    a numpy bool in some numpy versions), and json.dump raises
        TypeError: Object of type int64 is not JSON serializable
    on the first one it meets -- AFTER the images have been rendered, which on a 2-hour job is
    an expensive way to find out. Worse, the same values are later json.dumps'd by the web
    backend into Supabase's parameters_used column inside a try/except that only prints, so a
    stray np.int64 there degrades to a missing database record. Coerce once, at the boundary.
    export_for_website.py has the identical helper for the identical reason.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj


def git_state():
    """Which commit produced this gallery. Wrapped in try/except because the render must still
    work from a tarball with no .git directory (and Slurm jobs frequently run from one)."""
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                             stderr=subprocess.DEVNULL).decode().strip())
        return commit, dirty
    except Exception:
        return None, None


def atomic_write_json(path, payload):
    """Write JSON via a temp file + os.replace, so the manifest is never half-written.

    THIS MATTERS BECAUSE OF RESUME. The manifest is rewritten after EVERY cell precisely so a
    job killed by the SLURM wall clock leaves a usable one behind. If the kill landed in the
    middle of a plain `json.dump(fh)` the file would be truncated JSON, the next run's resume
    would fail to parse it, and the run would either crash or silently start from zero and
    re-render two hours of images. os.replace is atomic within a filesystem on both POSIX and
    Windows, so the file on disk is always either the previous complete manifest or the new
    complete one.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


# =============================================================================================
# SECTION 3 -- READING THE CONDITIONING BANK
#
# The bank is scripts/export_for_website.py's output. This script is a CONSUMER of it: it does
# not re-scan data/processed/slices/ and it does not re-decide admissibility. Two programs
# independently deciding which slices are usable is the same class of silent divergence as two
# copies of synth_mask, so export_for_website.py decides, records the decision in manifest.json
# (including the exact thresholds it used), and this script obeys the record -- while still
# re-verifying every array it loads, because a manifest is a hint and the array is the truth.
# =============================================================================================

def load_bundle(bundle_dir):
    """Open the export bundle and return everything this script needs from it.

    Returns a dict:
        dir          str   the bundle root
        manifest     dict  the parsed manifest.json
        slice_dir    str   <bundle>/cond_slices
        ckpt         str   <bundle>/diffusion_ema.pt
        by_file      dict  filename -> slice record {file, patient, z, level, brain_px,
                                                     tumour_px, lobes: {lobe: px}}
        cells        dict  "<lobe>_<level>" -> [filename, ...]   (export's Phase B selection)
        by_level     dict  level -> [slice record, ...]          (for the no-tumour cells)
        min_brain    int   the MIN_BRAIN the bank was actually built with
        ch_lo        int   0 if the .npy files carry 9 channels, 1 if they carry 8
        empty_cells  list  (lobe, level) combinations the bank could not fill

    Every failure here exits with an instruction, not a traceback: the overwhelmingly likely
    reason any of this is missing is "export_for_website.py has not been run yet", and that is
    a two-command fix the message should just say.
    """
    man_path = os.path.join(bundle_dir, BUNDLE_MANIFEST)
    if not os.path.isfile(man_path):
        raise SystemExit(
            "no conditioning bank found at %s\n"
            "  Expected %s -- the index scripts/export_for_website.py writes.\n"
            "\n"
            "  BUILD THE BANK FIRST (on Great Lakes; a login node is enough, no GPU needed):\n"
            "      module load python && source ~/envs/brainmri/bin/activate\n"
            "      cd ~/Summer2026\n"
            "      python scripts/export_for_website.py --out %s\n"
            "\n"
            "  That scans data/processed/slices/val/, picks the held-out slices the gallery is\n"
            "  conditioned on, copies the checkpoint in, and writes the manifest this script\n"
            "  reads. Then re-run this script.\n"
            "  (If you built the bank somewhere else, point at it with --bundle <dir>.)"
            % (bundle_dir, man_path, bundle_dir)
        )

    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)

    slice_dir = os.path.join(bundle_dir, BUNDLE_SLICE_SUBDIR)
    if not os.path.isdir(slice_dir):
        raise SystemExit("bank manifest exists but %s/ does not -- the bundle is incomplete. "
                         "Re-run scripts/export_for_website.py." % slice_dir)
    n_npy = len(glob.glob(os.path.join(slice_dir, "*.npy")))
    if n_npy == 0:
        raise SystemExit("%s/ contains no .npy files -- the bundle is incomplete (a partial "
                         "sftp?). Re-run scripts/export_for_website.py." % slice_dir)

    ckpt = os.path.join(bundle_dir, BUNDLE_CKPT_NAME)
    if not os.path.isfile(ckpt):
        # Fall back to the repo's own checkpoint. The bundle copy exists so the WEB repo gets a
        # verified one; for rendering we only need weights that load, and models/ is where they
        # live on the cluster.
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", BUNDLE_CKPT_NAME)
        if os.path.isfile(alt):
            print("  note: %s not in the bundle; falling back to %s" % (BUNDLE_CKPT_NAME, alt))
            ckpt = alt
        else:
            raise SystemExit("no checkpoint at %s and none at %s -- nothing to render with."
                             % (ckpt, alt))

    # ---- channel layout -----------------------------------------------------------------
    # export_for_website.py ships stack[1:9] by default (8 channels: [t1, mask, atlas x6]) and
    # the full 9 with --keep-flair. We must know which, because conddiff_core.build_cond takes
    # a 9-CHANNEL stack and slices stack[1:9] out of it itself. Trust the manifest's declared
    # order, then re-check it against the array's real shape at load time.
    ch = man.get("channels") or {}
    order = [str(s).lower() for s in (ch.get("order") or [])]
    count = int(ch.get("count") or len(order) or 0)
    if count == 8 and order[:2] == ["t1", "mask"]:
        ch_lo = 1                      # a FLAIR channel must be prepended; see to_stack9()
    elif count == 9 and order[:3] == ["flair", "t1", "mask"]:
        ch_lo = 0                      # already a full stack
    else:
        raise SystemExit(
            "the bank manifest declares an unrecognised channel layout: count=%r order=%r.\n"
            "  This script understands exactly the two layouts export_for_website.py writes:\n"
            "    8 channels [t1, mask, atlas x6]           (the default)\n"
            "    9 channels [flair, t1, mask, atlas x6]    (--keep-flair)\n"
            "  Anything else means the channel contract changed and every index in\n"
            "  conddiff_core (mask at cond[1], atlas lobe k at stack[3+k]) is now wrong."
            % (count, order)
        )

    records = man.get("slices") or []
    if not records:
        raise SystemExit("the bank manifest lists no slices. Re-run export_for_website.py.")
    by_file = {r["file"]: r for r in records}

    by_level = {}
    for r in records:
        by_level.setdefault(r.get("level", "middle"), []).append(r)

    sel = man.get("selection") or {}
    min_brain = int(sel.get("min_brain", MIN_BRAIN_FALLBACK))

    return dict(dir=bundle_dir, manifest=man, slice_dir=slice_dir, ckpt=ckpt,
                by_file=by_file, cells=(man.get("cells") or {}), by_level=by_level,
                min_brain=min_brain, ch_lo=ch_lo, n_npy=n_npy,
                empty_cells=(man.get("empty_cells") or []))


def to_stack9(arr, ch_lo):
    """Normalise one bank .npy into the (9,256,256) float32 stack everything downstream expects.

    THE .astype(np.float32) IS MANDATORY AND MUST BE THE FIRST OPERATION. 03_preprocess.py
    stores these arrays as float16 to halve the bank. If you assign a float32 synthetic mask
    INTO a half array, numpy silently DOWNCASTS the mask; later torch.from_numpy hands the fp32
    U-Net a half tensor. Both failures are silent. Cast first, assert, then touch anything.

    THE PREPENDED ZERO CHANNEL, when the bundle shipped 8 channels: channel 0 of the 9-channel
    layout is the REAL patient's FLAIR, which export_for_website.py deliberately does not ship
    (the model never reads it -- src/dataset.py slices `cond = stack[1:9]` -- and it is real
    patient imaging that nobody needs on a public web host). Nothing downstream reads channel 0
    either: conddiff_core.build_cond takes stack[1:9], and the admissibility tests read
    channels 1, 2 and 3..8. So filling it with zeros re-creates the indexing the rest of the
    code is written against without inventing any data. It is padding, not a substitute FLAIR.

    arr    : (8,256,256) or (9,256,256), float16 off disk
    ch_lo  : 1 if the array is the 8-channel form, 0 if it is already 9
    returns (9,256,256) float32
    """
    a = np.asarray(arr).astype(np.float32)                 # cast FIRST, always
    want = 9 - ch_lo
    if a.shape != (want, 256, 256):
        raise ValueError("bank slice has shape %r, expected %r from the manifest's channel "
                         "declaration" % (a.shape, (want, 256, 256)))
    if ch_lo == 1:
        a = np.concatenate([np.zeros((1, 256, 256), np.float32), a], axis=0)
    assert a.shape == (9, 256, 256) and a.dtype == np.float32
    return a


def slice_ok(stack, lobe_idx, min_brain):
    """The three conjunctive admissibility tests, verbatim from augment_tumor.py:130.

    These are the same three the web backend's _slice_ok runs and the same three
    export_for_website.py's Phase A ran when it built the bank. Running them AGAIN here is not
    redundancy for its own sake: the manifest is a hint written at export time, the array is
    the truth right now, and a stale manifest must degrade to "try the next candidate" rather
    than to a picture with a 60-px pinprick in it.

      1. enough brain  -- (stack[1] > 0.05).sum() > min_brain   channel 1 is the T1
      2. tumour-FREE   -- (stack[2] > 0.5).sum() < 50           channel 2 is the raw BraTS seg,
                          the ONE channel not in [0,1], so `> 0.5` catches any label >= 1
      3. lobe present  -- ((atlas > PROB_TH) & brain).sum() > MIN_LOBE for the requested lobe

    stack    : (9,256,256) float32
    lobe_idx : int index into core.LOBES, or None to skip test 3 (the no-tumour cells)
    returns (ok: bool, stats: dict)
    """
    brain = stack[1] > BRAIN_TH                            # (256,256) bool
    brain_px = int(brain.sum())
    tumour_px = int((stack[2] > 0.5).sum())
    lobe_px = 0
    if lobe_idx is not None:
        al = stack[core.ATLAS0 + lobe_idx]                 # (256,256) that lobe's atlas map
        lobe_px = int(((al > core.PROB_TH) & brain).sum())

    ok = (brain_px > min_brain) and (tumour_px < MAX_TUMOUR_PX)
    if lobe_idx is not None:
        ok = ok and (lobe_px > core.MIN_LOBE)
    return ok, {"brain_px": brain_px, "tumour_px": tumour_px, "lobe_px": lobe_px}


def candidates_for(spec, bundle):
    """Which bank slices can serve this cell? Returns a list of slice records.

    WITH a tumour: the export's Phase B already computed exactly this -- manifest["cells"] maps
    "<lobe>_<level>" to the filenames it selected for that combination, best-lobe-first and at
    most one slice per patient. Use it rather than re-deriving it.

    WITHOUT a tumour: there is no lobe constraint, so every slice at the requested level is a
    candidate. (The bank's slices are all tumour-free by construction, which is what makes
    "Without Tumor" simply a zeroed mask channel on the same anatomy.)
    """
    if spec["with_tumour"]:
        key = "%s_%s" % (spec["lobe"], spec["level"])
        files = bundle["cells"].get(key) or []
        return [bundle["by_file"][f] for f in files if f in bundle["by_file"]]
    return list(bundle["by_level"].get(spec["level"], []))


def pick_slice(spec, rng, bundle):
    """Choose one bank slice for this image. Returns (stack, record, stats).

    THE ORDER OF OPERATIONS MIRRORS THE BACKEND'S _pick_slice EXACTLY, and that is deliberate
    rather than incidental:

        candidates -> rng.permutation(len(candidates)) -> load -> re-verify -> first that passes

    Both sides consume the SAME numpy Generator in the SAME order before handing it to
    synth_mask_sized. Since the Generator is seeded from the recorded seed, a live deployment
    holding the same bank reproduces this exact image from the recorded seed -- mask and all.
    Had this script picked its slice by, say, `candidates[v % len(candidates)]`, the Generator
    would be in a different state by the time the mask was drawn and the live path would draw a
    DIFFERENT lesion from the same seed. The image would still look fine. It would just no
    longer be the image the manifest describes.

    (Caveat, stated because it is real: exact replay also requires the two sides to hold the
    same bank in the same order. A gallery deployment ships no bank at all, so in practice the
    manifest's recorded bank_file / bank_patient / bank_z is what makes an image auditable, and
    the seed is what makes it reproducible on a machine that does have the bank.)

    CONSEQUENCE FOR VARIETY: because each variant has its own seed, each gets its own
    permutation, so variants usually land on different patients -- but not always, and two
    variants of a cell may share anatomy. They are still different images (different lesion,
    different noise), and bank_patient in the manifest says exactly which anatomy each used.
    """
    cands = candidates_for(spec, bundle)
    if not cands:
        raise RuntimeError(
            "the conditioning bank has no %s slice%s. export_for_website.py records unfillable "
            "combinations in manifest.json['empty_cells']; this cell is one of them, so it will "
            "be listed as empty in the gallery manifest and the UI must disable it."
            % (spec["level"],
               " with a usable %s lobe" % spec["lobe"] if spec["lobe"] else "")
        )

    order = rng.permutation(len(cands))                    # deterministic given the seed
    last = None
    for i in order.tolist():
        rec = cands[i]
        path = os.path.join(bundle["slice_dir"], rec["file"])
        try:
            stack = to_stack9(np.load(path), bundle["ch_lo"])   # (9,256,256) float32
        except Exception as e:                             # truncated / missing / wrong shape
            print("      bank slice unreadable, skipping: %s: %s" % (rec["file"], e))
            continue
        ok, stats = slice_ok(stack, spec["lobe_idx"], bundle["min_brain"])
        last = stats
        if ok:
            return stack, rec, stats
        print("      manifest/array disagree for %s: %s -- trying next" % (rec["file"], stats))

    raise RuntimeError("every candidate for this cell failed re-verification against the real "
                       "array (last stats: %r). The bank manifest is stale -- rebuild it with "
                       "export_for_website.py." % (last,))


# =============================================================================================
# SECTION 4 -- SEEDS
#
# Every image gets its own seed, the seed is RECORDED, and the same (base, cell, variant)
# always produces the same seed. That last property is what makes resume safe: a relaunched job
# re-renders a missing variant with the seed it would originally have had, so the gallery does
# not become a patchwork of two different random draws depending on where the wall clock fell.
# =============================================================================================

def derive_seed(base_seed, cid, variant):
    """Deterministic per-image seed: sha256(base | cell_id | variant) truncated to 32 bits.

    WHY A HASH RATHER THAN `base + counter`: a counter ties the seed to the ENUMERATION ORDER,
    so inserting one new lobe or reordering iter_cells() would silently re-seed every image
    after the insertion point and the whole gallery would change. Hashing the cell's NAME ties
    the seed to the cell's identity instead, which is the thing that is actually stable.

    WHY 32 BITS: np.random.default_rng and torch.Generator.manual_seed both accept far larger
    integers, but a 64-bit value printed in a log is unreadable and this number goes into the
    UI's provenance panel. 2^32 possibilities across 228 images makes a collision irrelevant
    (and the caller asserts uniqueness anyway).

    WHY hashlib RATHER THAN hash(): Python's built-in hash() of a str is randomised per process
    by PYTHONHASHSEED, so it would give a different gallery on every run. That is precisely the
    bug this function exists to not have.
    """
    key = "%d|%s|%d" % (int(base_seed), cid, int(variant))
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


# =============================================================================================
# SECTION 5 -- THE MODEL
#
# torch and diffusers are imported INSIDE these functions, never at module scope, so --dry-run
# works on a login node with no GPU and no CUDA-linked torch to page in. Same discipline as
# conddiff_core.py, for a different reason (there it is a 1 GB container; here it is a login
# node and a queue wait).
# =============================================================================================

def build_and_load(ckpt_path, device):
    """Build the paper's U-Net, load the EMA weights, and refuse anything that does not fit.

    Returns (model, facts_dict).

    THREE GUARDS, EACH FOR A FAILURE THAT IS OTHERWISE SILENT:

    1. `strict=True` (the default, spelled out here for emphasis). Every loader in the deployed
       web repo uses strict=False, which on a key-name mismatch loads NOTHING, raises NOTHING,
       and leaves a randomly-initialised network that still produces plausible brain-shaped
       output. Rendering 228 images from random weights and only noticing on the website is the
       nightmare this line prevents.
    2. The checkpoint must be a BARE state_dict. train.py does `ema.copy_to(model)` then
       `torch.save(model.state_dict(), ...)`, so there is no {"model": ...} or {"epoch": ...}
       wrapper. If one appears, the file is not what we think it is.
    3. Parameter count within 1% of the paper's 85.3M. This is the same assertion the backend's
       _get_model makes. It catches "somebody edited src/model.py's block_out_channels", which
       strict=True alone would NOT catch if the edit changed only a width that still produces
       matching key names... it would, in fact, raise on shape -- but the count also documents
       the architecture in the log, and the log is what you read six months later.
    """
    import torch
    from src.model import build_model

    print("\n[2] loading the checkpoint")
    size = os.path.getsize(ckpt_path)
    print("    %s  (%s)" % (ckpt_path, human(size)))

    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise SystemExit("expected a bare state_dict (a dict), got %s" % type(sd))
    wrapper = [k for k in ("model", "state_dict", "ema", "module") if k in sd]
    if wrapper:
        raise SystemExit("checkpoint looks WRAPPED (found key %r). train.py saves a bare "
                         "state_dict; unwrap it before rendering." % wrapper[0])

    model = build_model()                                  # THE architecture; src/model.py
    model.load_state_dict(sd)                              # strict=True is the DEFAULT
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if not (0.99 * 85.3e6 <= n_params <= 1.01 * 85.3e6):
        raise SystemExit("parameter count %s is not within 1%% of the paper's 85.3M -- "
                         "src/model.py:build_model() has drifted and this checkpoint is not "
                         "the model the website advertises." % format(n_params, ","))

    key_hash = hashlib.sha256(",".join(sorted(sd.keys())).encode()).hexdigest()
    print("    loaded with strict=True on %s" % device)
    print("    parameters: %s  (%.1fM, paper reports ~85.3M)" % (format(n_params, ","),
                                                                n_params / 1e6))
    print("    state_dict entries: %d   key-name hash: %s..." % (len(sd), key_hash[:16]))

    import diffusers
    facts = dict(filename=os.path.basename(ckpt_path), bytes=size, param_count=n_params,
                 n_state_dict_entries=len(sd), state_dict_key_sha256=key_hash,
                 sha256=sha256_of(ckpt_path), loaded_strict=True,
                 torch_version=torch.__version__, diffusers_version=diffusers.__version__)
    return model, facts


def build_scheduler(steps):
    """The DDIM scheduler, built EXACTLY as sample.py:21 and augment_tumor.py:121 build it.

    EVERY UNSPECIFIED ARGUMENT IS A LOAD-BEARING DEFAULT. train.py defines the forward process
    as `betas = torch.linspace(1e-4, 0.02, 1000)`, which is byte-for-byte diffusers' defaults
    beta_start=1e-4, beta_end=0.02, beta_schedule="linear", num_train_timesteps=1000. And
    train.py's loss is `F.mse_loss(pred, noise)`, i.e. the network predicts EPSILON, which
    fixes prediction_type="epsilon" (also the default). eta defaults to 0 -> deterministic DDIM.

    Do NOT pass beta_schedule="scaled_linear" (Stable Diffusion's schedule, not ours) and do
    NOT pass prediction_type="v_prediction". Either produces noise instead of brains, silently.

    The web backend's _get_scheduler builds the identical object; only `steps` differs, and
    that difference is recorded per image as ddim_steps so it can never be silent.
    """
    from diffusers import DDIMScheduler
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(steps)                             # timesteps: descending int64 tensor
    return sched


def calibrate(model, device, n=CALIBRATION_FORWARDS):
    """Time `n` real forward passes and return seconds-per-forward.

    WHY MEASURE AT ALL when we already print an a-priori estimate: the a-priori number is a
    FLOP-count guess and can be off by 2x depending on the node, the driver, whether TF32 is
    enabled and whether the GPU is shared. The user is about to commit hours of wall clock; a
    measurement taken on the actual allocated GPU, before the loop starts, is the difference
    between "size the job correctly" and "find out at hour 8".

    THE WARM-UP PASS IS NOT OPTIONAL. The first CUDA forward pays for lazy context creation,
    kernel autotuning and memory-pool growth; including it would inflate the average by enough
    to make the estimate useless.

    torch.cuda.synchronize() before stopping the clock is equally non-optional: CUDA launches
    are ASYNCHRONOUS, so without it we would be timing how fast Python can enqueue work, which
    is roughly instant and would predict a 3-minute job.

    Uses a zero-filled conditioning tensor: the runtime of a convolution does not depend on the
    values in it, only on the shapes, and fabricating a real one here would mean loading a bank
    slice before the plan is even printed.
    """
    import torch
    x = torch.zeros(1, 9, 256, 256, device=device)         # (1,9,256,256) = [noisy flair | cond]
    t = torch.tensor(500, device=device)                   # any mid-schedule timestep
    with torch.no_grad():
        model(x, t)                                        # WARM-UP -- deliberately untimed
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x, t)
        if device.startswith("cuda"):
            torch.cuda.synchronize()                       # wait for the queue to actually drain
        dt = time.time() - t0
    return dt / n


# =============================================================================================
# SECTION 6 -- RENDERING ONE IMAGE
# =============================================================================================

def render_variant(spec, cid, variant_idx, seed, bundle, model, sched, steps, device, out_dir,
                   core_hash):
    """Render one image and write its PNG(s). Returns the manifest record for it.

    THE SEQUENCE IS THE LIVE PATH'S SEQUENCE, THROUGH THE SAME FUNCTIONS. Compare
    conddiff_inference.py:_generate_live -- pick a slice, derive `brain` from the T1, draw the
    mask (or zero it), build the 8-channel conditioning, run the DDIM loop, encode. Every one
    of those steps except "pick a slice" is a call into conddiff_core, so the only thing that
    differs between this render and a live one is the step count and the device, both recorded.

    spec        : the dict from core.params_to_spec
    cid         : this cell's stable id from core.cell_id
    variant_idx : 0..K-1
    seed        : the recorded seed; drives slice choice, mask shape AND diffusion noise
    returns a dict -- the manifest's per-image record
    """
    # ONE seed in, everything out. augment_tumor.py gets this half right and half wrong for a
    # service: it hard-seeds the mask rng to 0 (so every call draws the identical tumour) and
    # leaves the diffusion noise UNSEEDED (so no image is ever reproducible). Here one integer
    # drives the numpy Generator (slice choice + mask) and, inside core.render_flair, a torch
    # Generator (the initial noise) -- and that integer is written into the manifest.
    rng = np.random.default_rng(seed)

    # ---- 1. conditioning slice: a real held-out validation subject --------------------------
    stack, rec, sstats = pick_slice(spec, rng, bundle)      # (9,256,256) float32
    brain = stack[1] > BRAIN_TH                             # (256,256) bool -- ch 1 is the T1

    # ---- 2. the tumour mask (what becomes channel 2 of the conditioning) --------------------
    if spec["with_tumour"]:
        atlas_lobe = stack[core.ATLAS0 + spec["lobe_idx"]]  # (256,256) chosen lobe, probs [0,1]
        mask, _centre, mstats = core.synth_mask_sized(atlas_lobe, brain, rng, spec["size"])
    else:
        # "Without Tumor" ZEROES the mask channel. It does not synthesise a tiny lesion and it
        # does not reuse the bank slice's own mask. The bank slices are already tumour-free by
        # selection, but zeroing is explicit and cannot be defeated by a stale manifest.
        mask = np.zeros((256, 256), dtype=np.float32)
        mstats = {}

    # ---- 3. conditioning tensor + the reverse diffusion --------------------------------------
    cond = core.build_cond(stack, mask, device)             # (1,8,256,256) on `device`
    flair = core.render_flair(model, sched, cond, seed)     # (256,256) float32 in [0,1]

    # ---- 4. encode and write ------------------------------------------------------------------
    # core.to_png_bytes applies the repo's display orientation (np.flipud(arr.T), matching
    # `imshow(arr.T, origin="lower")` in every figure in the paper) and the FIXED vmin=0/vmax=1
    # intensity mapping. Encoding here, on the cluster, with the same function the live path
    # uses is what makes a gallery image and a live image visually interchangeable.
    cell_dir = os.path.join(out_dir, cid)
    os.makedirs(cell_dir, exist_ok=True)
    flair_rel = "%s/v%02d_flair.png" % (cid, variant_idx)   # manifest paths use "/" on purpose
    with open(os.path.join(out_dir, flair_rel.replace("/", os.sep)), "wb") as fh:
        fh.write(core.to_png_bytes(flair, kind="flair"))

    seg_rel = None
    if spec["with_tumour"]:
        # The mask is ground truth -- we drew it -- so shipping it is free extra value. The
        # backend files it under the 'seg' key, which the viewer already labels "Tumor Mask",
        # and having two images unlocks the viewer's Grid View toggle. For "Without Tumor" we
        # write no seg at all: an all-zero mask carries no information and is a constant image,
        # which divides by zero in any consumer that min-max normalises.
        seg_rel = "%s/v%02d_seg.png" % (cid, variant_idx)
        with open(os.path.join(out_dir, seg_rel.replace("/", os.sep)), "wb") as fh:
            fh.write(core.to_png_bytes(mask, kind="seg"))

    # ---- 5. the record ------------------------------------------------------------------------
    # THIS IS THE HONESTY CONTRACT. The backend copies these keys verbatim into the database
    # record it stores in the user's library, and it invents NOTHING to fill a gap: a key
    # missing here is a key listed as "not_recorded" there. So everything below is a
    # MEASUREMENT of what actually happened, never a restatement of what was asked for.
    #   - seed / ddim_steps        : what the sampler did
    #   - bank_*                   : whose anatomy this is (a real held-out subject, not output)
    #   - lesion_px, coverage, ... : the realised lesion, straight out of synth_mask_sized
    #   - core_sha256              : WHICH code drew it
    record = {
        "flair":         flair_rel,
        "seg":           seg_rel,
        "seed":          int(seed),
        "ddim_steps":    int(steps),
        # provenance of the ANATOMY
        "bank_file":     rec["file"],
        "bank_patient":  rec.get("patient", ""),
        "bank_z":        rec.get("z"),
        "bank_level":    rec.get("level", spec["level"]),
        "bank_brain_px": sstats["brain_px"],
        # provenance of the CODE (also stamped at manifest top level; repeated per image so a
        # single record pasted into an issue is self-contained)
        "core_sha256":   core_hash,
    }
    if seg_rel is None:
        record.pop("seg")                                  # absent, not null: see _load_gallery
    # mstats is exactly conddiff_core.synth_mask_sized's `stats` dict: lesion_px, core_px,
    # lobe_px, coverage, centre_axis0, centre_axis1, tries, size_requested, size_realized,
    # size_in_band. Empty for the no-tumour cells, where those numbers do not exist -- absent
    # because meaningless, not absent because lost.
    record.update(mstats)
    return record


# =============================================================================================
# SECTION 7 -- MAIN
# =============================================================================================

def main():
    # Env defaults, argparse overrides. The env vars are what the .sbatch sets (matching the
    # repo's other jobs, which are all driven by `export FOO=...` above the python line); the
    # flags are for interactive use. Reading env INSIDE the default= keeps both honest -- there
    # is one value, printed once, in the banner.
    def env_int(name, default):
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(str(raw).strip())
        except ValueError:
            print("  warning: %s=%r is not an integer; using %d" % (name, raw, default))
            return default

    ap = argparse.ArgumentParser(
        description="Pre-render the 57-cell gallery the Brain-Image-Generator backend serves.")
    ap.add_argument("--bundle", default=os.environ.get("GALLERY_BUNDLE", "web_bundle"),
                    help="the conditioning bank built by export_for_website.py "
                         "(default: web_bundle, env GALLERY_BUNDLE)")
    ap.add_argument("--out", default=os.environ.get("GALLERY_OUT", "conddiff_gallery"),
                    help="output folder; the backend expects it to be named conddiff_gallery "
                         "(default: conddiff_gallery, env GALLERY_OUT)")
    ap.add_argument("-k", "--variants", type=int, default=env_int("GALLERY_K", 4),
                    help="seed variants per cell (default 4, env GALLERY_K)")
    ap.add_argument("--steps", type=int, default=env_int("GALLERY_STEPS", 200),
                    help="DDIM steps (default 200 = the paper's value, env GALLERY_STEPS)")
    ap.add_argument("--base-seed", type=int, default=env_int("GALLERY_BASE_SEED", 20260101),
                    help="salt for the per-image seeds (default 20260101, env "
                         "GALLERY_BASE_SEED). Change it to re-roll the ENTIRE gallery.")
    ap.add_argument("--device", default=os.environ.get("GALLERY_DEVICE", "cuda"),
                    help="torch device (default cuda)")
    ap.add_argument("--cells", default="",
                    help="render only cells whose id contains this substring, e.g. "
                         "'cerebellum' or 'without_'. For repairing part of a gallery.")
    ap.add_argument("--force", action="store_true",
                    help="re-render even variants that already exist (disables resume)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the time estimate, write nothing, import no "
                         "torch. Safe on a LOGIN NODE -- use it to size the SLURM job.")
    args = ap.parse_args()

    if args.variants < 1:
        raise SystemExit("--variants must be >= 1 (got %d)" % args.variants)
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1 (got %d)" % args.steps)

    t_start = time.time()
    started = datetime.now(timezone.utc)

    print("=" * 92)
    print("render_gallery.py - pre-rendered gallery for %s" % MODEL_NAME)
    print("=" * 92)
    print("  bundle     : %s" % args.bundle)
    print("  out        : %s" % args.out)
    print("  variants K : %d per cell" % args.variants)
    print("  DDIM steps : %d%s" % (args.steps,
                                   "   (the paper's value)" if args.steps == 200 else
                                   "   (NOT the paper's 200 -- a degraded render)"))
    print("  base seed  : %d" % args.base_seed)
    print("  device     : %s" % args.device)

    # ---- the shared core, and its hash -------------------------------------------------------
    # This hash is the whole cross-machine correctness check. It goes into the manifest, the
    # backend prints it at boot, and comparing it against the sha256 of the backend's own copy
    # of conddiff_core.py is how a cluster/server divergence becomes a mechanical fact instead
    # of an argument about whether two files "look the same".
    core_path = os.path.abspath(core.__file__)
    if core_path.endswith(".pyc"):                          # __pycache__ shim; hash the source
        core_path = core_path.replace(".pyc", ".py")
    core_hash = sha256_of(core_path)
    print("  core       : %s" % core_path)
    print("  core sha256: %s" % core_hash)

    # ---- PHASE A: the bank -------------------------------------------------------------------
    print("\n[1] reading the conditioning bank")
    bundle = load_bundle(args.bundle)
    bman = bundle["manifest"]
    print("    %d slices on disk, %d recorded in the manifest, %d channels" %
          (bundle["n_npy"], len(bundle["by_file"]), 9 - bundle["ch_lo"]))
    print("    built %s by %s" % (bman.get("created_utc", "?"), bman.get("generator", "?")))
    print("    thresholds: PROB_TH=%s  MIN_LOBE=%s  MIN_BRAIN=%d%s" %
          (core.PROB_TH, core.MIN_LOBE, bundle["min_brain"],
           "" if bundle["min_brain"] == MIN_BRAIN_FALLBACK else "  (OVERRIDDEN at export)"))
    if bundle["empty_cells"]:
        print("    the bank could not fill %d of its 18 (lobe, level) cells: %s"
              % (len(bundle["empty_cells"]), ", ".join(bundle["empty_cells"])))
        print("      -> the gallery cells that depend on them will be listed as empty and MUST")
        print("         be disabled in the UI. That is anatomy, not a bug: the cerebellum")
        print("         genuinely does not exist in superior slices.")

    # ---- PHASE B: enumerate the work ----------------------------------------------------------
    # core.iter_cells() is the single definition of "how many cells exist" -- the backend's
    # selftest counts against the same 57. Building the list here from anything else (a nested
    # loop over the dropdown strings, say) is how the two sides come to disagree about the size
    # of the parameter space.
    # TWO LISTS, AND THE DISTINCTION IS LOAD-BEARING.
    #   all_cells  -- every cell the parameter space has. The MANIFEST is always written over
    #                 this list, so a partial run never orphans the cells it did not touch.
    #   cells      -- the subset this run will RENDER (--cells narrows it).
    # Collapsing the two would mean `--cells frontal` rewrites manifest.json with 9 cells in
    # it, leaving the other 48 cells' PNGs on disk but invisible to the backend, which indexes
    # the manifest and never the directory. That is a silent, total loss of most of a gallery,
    # and it would look like a successful run.
    all_cells = []
    for cid, ui in core.iter_cells():
        spec = core.params_to_spec(ui["tumour"], ui["lobe"], ui["slice_location"],
                                   ui["tumour_size"])
        # Round-trip assertion: the name iter_cells generated must be the name params_to_spec +
        # cell_id produce for the same dropdown strings. If these ever disagree, the renderer
        # would write files under names the backend will never look up, and every request would
        # 'miss' a gallery that is sitting right there on disk.
        assert core.cell_id(spec) == cid, "cell id round-trip failed: %r vs %r" % (cid, spec)
        all_cells.append((cid, ui, spec))

    cells = [c for c in all_cells if not args.cells or args.cells in c[0]]
    if not cells:
        raise SystemExit("--cells %r matched none of the %d cells."
                         % (args.cells, len(all_cells)))
    selected_ids = {c[0] for c in cells}
    print("\n[2] parameter space: %d cells total; rendering %d%s x %d variants = %d images"
          % (len(all_cells), len(cells), "" if not args.cells else " (--cells filter)",
             args.variants, len(cells) * args.variants))

    # ---- PHASE C: resume ----------------------------------------------------------------------
    # A variant counts as DONE only if BOTH its PNGs are on disk AND a manifest record for it
    # survived. PNGs without a record are re-rendered: the record is the provenance (seed, bank
    # subject, realised lesion area) and this script will not invent one to match a file it did
    # not write. That is the same rule the backend follows when a manifest key is missing.
    out_dir = args.out
    out_manifest_path = os.path.join(out_dir, OUT_MANIFEST)
    existing = {}
    # The previous manifest is kept whole, not just mined for variants: a run that finds
    # everything already rendered never imports torch, so it has no torch/diffusers versions
    # and no checkpoint facts of its own. Rewriting the manifest with empty ones would DELETE
    # provenance that the earlier run had legitimately recorded -- so those blocks are carried
    # forward from `prev_manifest` when this run has nothing better to say.
    # NOTE the previous manifest is read even under --force. --force means "re-render the cells
    # I asked for", not "forget every image on disk": the records for cells OUTSIDE this run's
    # selection are still carried forward, and only the selected ones are dropped (below).
    prev_manifest = {}
    if os.path.isfile(out_manifest_path):
        try:
            with open(out_manifest_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            prev_manifest = prev if isinstance(prev, dict) else {}
            if prev.get("schema") != GALLERY_SCHEMA:
                print("    existing manifest has schema %r, not %r -- ignoring it and starting "
                      "fresh" % (prev.get("schema"), GALLERY_SCHEMA))
            else:
                for cid, cell in (prev.get("cells") or {}).items():
                    for v in (cell.get("variants") or []):
                        rel = v.get("flair")
                        if not rel:
                            continue
                        idx = v.get("variant_index")
                        if idx is None:
                            # Older records did not carry the index; recover it from the
                            # filename "cid/vNN_flair.png" rather than dropping the record.
                            try:
                                idx = int(os.path.basename(rel).split("_")[0][1:])
                            except Exception:
                                continue
                        ok = os.path.isfile(os.path.join(out_dir, rel.replace("/", os.sep)))
                        if ok and v.get("seg"):
                            ok = os.path.isfile(os.path.join(out_dir,
                                                             v["seg"].replace("/", os.sep)))
                        if ok:
                            existing[(cid, int(idx))] = v
        except Exception as e:
            print("    could not read the existing manifest (%s) -- starting fresh" % e)

    if args.force:
        # Drop only the records this run is going to replace. Keeping the others is what makes
        # `--force --cells insula` a repair of nine cells rather than a demolition of 57.
        existing = {k: v for k, v in existing.items() if k[0] not in selected_ids}

    todo = []
    for cid, ui, spec in cells:
        for v in range(args.variants):
            if (cid, v) not in existing:
                todo.append((cid, ui, spec, v))
    if existing:
        print("    resume: %d image(s) already rendered and recorded, %d to go"
              % (len(existing), len(todo)))
        # A resumed gallery can legitimately contain images rendered at DIFFERENT step counts
        # (say, a 50-step smoke test that was then topped up at 200). Every record carries its
        # own ddim_steps, so nothing is mislabelled -- but a mixed gallery means two visitors
        # asking for the same thing can get visibly different image quality, which is worth
        # knowing about before it ships rather than after somebody notices.
        mixed = sorted({int(v.get("ddim_steps", -1)) for v in existing.values()} - {args.steps})
        if mixed:
            print("    WARNING: %d existing image(s) were rendered at %s DDIM steps, not %d. "
                  "The manifest records each image's own step count, so the record stays "
                  "honest -- but re-render with --force for a uniform gallery."
                  % (sum(1 for v in existing.values()
                         if int(v.get("ddim_steps", -1)) != args.steps),
                     ", ".join(str(m) for m in mixed), args.steps))
    if args.force:
        print("    --force: re-rendering the %d selected cell(s) even if they already exist "
              "(records for other cells are kept)" % len(cells))

    # ---- PHASE D: the time estimate, BEFORE committing --------------------------------------
    forwards = len(todo) * args.steps
    est_guess = forwards * SEC_PER_FORWARD_GUESS
    print("\n[3] cost estimate")
    print("    %d images x %d DDIM steps = %s U-Net forward passes"
          % (len(todo), args.steps, format(forwards, ",")))
    print("    a-priori guess at %.3f s/forward (~950 GFLOP each on an RTX 6000):" %
          SEC_PER_FORWARD_GUESS)
    print("        ~%s per image,  ~%s TOTAL  <-- size the SLURM --time from this"
          % (hms(args.steps * SEC_PER_FORWARD_GUESS), hms(est_guess)))
    print("    (a guess only -- the real run measures the GPU it was actually given and")
    print("     reprints this before the loop starts)")

    if args.dry_run:
        print("\n--dry-run: nothing written, no torch imported. Re-run without --dry-run, or")
        print("            sbatch scripts/render_gallery.sbatch")
        return

    if not todo:
        print("\nnothing to render -- the gallery is already complete. "
              "Use --force to re-render it anyway.")
        # Still fall through to the summary so the operator gets the size + sftp block.

    # ---- PHASE E: the model -------------------------------------------------------------------
    model = sched = None
    sec_per_fwd = SEC_PER_FORWARD_GUESS
    ckpt_facts = {}
    if todo:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                "--device %s but torch.cuda.is_available() is False.\n"
                "  On Great Lakes that almost always means the job did not get a GPU (check\n"
                "  --gres=gpu:1 and the partition) or the torch build does not match the card.\n"
                "  NOTE the cu128 wheel supports compute capability 7.5-12.0, so the V100\n"
                "  (CC 7.0) fails with 'no kernel image available' -- this job must run on the\n"
                "  RTX 6000 partition (gpu-rtx6000). Use --device cpu only for a smoke test;\n"
                "  a full 200-step gallery on CPU is days, not hours." % args.device)
        model, ckpt_facts = build_and_load(bundle["ckpt"], args.device)
        sched = build_scheduler(args.steps)
        print("    scheduler: DDIM, %d steps of %d training timesteps, clip_sample=True, eta=0"
              % (args.steps, 1000))

        print("\n[4] calibrating on the GPU we were actually given")
        sec_per_fwd = calibrate(model, args.device)
        est = len(todo) * args.steps * sec_per_fwd
        print("    measured %.3f s per forward pass over %d timed passes"
              % (sec_per_fwd, CALIBRATION_FORWARDS))
        print("    revised estimate: ~%s per image,  ~%s TOTAL"
              % (hms(args.steps * sec_per_fwd), hms(est)))
        if args.device.startswith("cuda"):
            print("    GPU: %s" % torch.cuda.get_device_name(0))

    # ---- PHASE F: render ----------------------------------------------------------------------
    # The manifest is assembled INCREMENTALLY and rewritten after every cell. That is the whole
    # resume story: whatever the wall clock interrupts, what is on disk is a valid manifest
    # describing exactly the images that are also on disk.
    os.makedirs(out_dir, exist_ok=True)
    commit, dirty = git_state()

    # Over all_cells, NOT over `cells`: a --cells run must leave the rest of the manifest
    # intact (their records are carried forward from `existing` just below).
    manifest_cells = {}
    for cid, ui, spec in all_cells:
        manifest_cells[cid] = {
            # The four user choices, in the EXACT strings the frontend POSTs, so the manifest
            # says what a cell means without anyone having to decode its name.
            "tumour":         ui["tumour"],
            "lobe":           ui["lobe"] if spec["with_tumour"] else None,
            "slice_location": ui["slice_location"],
            "tumour_size":    ui["tumour_size"] if spec["with_tumour"] else None,
            "variants":       [],
        }
        # Carry forward anything a previous run already rendered, in variant order.
        for v in range(args.variants):
            if (cid, v) in existing:
                manifest_cells[cid]["variants"].append(existing[(cid, v)])

    failures = []                     # (cell_id, variant, reason) -- printed and shipped
    used_seeds = {}                   # seed -> "cid[v]", to assert per-image seed uniqueness
    n_done = 0
    t_loop = time.time()

    def write_manifest():
        """Serialise the manifest as it currently stands. Called after every cell."""
        empty = sorted(cid for cid, c in manifest_cells.items() if not c["variants"])
        prev_renderer = (prev_manifest.get("renderer") or {})
        ckpt_block = ckpt_facts or (prev_manifest.get("checkpoint") or {})
        payload = jsonable({
            "schema":        GALLERY_SCHEMA,       # REQUIRED by the backend's _load_gallery
            "model_version": MODEL_NAME,
            "rendered_utc":  started.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "core_sha256":   core_hash,            # the cross-machine correctness check
            "renderer": {
                "script":          "scripts/render_gallery.py",
                "ddim_steps":      args.steps,
                "variants_per_cell": args.variants,
                "base_seed":       args.base_seed,
                "device":          args.device,
                "gpu":             _gpu_name(args.device) or prev_renderer.get("gpu"),
                "python":          sys.version.split()[0],
                "numpy":           np.__version__,
                "torch":           ckpt_block.get("torch_version") or prev_renderer.get("torch"),
                "diffusers":       (ckpt_block.get("diffusers_version")
                                    or prev_renderer.get("diffusers")),
                "git_commit":      commit,
                "git_dirty":       dirty,
                "sec_per_forward": round(sec_per_fwd, 4),
            },
            "checkpoint": ckpt_block,
            # Where the anatomy came from, so the gallery is traceable back to the bank even
            # though a gallery deployment ships no bank at all.
            "bank": {
                "dir":          os.path.abspath(bundle["dir"]),
                "created_utc":  bman.get("created_utc"),
                "split":        (bman.get("source") or {}).get("split"),
                "n_slices":     len(bundle["by_file"]),
                "min_brain":    bundle["min_brain"],
                "empty_cells":  bundle["empty_cells"],
            },
            "cells":       manifest_cells,
            "empty_cells": empty,
            "failures":    failures,
        })
        atomic_write_json(out_manifest_path, payload)

    if todo:
        print("\n[5] rendering")
        # Group the work by cell so the manifest is written at a natural boundary and the log
        # reads as a table rather than a stream.
        by_cell = {}
        for cid, ui, spec, v in todo:
            by_cell.setdefault(cid, []).append((spec, v))

        for cid, _ui, _spec in cells:
            work = by_cell.get(cid)
            if not work:
                continue
            for spec, v in work:
                seed = derive_seed(args.base_seed, cid, v)
                # A repeated seed would mean two "different" variants are the same image with
                # the same recorded provenance -- the gallery would claim more material than it
                # has. sha256-truncated-to-32-bits makes this astronomically unlikely across
                # 228 images; assert rather than assume.
                if seed in used_seeds:
                    raise SystemExit("seed collision: %d used by both %s and %s[%d]. Change "
                                     "--base-seed." % (seed, used_seeds[seed], cid, v))
                used_seeds[seed] = "%s[%d]" % (cid, v)

                t_img = time.time()
                try:
                    rec = render_variant(spec, cid, v, seed, bundle, model, sched, args.steps,
                                         args.device, out_dir, core_hash)
                except Exception as e:
                    # ONE unrenderable cell must not cost the other 56. The commonest cause is
                    # a genuinely empty (lobe, level) combination -- the cerebellum does not
                    # exist in superior slices -- which is anatomy, not a bug. Record it, print
                    # it, and let the backend's coverage report and the UI disable the cell.
                    failures.append({"cell": cid, "variant": v, "error": "%s: %s"
                                                                        % (type(e).__name__, e)})
                    print("    %-38s v%02d  FAILED: %s" % (cid, v, e))
                    continue

                rec["variant_index"] = v
                manifest_cells[cid]["variants"].append(rec)
                n_done += 1

                # Running average over images ACTUALLY rendered this run (not the resumed
                # ones), which is what makes the ETA converge quickly and stay honest.
                dt = time.time() - t_img
                avg = (time.time() - t_loop) / n_done
                left = len(todo) - n_done
                lesion = rec.get("lesion_px")
                print("    %-38s v%02d  seed=%-10d %5.1fs  %s  [%d/%d, ETA %s]"
                      % (cid, v, seed, dt,
                         ("lesion %4d px (%s%s)" % (lesion, rec.get("size_realized", "?"),
                                                    "" if rec.get("size_in_band", True) else "!"))
                         if lesion is not None else "no lesion            ",
                         n_done, len(todo), hms(avg * left)))

            # Keep the variants in index order even when a resumed record and a fresh one were
            # appended out of sequence, then checkpoint the manifest.
            manifest_cells[cid]["variants"].sort(key=lambda r: r.get("variant_index", 0))
            write_manifest()

    write_manifest()

    # ---- PHASE G: summary, size and the transfer command --------------------------------------
    elapsed = time.time() - t_start
    total_bytes = dir_size(out_dir)
    n_images = sum(len(c["variants"]) for c in manifest_cells.values())
    n_png = len(glob.glob(os.path.join(out_dir, "*", "*.png")))
    covered = sum(1 for c in manifest_cells.values() if c["variants"])
    empty = sorted(cid for cid, c in manifest_cells.items() if not c["variants"])

    print("\n" + "=" * 92)
    print("GALLERY COMPLETE" if not empty else "GALLERY COMPLETE (with empty cells)")
    print("=" * 92)
    print("  wall clock      : %s" % hms(elapsed))
    if n_done:
        print("  rendered now    : %d images  (%.1f s/image average)" % (n_done, (time.time() - t_loop) / n_done))
    print("  images recorded : %d across %d/%d cells  (%d PNG files on disk)"
          % (n_images, covered, len(all_cells), n_png))
    print("  bundle size     : %s   (manifest %s)"
          % (human(total_bytes), human(os.path.getsize(out_manifest_path))))
    if n_images:
        print("  per image       : ~%s" % human(total_bytes / max(1, n_png)))
    print("  core sha256     : %s" % core_hash)

    if failures:
        print("\n  %d render failure(s):" % len(failures))
        for f in failures[:12]:
            print("    %s[%d]  %s" % (f["cell"], f["variant"], f["error"]))
        if len(failures) > 12:
            print("    ... and %d more (all of them are in manifest.json['failures'])"
                  % (len(failures) - 12))
    if empty:
        print("\n  %d EMPTY cell(s) -- the site MUST disable these dropdown combinations, or"
              % len(empty))
        print("  the backend will raise on them (it refuses to substitute a different cell):")
        for cid in empty:
            print("    %s" % cid)

    remote = os.path.abspath(out_dir).replace("\\", "/")
    print("""
TRANSFER  (sftp, NOT scp)
  scp on Great Lakes is unreliable: the U-M login banner is printed into the same channel scp
  uses for its own protocol, and the transfer dies with
      "Received message too long 1349874536"
  (those bytes are literally the start of the banner text being read as a length field). sftp
  negotiates a proper subsystem after login, so the banner is harmless. Use sftp.

  1. On the LAPTOP, protect the shared web repo BEFORE dropping anything into it. The lab's
     .gitignore does not cover .png/.json under inference/, and a `git add -A` would try to
     stage the whole gallery into github.com/SOCR/Brain-Image-Generator:

       cd C:/Users/alexl/code/Research/socr/Brain-Image-Generator
       printf 'backend/inference/conddiff_gallery/\\n' >> .git/info/exclude

     (.git/info/exclude is LOCAL-ONLY, so it cannot be committed to the shared repo.)

  2. Pull the FOLDER down (not its contents -- `get -r .` is not a portable sftp idiom and
     would scatter manifest.json next to the .py files):

       sftp <uniqname>@greatlakes.arc-ts.umich.edu
       sftp> lcd C:/Users/alexl/code/Research/socr/Brain-Image-Generator/backend/inference
       sftp> cd {parent}
       sftp> get -r {name}
       sftp> bye

  3. Verify on the laptop. All three must agree with this run:

       python -c "import json;m=json.load(open('backend/inference/{name}/manifest.json'));print(len(m['cells']),'cells',sum(len(c['variants']) for c in m['cells'].values()),'images')"
       # must read: {ncells} cells {nimages} images
       python -c "import glob;print(len(glob.glob('backend/inference/{name}/*/*.png')),'PNGs')"
       # must read: {npng} PNGs

     And the check this entire two-copy design exists for -- the backend's core must be the
     same bytes as the one that drew these images:

       python -c "import hashlib;print(hashlib.sha256(open('backend/inference/conddiff_core.py','rb').read()).hexdigest())"
       # must equal manifest.json['core_sha256'] = {corehash}

     If those differ, STOP: the gallery was rendered by different maths than the server runs,
     and every image on the site is labelled by a parameter the server would have honoured
     differently. Re-sync conddiff_core.py to both repos and re-render.

  4. Start the backend in gallery mode (it is the default, but set it explicitly so the
     deployment documents itself):

       CONDDIFF_MODE=gallery
""".format(parent=os.path.dirname(remote), name=os.path.basename(remote),
           ncells=covered, nimages=n_images, npng=n_png, corehash=core_hash))


def _gpu_name(device):
    """The GPU's marketing name for the manifest, or None. Import-guarded because --dry-run and
    a fully-resumed run never import torch, and this is called from write_manifest()."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


if __name__ == "__main__":
    main()
