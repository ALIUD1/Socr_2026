#!/usr/bin/env python3
"""verify_conddiff_port.py -- prove the scipy-free numpy port computes the same maths as scipy.

WHY THIS SCRIPT EXISTS
======================
`src/conddiff_core.py` (and its byte-identical twin inside the web repo at
`backend/inference/conddiff_core.py`) reimplements two scipy functions in pure numpy:

    scipy.ndimage.gaussian_filter  ->  conddiff_core._gaussian_blur
    scipy.ndimage.label + argmax   ->  conddiff_core._largest_connected_component

The reason is deployment, not science: the web container would otherwise need scipy (~35 MB,
plus a dependency-resolver run against numpy==1.24.3 in a shared image that five other working
models depend on). Both functions are used by `synth_mask`, which draws EVERY synthetic lesion
the website shows.

UNTIL THIS SCRIPT, NOBODY HAD EVER CHECKED THAT THE REIMPLEMENTATIONS MATCH SCIPY.

That is a real and nasty gap, because the failure is SILENT and PLAUSIBLE:

  * `_gaussian_blur` smooths the random field that makes a lesion's outline organic. If it used
    a different boundary convention, or a different truncation radius, the field would be a
    DIFFERENT field. Every lesion would then be a different shape from every lesion in the
    paper -- and it would still look exactly like a tumour. No exception, no warning, no
    visible artefact. The website would simply be showing a different model's output while
    citing the paper's numbers.

  * `_largest_connected_component` decides which blob survives when the score field carves the
    top-N pixel set into a main mass plus specks. scipy's `label` defaults to 4-CONNECTIVITY
    (the plus-shaped structuring element). If the port had used 8-connectivity -- the more
    "obvious" choice, and what most hand-written flood fills do -- it would MERGE blobs that
    scipy keeps separate, so the surviving lesion would sometimes be a different, larger,
    corner-linked object. Again: still looks like a tumour.

The border case is the one a casual eyeball test is guaranteed to miss. A boundary-mode
mismatch only changes the outermost `radius = int(4*sigma + 0.5)` pixels -- 24 px at
NOISE_SIGMA=6.0 -- which for our data is the zero frame around the brain. You would never see
it in a picture, and it would still change the field's min/max, which the code then uses to
normalise the whole field to [0,1]. So a purely-cosmetic-looking edge bug propagates into every
interior pixel through the normalisation. That is exactly why this test measures the border and
the interior SEPARATELY and reports both.

WHERE THIS MUST BE RUN
======================
Great Lakes. It is the only machine that has BOTH scipy (to be the reference) AND the real
preprocessed BraTS slices (to be the input). Running it on the web server is pointless: the
whole reason the port exists is that scipy is not there.

    module load python                     # or: source your conda env
    cd ~/Summer2026
    python scripts/verify_conddiff_port.py                        # uses data/processed/slices/val
    python scripts/verify_conddiff_port.py --bank export/conddiff_bundle
    python scripts/verify_conddiff_port.py --require-real-data    # gate a deployment with this

EXIT CODES (this script is meant to gate the gallery render + deployment)
    0  every stated tolerance was met
    1  at least one check FAILED  -- do not render the gallery, do not deploy
    2  the script could not run at all (no scipy, or a source file is missing). This is
       deliberately a different code from 1: "I could not check" is not "I checked and it was
       fine", and a CI job should treat it as loudly as a failure.

WHAT IS AND IS NOT PROVEN HERE
==============================
PROVEN:  that the two numpy reimplementations agree with scipy, and that running the ORIGINAL
         scipy `synth_mask` from scripts/augment_tumor.py and the PORTED `synth_mask` from
         src/conddiff_core.py on the same slice with the same seeded rng gives the same mask.
NOT PROVEN: that the U-Net, the DDIM schedule, or the conditioning-tensor assembly agree.
         Those are checked elsewhere (conddiff_inference.selftest, the strict=True load, and
         the +/-1% 85.3M parameter assertion). This script is about the mask maths only.
"""
import argparse
import ast
import builtins
import glob
import hashlib
import json
import os
import sys
import time

# Put the repo root on sys.path so `from src import conddiff_core` works no matter where the
# script is invoked from. Same idiom as every other script in scripts/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import numpy as np

# scipy is the REFERENCE here, so its absence is not a failed check, it is an inability to
# check. Hence exit code 2 and a message that names the machine this belongs on.
try:
    from scipy.ndimage import gaussian_filter, label
    import scipy as _scipy
except ImportError as e:                                    # pragma: no cover -- environment
    print("FATAL: scipy is not importable, so there is nothing to compare against.")
    print(f"       ({e})")
    print("       This script is the one piece of the pipeline that REQUIRES scipy. Run it on")
    print("       Great Lakes (module load python / your conda env), not on the web server --")
    print("       the web server not having scipy is the entire reason the port exists.")
    sys.exit(2)

from src import conddiff_core as core

# The brain-footprint threshold. augment_tumor.py:139 and export_for_website.py:124 both use
# 0.05 on the T1 channel. It is NOT in conddiff_core because conddiff_core takes `brain` as an
# argument -- deciding what counts as brain is the caller's job on each side. Repeated here so
# that the slices this script feeds to synth_mask are the same ones the pipeline would feed it.
BRAIN_TH = 0.05

ORIGINAL_SRC = os.path.join(_REPO_ROOT, "scripts", "augment_tumor.py")
CORE_SRC     = os.path.join(_REPO_ROOT, "src", "conddiff_core.py")
# The web repo's copy, which is supposed to be byte-identical. Checked if present; a missing
# web repo is not an error (this script must run on the cluster, where it will be absent).
WEB_CORE_SRC = os.path.join(os.path.dirname(_REPO_ROOT), "Brain-Image-Generator",
                            "backend", "inference", "conddiff_core.py")

BANNER = "=" * 92


# =============================================================================================
# SECTION 0 -- THE RESULT LEDGER
#
# Every check funnels through one Report object so that (a) the final summary cannot disagree
# with the body of the output, and (b) the exit code is a pure function of what was recorded.
# A check that forgets to record itself is invisible, so each section returns its own count and
# the summary prints the total -- if that total ever looks too small, a section silently
# returned early.
# =============================================================================================

class Report:
    """Accumulates (section, name, status, detail) rows and decides the exit code.

    Statuses:
      PASS  the check ran and met its stated tolerance
      FAIL  the check ran and did not  -> exit 1
      WARN  something is worth a human's attention but is not a correctness violation
            (e.g. the masks matched to Dice 1.0 but were not BIT-identical)
      SKIP  the check could not run (e.g. no real data on this machine). Never exit-worthy on
            its own; `--require-real-data` converts the specific skips that matter into FAILs.
    """

    def __init__(self):
        self.rows = []

    def add(self, section, name, status, detail=""):
        self.rows.append((section, name, status, detail))
        return status == "PASS"

    def ok(self, section, name, detail=""):
        return self.add(section, name, "PASS", detail)

    def fail(self, section, name, detail=""):
        return self.add(section, name, "FAIL", detail)

    def warn(self, section, name, detail=""):
        return self.add(section, name, "WARN", detail)

    def skip(self, section, name, detail=""):
        return self.add(section, name, "SKIP", detail)

    def check(self, section, name, condition, detail=""):
        """The workhorse: record PASS/FAIL from a boolean and echo it immediately.

        Printing at record time (rather than only in the summary) matters because these loops
        can take minutes on real data -- you want to see the first failure the moment it
        happens, not after the whole scan."""
        status = "PASS" if condition else "FAIL"
        self.add(section, name, status, detail)
        if status == "FAIL":
            print(f"    FAIL  {name}   {detail}")
        return bool(condition)

    def count(self, status):
        return sum(1 for r in self.rows if r[2] == status)

    @property
    def failed(self):
        return self.count("FAIL") > 0


def sha256_of(path, chunk=1 << 20):
    """Streaming sha256 of a file. Streaming, not read-all, because this is also used on the
    341 MB checkpoint elsewhere in the repo and the idiom should be the same everywhere."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# =============================================================================================
# SECTION 1 -- LOAD THE *ORIGINAL* scipy synth_mask WITHOUT IMPORTING augment_tumor.py
#
# `import augment_tumor` would work on Great Lakes, but it executes the module's top level,
# which does `import torch`, `import diffusers` and `matplotlib.use("Agg")`. That is ~10 s of
# import time, a CUDA context we do not want, and a hard dependency on the full training
# environment for a test that only needs numpy + scipy.
#
# So instead we PARSE the file and execute only two kinds of top-level node:
#   * every `Assign` whose value is a plain literal (that is LOBES, ATLAS0, PROB_TH, MIN_BRAIN,
#     MIN_LOBE, AREA_FRAC, CORE_FRAC, MIN_AREA, COMPACT, NOISE_SIGMA, NOISE_AMP -- and it
#     naturally excludes DEVICE, which calls torch.cuda.is_available(), and STEPS, which reads
#     os.environ), and
#   * the `synth_mask` FunctionDef itself.
#
# The point is not just speed. It is that we execute the ORIGINAL SOURCE TEXT rather than a
# re-typed copy of it. A re-typed reference would be a second port, and then the test would
# only prove that two of my transcriptions agree with each other.
# =============================================================================================

def _free_global_names(fn_node):
    """Names a function body reads but never binds -- i.e. what it expects from its globals.

    Used as a pre-flight guard. Without it, an edit to augment_tumor.py that introduces a new
    module-level helper would surface as a bare NameError from deep inside synth_mask, in the
    middle of a numeric test, which is a confusing way to learn that the extraction is stale.

    fn_node : ast.FunctionDef
    returns set[str]
    """
    bound = set()
    a = fn_node.args
    for group in (getattr(a, "posonlyargs", []), a.args, a.kwonlyargs):
        for arg in group:
            bound.add(arg.arg)
    if a.vararg:
        bound.add(a.vararg.arg)
    if a.kwarg:
        bound.add(a.kwarg.arg)

    used = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else used).add(n.id)
    return used - bound


def load_original_synth_mask(path):
    """Extract and compile scripts/augment_tumor.py's synth_mask with scipy wired in.

    Returns (func, consts, meta) where
        func   : the original synth_mask(atlas_lobe, brain, rng) -> (mask, centre)
        consts : dict of the literal module-level constants it was compiled against
        meta   : provenance (sha256, line span) for the printout
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)

    keep, consts, fn_node = [], {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)   # raises for torch.cuda..., os.environ...
            except Exception:
                continue                               # not a literal -> not a constant we want
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not names:
                continue
            keep.append(node)
            for n in names:
                consts[n] = value
        elif isinstance(node, ast.FunctionDef) and node.name == "synth_mask":
            keep.append(node)
            fn_node = node

    if fn_node is None:
        print(f"FATAL: no top-level `def synth_mask` found in {path}. Either it was renamed or "
              f"moved into a class; this script's extraction needs updating.")
        sys.exit(2)

    # The namespace the extracted code will see. THESE THREE NAMES ARE THE REFERENCE
    # IMPLEMENTATION: scipy's real gaussian_filter and label, called with their real defaults.
    ns = {"np": np, "gaussian_filter": gaussian_filter, "label": label, "__name__": "augment_tumor_orig"}

    missing = _free_global_names(fn_node) - set(ns) - set(consts) - set(dir(builtins))
    if missing:
        print(f"FATAL: augment_tumor.synth_mask references module-level name(s) this script "
              f"does not provide: {sorted(missing)}.")
        print( "       augment_tumor.py has changed shape. Add them to the extraction "
               "namespace above (and ask whether conddiff_core needs them too).")
        sys.exit(2)

    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), ns)            # noqa: S102 -- executing our own repo

    meta = dict(sha256=sha256_of(path),
                lines=(fn_node.lineno, getattr(fn_node, "end_lineno", -1)),
                n_consts=len(consts))
    return ns["synth_mask"], consts, meta


def scipy_largest_component(blob):
    """The REFERENCE for _largest_connected_component: augment_tumor.py:95-98, verbatim.

    Transcribed exactly, INCLUDING the `if n > 1` guard, because that guard is load-bearing in
    two cases the port has to reproduce:
      * n == 1 -> the original returns `blob` untouched (which equals the sole component), and
      * n == 0 -> `np.bincount(lab.ravel())` would be `[65536]`, `sizes[0] = 0` makes it `[0]`,
        and `argmax()` would return 0, so `lab == 0` would light up the ENTIRE image. The
        guard is the only thing standing between an empty mask and a full-frame lesion.

    blob : (H, W) bool
    returns (H, W) bool
    """
    lab, n = label(blob)                                   # scipy: 4-connectivity by default
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0     # index 0 = background, ignore it
        blob = lab == sizes.argmax()                       # ties -> lowest label -> raster-first
    return blob


# =============================================================================================
# SECTION 2 -- REAL DATA: turn slices into (atlas_lobe, brain) test cases
#
# Two sources, because this script has to be runnable both before and after the conditioning
# bank exists:
#   --slices  data/processed/slices/val/*.npy   9 channels: 0=FLAIR 1=T1 2=mask 3+k=atlas lobe k
#   --bank    export/conddiff_bundle/           8 channels: 0=T1 1=mask 2+k=atlas lobe k
#             (or 9, if the bank was built with --keep-flair; the manifest says which)
#
# The bank is the more meaningful input: it is literally the array the web backend will hand to
# synth_mask, float16-on-disk and all. The raw split is the fallback and is a superset.
# =============================================================================================

def _cases_from_stack(stack, t1_idx, atlas0_idx, source, rep_lobes):
    """One (9or8, 256, 256) stack -> a list of per-lobe test cases that are actually viable.

    A case is only useful if the lobe is genuinely present in the slice: `synth_mask` returns
    (None, None) below MIN_LOBE, and comparing None to None proves nothing. So we apply the
    same gate the pipeline applies.

    Returns list of dicts: {lobe, lobe_px, atlas (256,256) f32, brain (256,256) bool, source}
    """
    # The .astype(np.float32) is MANDATORY, not cosmetic: the bank ships float16 (see
    # export_for_website.py's DTYPE JUSTIFICATION), and float16 arithmetic inside synth_mask's
    # score field would quantise the scores so coarsely that argpartition's tie-breaking --
    # and therefore the lesion outline -- would change. This is the same cast the manifest
    # records under channels.loader.
    stack = np.asarray(stack).astype(np.float32)
    brain = stack[t1_idx] > BRAIN_TH                        # (256,256) bool
    out = []
    for k, lobe in enumerate(core.LOBES):
        if rep_lobes and lobe not in rep_lobes:
            continue
        atlas = stack[atlas0_idx + k]                       # (256,256) float32 in [0,1]
        lobe_px = int(((atlas > core.PROB_TH) & brain).sum())
        if lobe_px < core.MIN_LOBE:
            continue                                        # synth_mask would return None
        out.append(dict(lobe=lobe, lobe_px=lobe_px, atlas=atlas, brain=brain, source=source))
    return out


def load_real_cases(args, rep):
    """Collect up to --max-cases real (atlas_lobe, brain) pairs. Returns (cases, note)."""
    cases, note = [], ""

    if args.bank:
        man_path = os.path.join(args.bank, "manifest.json")
        if not os.path.isfile(man_path):
            return [], f"--bank {args.bank} has no manifest.json"
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        n_ch = int(man.get("channels", {}).get("count", 8))
        # The bank drops the real patient FLAIR (channel 0) unless built with --keep-flair, so
        # every index shifts by one. Reading it from the manifest rather than assuming is the
        # difference between testing the atlas and testing the tumour mask by mistake.
        t1_idx, atlas0_idx = (0, 2) if n_ch == 8 else (1, core.ATLAS0)
        files = sorted(glob.glob(os.path.join(args.bank, "cond_slices", "*.npy")))
        note = f"conditioning bank {args.bank} ({n_ch} channels, t1@{t1_idx}, atlas@{atlas0_idx})"
    else:
        files = sorted(glob.glob(os.path.join(args.slices, "*.npy")))
        t1_idx, atlas0_idx = 1, core.ATLAS0
        note = f"raw split {args.slices} (9 channels, t1@1, atlas@3)"

    if not files:
        return [], f"no .npy files under {args.bank or args.slices}"

    # Take EVENLY SPACED files, not the first N. Filenames are "<patient>_z###.npy" and sort
    # by patient, so the first N are all the same brain at adjacent z -- which would make the
    # test look broad while actually testing one anatomy.
    n_take = min(args.max_slices, len(files))
    picks = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, n_take).astype(int))]

    wanted = [l.strip().lower() for l in args.lobes.split(",")] if args.lobes else []
    for path in picks:
        try:
            stack = np.load(path)
        except Exception as e:
            rep.warn("data", f"unreadable {os.path.basename(path)}", str(e))
            continue
        if stack.ndim != 3 or stack.shape[-2:] != (256, 256):
            rep.warn("data", f"odd shape {os.path.basename(path)}", str(stack.shape))
            continue
        for c in _cases_from_stack(stack, t1_idx, atlas0_idx, os.path.basename(path), wanted):
            cases.append(c)
            if len(cases) >= args.max_cases:
                return cases, note
    return cases, note


def phantom_case():
    """A synthetic (atlas_lobe, brain) pair, so the script still has teeth with no data at all.

    This is NOT a substitute for real atlas channels -- a real lobe map is ragged, has holes,
    and its probability gradient is not a clean Gaussian -- but it exercises exactly the same
    code path, and it means a developer on a laptop can catch a broken port before pushing.

    brain : an ellipse roughly the size of a mid-axial BraTS brain footprint
    atlas : a smooth bump offset from centre, thresholded by the brain, ~7000 px above PROB_TH
    """
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)          # (256,256) each
    brain = (((yy - 128) / 95.0) ** 2 + ((xx - 128) / 78.0) ** 2) < 1.0
    atlas = np.exp(-(((yy - 100) ** 2 + (xx - 110) ** 2) / (2 * 35.0 ** 2))).astype(np.float32)
    atlas *= brain                                              # the atlas never claims non-brain
    lobe_px = int(((atlas > core.PROB_TH) & brain).sum())
    return dict(lobe="phantom", lobe_px=lobe_px, atlas=atlas, brain=brain, source="phantom")


# =============================================================================================
# SECTION 3 -- CHECK 0: DO THE CONSTANTS EVEN AGREE?
#
# Runs first and is the cheapest check in the file, but it is the one most likely to fire in
# practice. The numeric ports can be perfect and the two sides still disagree, because
# conddiff_core RETYPED the constants instead of importing them (it has to: it must not import
# augment_tumor, which imports torch). A retyped 0.4 that becomes 0.04 changes every lesion and
# no other check in this script would notice, since both halves of every comparison below use
# conddiff_core's copy for the port and augment_tumor's copy for the reference... except
# synth_mask, where they would silently diverge. So: compare them explicitly.
# =============================================================================================

def check_constants(consts, rep):
    print(f"\n[0] CONSTANT PARITY  augment_tumor.py  vs  src/conddiff_core.py")
    print("    " + "-" * 86)
    # (name in augment_tumor, name in conddiff_core). AREA_FRAC was renamed when it became a
    # parameter with a default, so it is the one pair whose names differ.
    pairs = [("LOBES", "LOBES"), ("ATLAS0", "ATLAS0"), ("PROB_TH", "PROB_TH"),
             ("MIN_LOBE", "MIN_LOBE"), ("AREA_FRAC", "AREA_FRAC_DEFAULT"),
             ("CORE_FRAC", "CORE_FRAC"), ("MIN_AREA", "MIN_AREA"), ("COMPACT", "COMPACT"),
             ("NOISE_SIGMA", "NOISE_SIGMA"), ("NOISE_AMP", "NOISE_AMP")]
    n = 0
    for a_name, c_name in pairs:
        if a_name not in consts:
            rep.fail("constants", a_name, f"not found as a literal in augment_tumor.py")
            n += 1
            continue
        a_val = consts[a_name]
        c_val = getattr(core, c_name, "<MISSING>")
        # tuple vs list: AREA_FRAC is a tuple both sides, but be robust to a list creeping in --
        # a (0.15,0.35) tuple and a [0.15,0.35] list are the same science and `==` says False.
        equal = (list(a_val) == list(c_val)) if isinstance(a_val, (tuple, list)) else (a_val == c_val)
        mark = "ok " if equal else "!! "
        print(f"    {mark}{a_name:<14} {str(a_val):<34} {c_name:<18} {c_val}")
        rep.check("constants", f"{a_name} == core.{c_name}", equal,
                  f"augment_tumor={a_val!r} core={c_val!r}")
        n += 1
    # MIN_BRAIN is deliberately NOT compared: conddiff_core does not define it, on purpose (see
    # its SECTION 1 comment). It is the one threshold the two sides are allowed to differ on.
    print(f"    (MIN_BRAIN is intentionally absent from conddiff_core -- deployment-tunable, "
          f"cosmetic, not compared)")
    return n


# =============================================================================================
# SECTION 4 -- CHECK 1: _gaussian_blur  vs  scipy.ndimage.gaussian_filter
# =============================================================================================

def _blur_compare(img, sigma, rep, label_txt, rtol, atol, verbose=True):
    """One field, one sigma: port vs scipy across every boundary mode. Returns n_checks.

    The GATE is the comparison against scipy's DEFAULTS -- mode='reflect', truncate=4.0 -- which
    is what conddiff_core's docstring claims to reproduce. Every other mode is computed too,
    but only as a DIAGNOSTIC: they exist so that a failure tells you *which* convention the port
    actually implements instead of just "the numbers differ". In practice the wrong modes are
    off by 10-100% relative while the right one is off by ~1e-7, so the contrast column below is
    also the evidence that this test has teeth at all.

    img   : (H, W) float array
    sigma : float
    returns int, the number of PASS/FAIL rows recorded
    """
    port = _as64(core._gaussian_blur(img, sigma))                    # (H,W) f64 view of f32 out
    ref  = _as64(gaussian_filter(img, sigma))                        # (H,W) scipy DEFAULTS
    diff = np.abs(port - ref)                                        # (H,W)
    scale = float(np.abs(ref).max())
    scale_safe = scale if scale > 0 else 1.0                         # all-zero field -> no div0

    max_abs = float(diff.max())
    max_rel = max_abs / scale_safe
    # plain ints, not np.int64, so the printed coordinate is readable in a log
    where = tuple(int(v) for v in np.unravel_index(int(diff.argmax()), diff.shape))

    # THE BORDER SPLIT. radius is exactly what conddiff_core computes, so the band it names is
    # exactly the set of pixels whose value depends on the boundary convention. If max_rel is
    # small in the interior and large in the band, the port's blur is right and its EDGE
    # HANDLING is wrong -- the single most likely, least visible way for this to break.
    radius = int(4.0 * sigma + 0.5)
    h, w = diff.shape
    if 2 * radius < min(h, w):
        interior = diff[radius:h - radius, radius:w - radius]
        border_max = float(max(diff[:radius].max(), diff[h - radius:].max(),
                               diff[:, :radius].max(), diff[:, w - radius:].max()))
        interior_max = float(interior.max())
    else:
        border_max, interior_max = max_abs, float("nan")             # field smaller than the kernel

    passed = rep.check("blur", f"{label_txt} sigma={sigma}",
                       (max_abs <= atol) or (max_rel <= rtol),
                       f"max|diff|={max_abs:.3e} rel={max_rel:.3e} at {where} "
                       f"(atol={atol:.1e} rtol={rtol:.1e})")

    if verbose or not passed:
        # The contrast row: how far the port is from every OTHER convention. Reading this is how
        # you diagnose a failure in one glance -- it tells you WHICH convention the port
        # actually implements, not merely that the numbers differ.
        #
        # CAVEAT, stated so nobody over-reads this row: a field that is flat or zero near its
        # edges is MODE-INVARIANT -- every boundary convention extrapolates the same values, so
        # every column here reads ~0 and the row proves nothing about edge handling. That is the
        # case for `constant_ones` and for every real atlas channel (they are zero outside the
        # skull). The fields that genuinely discriminate are the random ones, which have real
        # signal pressed against all four edges.
        others = []
        for mode in ("mirror", "nearest", "constant", "wrap"):
            d = float(np.abs(port - _as64(gaussian_filter(img, sigma, mode=mode))).max())
            others.append(f"{mode}={d / scale_safe:.1e}")
        t2 = float(np.abs(port - _as64(gaussian_filter(img, sigma, truncate=2.0))).max())
        print(f"    {'ok ' if passed else '!! '}{label_txt:<22} sigma={sigma:<5} "
              f"rel(reflect)={max_rel:.2e}  interior={interior_max:.2e} border={border_max:.2e}"
              f"  worst@{where}")
        print(f"        contrast: rel vs {'  '.join(others)}   |  vs truncate=2.0: "
              f"{t2 / scale_safe:.1e}")
    return 1


def _as64(a):
    """float64 view for differencing. The port returns float32 by contract; scipy returns the
    input dtype. Differencing in float64 means the numbers we print are the true differences and
    not themselves rounded."""
    return np.asarray(a, dtype=np.float64)


def check_gaussian(cases, args, rep):
    print(f"\n[1] GAUSSIAN BLUR   conddiff_core._gaussian_blur  vs  scipy.ndimage.gaussian_filter")
    print(f"    scipy defaults being reproduced: mode='reflect'  (d c b a | a b c d), truncate=4.0")
    print(f"    numpy calls scipy's 'reflect' by the name 'symmetric'; numpy's own 'reflect' is")
    print(f"    scipy's 'mirror'. Getting that backwards changes only the outer int(4*sigma+0.5)")
    print(f"    pixels -- for us the zero frame around the brain -- and would be invisible.")
    print(f"    The GATE is the 'rel(reflect)' column. The 'contrast' line under each row is a")
    print(f"    diagnostic: it says how far the port sits from every OTHER convention, so a")
    print(f"    failure names the bug. Note the RANDOM fields are what discriminate boundary")
    print(f"    mode -- a real atlas channel is zero at every edge, so for it EVERY mode agrees")
    print(f"    and the contrast row is uninformative by construction. That is exactly why this")
    print(f"    check does not rely on real data alone.")
    print("    " + "-" * 86)
    n = 0
    sigmas = [float(s) for s in args.sigmas.split(",")]
    rng = np.random.default_rng(args.seed)

    # ---- (a) random and adversarial synthetic fields ----------------------------------------
    fields = {
        # A standard normal field is exactly what synth_mask blurs (rng.standard_normal((256,256))
        # cast to float32), so this is the real workload, not a proxy.
        "gauss_f32":     rng.standard_normal((256, 256)).astype(np.float32),
        "gauss_f64":     rng.standard_normal((256, 256)),
        "uniform_f32":   rng.random((256, 256)).astype(np.float32),
        # Blurring a constant must return the same constant. This is the one field that tests the
        # KERNEL NORMALISATION (weights summing to 1) independently of everything else.
        "constant_ones": np.ones((256, 256), np.float32),
        # A step against the top edge: maximally sensitive to the boundary convention, because
        # every mode extrapolates that edge differently.
        "edge_step":     np.concatenate([np.ones((8, 256), np.float32),
                                         np.zeros((248, 256), np.float32)], axis=0),
        # A single impulse in the corner: the blurred result IS the kernel, clipped by the
        # boundary. If the radius or the taps are wrong this is where it shows up cleanly.
        "corner_impulse": np.zeros((256, 256), np.float32),
        # A non-square field, to catch an axis mix-up in the separable two-pass loop. A square
        # test can never see a transposed kernel.
        "nonsquare_f32": rng.standard_normal((97, 211)).astype(np.float32),
    }
    fields["corner_impulse"][0, 0] = 1.0

    for name, img in fields.items():
        for sigma in sigmas:
            n += _blur_compare(img, sigma, rep, name, args.rtol_blur, args.atol_blur,
                               verbose=not args.quiet)

    # ---- (b) ATLAS channels (real BraTS when available, phantom otherwise) -------------------
    # Random fields prove the arithmetic and the boundary convention. Atlas channels prove it on
    # the actual value distribution the pipeline feeds in: mostly exact zeros outside the brain,
    # a smooth probability ramp inside, and a hard cut at the skull -- a combination a synthetic
    # field does not reproduce. They cannot test edge handling (see the note above); the two
    # halves of this check are complementary, not redundant.
    if not cases:
        rep.skip("blur", "atlas channels", "no atlas cases available at all")
        print("    SKIP  atlas channels (no data and no phantom)")
    else:
        real = cases[:args.max_real_blur]
        for c in real:
            if float(np.abs(c["atlas"]).max()) == 0.0:
                continue                                 # a lobe absent from this slice: no signal
            n += _blur_compare(c["atlas"], core.NOISE_SIGMA, rep,
                               f"atlas:{c['lobe'][:8]}", args.rtol_blur, args.atol_blur,
                               verbose=not args.quiet)
    return n


# =============================================================================================
# SECTION 5 -- CHECK 2: _largest_connected_component  vs  scipy.ndimage.label + argmax
# =============================================================================================

def _build_cc_cases(cases, args, rep):
    """The blob zoo. Returns an ordered dict-like list of (name, (H,W) bool).

    The adversarial cases are chosen to separate the port from the WRONG implementations
    somebody would plausibly write, not just to cover lines:

      diagonal_chain / checkerboard  -- 8 pixels touching only at corners. scipy's default
          structure is generate_binary_structure(2,1), the PLUS shape, so these are 8 separate
          one-pixel components and the answer is a SINGLE pixel. An 8-connected port answers
          "all 8 pixels", a 700% error that looks like a perfectly reasonable lesion.
      two_equal / three_equal        -- exact size ties. scipy's argmax-of-bincount keeps the
          LOWEST LABEL, and scipy labels in raster-scan order of first encounter; the port keeps
          the first component found by np.nonzero's row-major order and uses `>` not `>=`. Those
          are the same rule, but only if both halves are right. A `>=` in the port would silently
          prefer the LAST tied component.
      empty                          -- n == 0. The `if n > 1` guard is the only thing stopping
          the reference from returning an all-True frame here (see scipy_largest_component).
      single_px / all_true           -- n == 1, where the original returns blob untouched.
      annulus / border_touching      -- a component that wraps around a hole and one that runs
          along the array edge, to catch an off-by-one in the port's neighbour bounds check.
    """
    z = lambda: np.zeros((32, 32), bool)
    out = []

    out.append(("empty", z()))

    a = z(); a[5, 5] = True
    out.append(("single_px", a))

    b = z(); b[2:5, 2:5] = True; b[20:23, 20:23] = True
    out.append(("two_equal_3x3", b))

    c = z(); c[1:3, 1:3] = True; c[1:3, 12:14] = True; c[24:26, 6:8] = True
    out.append(("three_equal_2x2", c))

    d = z()
    for i in range(12):
        d[i, i] = True                                   # corner-touch only: 12 components to scipy
    out.append(("diagonal_chain", d))

    e = z(); e[::2, ::2] = True                          # 256 one-pixel components
    out.append(("checkerboard", e))

    out.append(("all_true", np.ones((32, 32), bool)))

    f = z(); f[0, :] = True; f[10:14, 10:12] = True      # component hugging row 0
    out.append(("border_touching", f))

    g = z(); g[6:24, 6:24] = True; g[10:20, 10:20] = False
    out.append(("annulus", g))

    h = z(); h[2:8, 2:14] = True; h[20:24, 3:6] = True   # one clear winner + a decoy
    out.append(("big_plus_small", h))

    # Random binary fields at several densities. Below the percolation threshold these produce
    # hundreds of small components (stressing the tie/ordering logic); above it, one giant
    # component with holes (stressing the flood fill).
    rng = np.random.default_rng(args.seed + 17)
    for p in (0.15, 0.30, 0.45, 0.55, 0.70):
        out.append((f"random_p{int(p*100)}", rng.random((64, 64)) < p))

    # ---- REAL blobs, and specifically the blobs synth_mask actually produces ----------------
    # Step 4 of synth_mask takes the top-N pixels of (atlas x compactness x smoothed noise)
    # restricted to the lobe. That set is usually one mass plus a few specks -- the exact
    # distribution step 5 has to resolve. Reconstructing it here (rather than using a plain
    # threshold of the atlas, which gives one clean blob and tests nothing) is what makes this
    # section a test of the real workload.
    for c_i, case in enumerate(cases[:args.max_real_cc]):
        atlas, brain = case["atlas"], case["brain"]
        lobe = (atlas > core.PROB_TH) & brain
        n_lobe = int(lobe.sum())
        if n_lobe < core.MIN_LOBE:
            continue
        r2 = np.random.default_rng(args.seed + 1000 + c_i)
        noise = core._gaussian_blur(r2.standard_normal(atlas.shape).astype(np.float32),
                                    core.NOISE_SIGMA)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        score = atlas.astype(np.float32) * (1.0 + core.NOISE_AMP * (noise - 0.5))
        score[~lobe] = 0.0
        flat = score.ravel()
        for frac in (0.20, 0.50):                        # two different "top-N" pressures
            n_les = max(core.MIN_AREA, int(frac * n_lobe))
            n_les = min(n_les, int((flat > 0).sum()))
            top = np.argpartition(flat, -n_les)[-n_les:]
            blob = np.zeros(flat.size, bool); blob[top] = True
            out.append((f"topN:{case['lobe'][:8]}:{c_i}:{int(frac*100)}",
                        blob.reshape(atlas.shape)))
        # Plus the plain thresholded lobe, which is the simplest real shape there is.
        out.append((f"lobe:{case['lobe'][:8]}:{c_i}", lobe))

    if not cases:
        rep.skip("components", "atlas-derived blobs", "no atlas cases available")
    return out


def check_components(cases, args, rep):
    print(f"\n[2] CONNECTED COMPONENTS  conddiff_core._largest_connected_component")
    print(f"    vs  scipy.ndimage.label + np.bincount().argmax()   (augment_tumor.py:95-98)")
    print(f"    scipy's default structuring element is 4-CONNECTED (the plus shape). An")
    print(f"    8-connected port would MERGE corner-touching blobs that scipy keeps apart.")
    print("    " + "-" * 86)
    blobs = _build_cc_cases(cases, args, rep)
    n_shown = 0
    for name, blob in blobs:
        port = core._largest_connected_component(blob)
        ref  = scipy_largest_component(blob)
        # Tolerance here is EXACT. These are boolean arrays; there is no floating point in the
        # question, so "close" would be a meaningless thing to allow.
        same = bool(np.array_equal(port, ref))
        _, ncomp = label(blob)
        detail = (f"ncomp={ncomp} port_px={int(port.sum())} scipy_px={int(ref.sum())} "
                  f"disagree_px={int((port != ref).sum())}")
        rep.check("components", name, same, detail)
        if not same or (not args.quiet and n_shown < 40):
            print(f"    {'ok ' if same else '!! '}{name:<26} {detail}")
            n_shown += 1
    n_atlas = sum(1 for nm, _ in blobs if nm.startswith(("topN", "lobe")))
    kind = "phantom" if (cases and cases[0]["source"] == "phantom") else "real BraTS"
    print(f"    {len(blobs)} blobs compared "
          f"({n_atlas} of them derived from {kind} atlas channels, including the top-N sets "
          f"synth_mask itself produces)")
    return len(blobs)


# =============================================================================================
# SECTION 6 -- CHECK 3: THE END-TO-END TEST THAT ACTUALLY MATTERS
#
# Checks 1 and 2 verify the two building blocks. This verifies the building. It runs the
# ORIGINAL scipy synth_mask and the PORTED numpy synth_mask on the SAME atlas slice with the
# SAME seeded rng and compares the resulting masks pixel by pixel.
#
# WHY THE SAME SEED IS SUFFICIENT AND NECESSARY: synth_mask draws from rng exactly three times,
# in this order -- rng.choice for the seed pixel, rng.uniform for the target area, and
# rng.standard_normal((256,256)) for the noise field. Both versions consume the generator in
# that same order, so two generators constructed from the same seed are in identical states at
# each draw. That makes the entire mask a pure function of (atlas, brain, seed), and any
# difference in the output is a difference in the CODE, which is precisely what we want to
# measure. It also means the CENTRE is a free canary: it is decided before either ported
# function is called, so if the centres ever differ, the two sides are consuming the rng
# differently and every other number below is meaningless.
# =============================================================================================

def _mask_metrics(m1, m2):
    """Compare two (256,256) label masks with values in {0,2,3}. Returns a dict.

    Dice and IoU are computed on the BINARY lesion (m > 0), which is what the model conditions
    on most strongly and what the website reports as lesion area. `exact_px_frac` is stricter
    and is computed over the full frame INCLUDING the label distinction, so a mask that got the
    outline right but the edema/core split wrong cannot hide behind a Dice of 1.0.
    """
    b1, b2 = m1 > 0, m2 > 0
    inter = int((b1 & b2).sum())
    union = int((b1 | b2).sum())
    s1, s2 = int(b1.sum()), int(b2.sum())
    return dict(
        identical     = bool(np.array_equal(m1, m2)),
        dice          = (2.0 * inter / (s1 + s2)) if (s1 + s2) else 1.0,
        iou           = (inter / union) if union else 1.0,
        exact_px_frac = float((m1 == m2).mean()),      # over all 65536 px, labels included
        disagree_px   = int((m1 != m2).sum()),
        lesion_a      = s1, lesion_b = s2,
        core_a        = int((m1 == 3).sum()), core_b = int((m2 == 3).sum()),
        edema_a       = int((m1 == 2).sum()), edema_b = int((m2 == 2).sum()),
    )


def check_end_to_end(cases, args, rep, orig_synth_mask):
    print(f"\n[3] END-TO-END SYNTH_MASK   augment_tumor.synth_mask (scipy)")
    print(f"                        vs  conddiff_core.synth_mask  (numpy port)")
    print(f"    same atlas slice, same brain mask, same seeded rng -> the mask is a pure")
    print(f"    function of (atlas, brain, seed), so any difference is a difference in the CODE.")
    print("    " + "-" * 86)

    if not cases:
        rep.skip("end_to_end", "no atlas cases", "no real slices and no phantom")
        return 0

    seeds = list(range(args.seeds))
    n_cmp = n_identical = 0
    worst = None                                   # (dice, description, metrics)
    max_lesion_frac_delta = 0.0
    total_disagree = 0
    n_shown = 0

    for case in cases:
        for seed in seeds:
            # Two FRESH generators from the same integer seed. Constructing them separately
            # (rather than reusing one) is what guarantees both functions see the same stream.
            m_ref,  c_ref  = orig_synth_mask(case["atlas"], case["brain"],
                                             np.random.default_rng(seed))
            m_port, c_port = core.synth_mask(case["atlas"], case["brain"],
                                             np.random.default_rng(seed),
                                             area_frac=core.AREA_FRAC_DEFAULT)

            tag = f"{case['source']}:{case['lobe']}:seed{seed}"

            # ---- agreement on FAILURE is part of the contract ------------------------------
            # Both return (None, None) when the lobe is too small or the drawn area is under
            # MIN_AREA. If one gives up and the other does not, the website would show a lesion
            # the paper's code refuses to draw (or a blank image where the paper draws one).
            if (m_ref is None) != (m_port is None):
                rep.fail("end_to_end", f"{tag} None-agreement",
                         f"scipy returned {'None' if m_ref is None else 'a mask'}, "
                         f"port returned {'None' if m_port is None else 'a mask'}")
                n_cmp += 1
                continue
            if m_ref is None:
                rep.ok("end_to_end", f"{tag} both-None", "both declined this slice/lobe")
                n_cmp += 1
                continue

            # ---- the canary: the seed pixel is drawn BEFORE any ported code runs -----------
            rep.check("end_to_end", f"{tag} centre", c_ref == c_port,
                      f"scipy centre {c_ref} != port centre {c_port} -- the two sides are "
                      f"consuming the rng differently; nothing else below is meaningful")

            m = _mask_metrics(m_ref, m_port)
            n_cmp += 1
            n_identical += int(m["identical"])
            total_disagree += m["disagree_px"]

            # Lesion-count delta as a FRACTION, because an absolute px tolerance means something
            # completely different for a 200 px small lesion and a 4000 px large one.
            denom = max(m["lesion_a"], 1)
            lesion_frac_delta = abs(m["lesion_a"] - m["lesion_b"]) / denom
            max_lesion_frac_delta = max(max_lesion_frac_delta, lesion_frac_delta)

            ok_dice   = m["dice"] >= args.dice_min
            ok_lesion = lesion_frac_delta <= args.lesion_tol_frac
            rep.check("end_to_end", f"{tag} dice", ok_dice,
                      f"dice={m['dice']:.6f} < {args.dice_min}")
            rep.check("end_to_end", f"{tag} lesion_px", ok_lesion,
                      f"lesion {m['lesion_a']} vs {m['lesion_b']} "
                      f"({lesion_frac_delta:.4%} > {args.lesion_tol_frac:.4%})")

            if worst is None or m["dice"] < worst[0]:
                worst = (m["dice"], tag, m)

            bad = not (ok_dice and ok_lesion and m["identical"])
            if bad or (not args.quiet and n_shown < 25):
                print(f"    {'ok ' if not bad else '!! '}{tag:<44} "
                      f"identical={str(m['identical']):<5} dice={m['dice']:.6f} "
                      f"iou={m['iou']:.6f} exact_px={m['exact_px_frac']:.6f} "
                      f"lesion {m['lesion_a']}/{m['lesion_b']} core {m['core_a']}/{m['core_b']}")
                n_shown += 1

    # ---- the verdict, stated plainly --------------------------------------------------------
    print("    " + "-" * 86)
    print(f"    {n_cmp} comparisons ({len(cases)} atlas/lobe cases x {len(seeds)} seeds)")
    if n_cmp:
        print(f"    BIT-IDENTICAL: {n_identical}/{n_cmp} ({100.0*n_identical/n_cmp:.1f}%)  "
              f"total disagreeing pixels across all comparisons: {total_disagree}")
    if worst:
        w_dice, w_tag, w = worst
        print(f"    worst case:    {w_tag}  dice={w_dice:.6f} iou={w['iou']:.6f} "
              f"exact_px={w['exact_px_frac']:.6f} disagree={w['disagree_px']} px")
        print(f"                   lesion px scipy={w['lesion_a']} port={w['lesion_b']}   "
              f"core px scipy={w['core_a']} port={w['core_b']}   "
              f"edema px scipy={w['edema_a']} port={w['edema_b']}")
    print(f"    max lesion-count difference: {max_lesion_frac_delta:.4%} "
          f"(tolerance {args.lesion_tol_frac:.4%})")

    if n_cmp and n_identical == n_cmp:
        print(f"\n    VERDICT: the port is BIT-IDENTICAL to the scipy original on every one of")
        print(f"             the {n_cmp} comparisons. Not 'close' -- the same array. The website")
        print(f"             draws exactly the lesions the paper's code draws.")
    elif n_cmp:
        # Not automatically a failure. Explain the ONE mechanism that can legitimately cause it,
        # so a reader can tell a rounding artefact from a real divergence.
        frac = 1.0 - n_identical / n_cmp
        rep.warn("end_to_end", "not bit-identical",
                 f"{n_cmp - n_identical}/{n_cmp} comparisons differed in at least one pixel")
        print(f"\n    VERDICT: the port agrees to within tolerance but is NOT bit-identical on")
        print(f"             {frac:.1%} of comparisons. The expected benign cause is that")
        print(f"             _gaussian_blur accumulates in float64 and returns float32 while")
        print(f"             scipy works in the input dtype, so the noise field differs by")
        print(f"             ~1e-7 relative. synth_mask then takes the top-N pixels by score,")
        print(f"             and a pixel whose score sits within 1e-7 of the Nth can flip sides.")
        print(f"             That moves a boundary pixel; it does not move the lesion. Judge it")
        print(f"             by 'disagree_px' above: single digits is rounding, hundreds is a bug.")
    return n_cmp


# =============================================================================================
# SECTION 7 -- CHECK 4: THE TWO COPIES OF conddiff_core.py ARE BYTE-IDENTICAL
#
# Not a numeric check, but the same class of guarantee and it costs two file reads. The whole
# design rests on the research repo and the web repo running the SAME file; if they have drifted,
# then everything proven above was proven about the cluster's copy only, and the server is
# running something else. Informational when the web repo is simply not on this machine (which is
# the normal case on Great Lakes) -- that is a SKIP, not a FAIL.
# =============================================================================================

def check_core_twins(rep):
    print(f"\n[4] SHARED-CORE BYTE IDENTITY")
    print("    " + "-" * 86)
    h_res = sha256_of(CORE_SRC)
    print(f"    research  {CORE_SRC}")
    print(f"              sha256 {h_res}")
    if not os.path.isfile(WEB_CORE_SRC):
        rep.skip("core_twins", "web repo copy", f"not present at {WEB_CORE_SRC}")
        print(f"    SKIP  web repo copy not on this machine ({WEB_CORE_SRC})")
        print(f"          Expected on the cluster. Re-run this check on the dev box before deploy.")
        return 1
    h_web = sha256_of(WEB_CORE_SRC)
    print(f"    web       {WEB_CORE_SRC}")
    print(f"              sha256 {h_web}")
    rep.check("core_twins", "conddiff_core.py identical in both repos", h_res == h_web,
              f"research={h_res[:16]}... web={h_web[:16]}... -- the two repos have DRIFTED; "
              f"copy one over the other before trusting anything above")
    if h_res == h_web:
        print(f"    ok  identical")
    return 1


# =============================================================================================
# SECTION 8 -- MAIN
# =============================================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Verify conddiff_core's scipy-free numpy ports against real scipy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--slices", default="data/processed/slices/val",
                    help="raw preprocessed split to draw real atlas channels from")
    ap.add_argument("--bank", default=None,
                    help="a conditioning bundle built by export_for_website.py; if given, its "
                         "cond_slices/ are used INSTEAD of --slices. Preferred, because these "
                         "are literally the arrays the web backend will feed to synth_mask.")
    ap.add_argument("--max-slices", type=int, default=8,
                    help="how many .npy files to open (evenly spaced across the split)")
    ap.add_argument("--max-cases", type=int, default=18,
                    help="cap on (slice, lobe) test cases; each costs --seeds mask synthesises")
    ap.add_argument("--lobes", default=None,
                    help="comma-separated lobe subset, e.g. frontal,temporal (default: all six)")
    ap.add_argument("--seeds", type=int, default=6,
                    help="rng seeds per case in the end-to-end test")
    ap.add_argument("--seed", type=int, default=0, help="base seed for the synthetic fields")
    ap.add_argument("--sigmas", default="1.0,2.5,6.0,10.0",
                    help="blur sigmas to test; NOISE_SIGMA=6.0 is the one that ships")
    ap.add_argument("--max-real-blur", type=int, default=6,
                    help="how many real atlas channels to blur-test")
    ap.add_argument("--max-real-cc", type=int, default=4,
                    help="how many real cases to build connected-component blobs from")
    # ---- tolerances, all stated up front so the exit code has a written definition ----------
    ap.add_argument("--rtol-blur", type=float, default=1e-6,
                    help="relative tolerance for the blur: max|diff| / max|scipy|. The port "
                         "returns float32, so ~1e-7 is the floor; a wrong boundary mode is "
                         "~1e-1. 1e-6 sits two orders below the smallest real error.")
    ap.add_argument("--atol-blur", type=float, default=1e-9,
                    help="absolute tolerance, used instead of rtol for near-zero fields where "
                         "the relative measure has no meaningful denominator")
    ap.add_argument("--dice-min", type=float, default=0.999,
                    help="minimum Dice between the scipy mask and the ported mask")
    ap.add_argument("--lesion-tol-frac", type=float, default=0.005,
                    help="max fractional difference in lesion pixel count")
    ap.add_argument("--require-real-data", action="store_true",
                    help="turn 'no real slices found' from a SKIP into a FAIL. Use this when "
                         "gating a deployment: a green run on phantom data alone proves the "
                         "code paths work, not that they work on BraTS.")
    ap.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = ap.parse_args()

    t0 = time.time()
    rep = Report()

    print(BANNER)
    print("VERIFY CONDDIFF PORT -- does the scipy-free numpy code compute the same thing?")
    print(BANNER)
    print(f"  repo root  : {_REPO_ROOT}")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  numpy      : {np.__version__}")
    print(f"  scipy      : {_scipy.__version__}   <- the reference implementation")
    print(f"  core       : {CORE_SRC}")

    # ---- load the original -------------------------------------------------------------------
    if not os.path.isfile(ORIGINAL_SRC):
        print(f"FATAL: {ORIGINAL_SRC} not found. There is no original to compare against.")
        sys.exit(2)
    orig_synth_mask, consts, meta = load_original_synth_mask(ORIGINAL_SRC)
    print(f"  original   : {ORIGINAL_SRC}")
    print(f"               sha256 {meta['sha256']}")
    print(f"               synth_mask extracted from lines {meta['lines'][0]}-{meta['lines'][1]} "
          f"with {meta['n_consts']} literal constants, compiled against REAL scipy")
    print(f"               (extracted via ast, so augment_tumor's torch/diffusers/matplotlib "
          f"imports never run)")

    # ---- gather real data --------------------------------------------------------------------
    cases, note = load_real_cases(args, rep)
    print(f"  data       : {note}")
    if cases:
        by_lobe = {}
        for c in cases:
            by_lobe[c["lobe"]] = by_lobe.get(c["lobe"], 0) + 1
        print(f"               {len(cases)} viable (slice, lobe) cases: "
              + ", ".join(f"{k}x{v}" for k, v in sorted(by_lobe.items())))
    else:
        msg = "no real atlas slices found on this machine"
        if args.require_real_data:
            rep.fail("data", "real slices required", msg + " and --require-real-data was set")
            print(f"               FAIL {msg} (--require-real-data)")
        else:
            rep.skip("data", "real slices", msg)
            print(f"               {msg}; falling back to a synthetic phantom atlas.")
            print(f"               This still exercises every code path, but it is NOT a")
            print(f"               deployment gate -- re-run on Great Lakes with the real split.")
        cases = [phantom_case()]

    # ---- run the checks ----------------------------------------------------------------------
    check_constants(consts, rep)
    check_gaussian(cases, args, rep)
    check_components(cases, args, rep)
    check_end_to_end(cases, args, rep, orig_synth_mask)
    check_core_twins(rep)

    # ---- summary -----------------------------------------------------------------------------
    print("\n" + BANNER)
    print("SUMMARY")
    print(BANNER)
    # One row per section, in first-seen order (dicts preserve insertion order in 3.7+), so the
    # summary reads in the same sequence the checks ran.
    per_section = {}
    for section, name, status, detail in rep.rows:
        per_section.setdefault(section, {})[status] = \
            per_section.setdefault(section, {}).get(status, 0) + 1
    for section, counts in per_section.items():
        bits = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        # A section that only skipped is reported as SKIP, not PASS: "I did not check" must not
        # read as "I checked and it was fine".
        if counts.get("FAIL"):
            verdict = "FAIL"
        elif counts.get("PASS"):
            verdict = "PASS"
        else:
            verdict = "SKIP"
        print(f"  {verdict:<5} {section:<14} {bits}")

    if rep.failed:
        print("\n  FAILURES:")
        for section, name, status, detail in rep.rows:
            if status == "FAIL":
                print(f"    [{section}] {name}: {detail}")
    warns = [r for r in rep.rows if r[2] == "WARN"]
    if warns:
        print("\n  WARNINGS (not exit-worthy, but read them):")
        for section, name, status, detail in warns:
            print(f"    [{section}] {name}: {detail}")

    total = len(rep.rows)
    print(f"\n  {rep.count('PASS')}/{total} checks passed   "
          f"({rep.count('FAIL')} failed, {rep.count('WARN')} warnings, {rep.count('SKIP')} skipped)"
          f"   in {time.time() - t0:.1f}s")

    if rep.failed:
        print("\n  RESULT: FAIL")
        # The epilogue has to name the RIGHT problem. "The port does not reproduce scipy" is a
        # very alarming sentence and it must not be printed when the only thing that failed was
        # `--require-real-data` finding no slices, or the two repos having drifted. Those are
        # real failures with completely different fixes.
        failed_sections = {r[0] for r in rep.rows if r[2] == "FAIL"}
        maths = failed_sections & {"constants", "blur", "components", "end_to_end"}
        if maths:
            print(f"  The numpy port does NOT reproduce scipy (failing: {', '.join(sorted(maths))}).")
            print("  Do NOT render the gallery and do NOT deploy: every lesion the website would")
            print("  show is a different shape from every lesion in the paper, and nothing")
            print("  downstream would notice. Fix src/conddiff_core.py, copy it over the web")
            print("  repo's twin, re-run this, and only then re-render.")
        elif "core_twins" in failed_sections:
            print("  The maths is correct, but the two copies of conddiff_core.py have DRIFTED.")
            print("  Everything proven above was proven about the RESEARCH repo's copy only; the")
            print("  web server is running different code. Copy one over the other, then re-run.")
        else:
            print("  The port itself was not shown to be wrong -- what failed was the evidence.")
            print("  See FAILURES above (typically: --require-real-data was set and no BraTS")
            print("  slices were found). Re-run on Great Lakes against the real split before")
            print("  treating this as a deployment gate.")
        print(BANNER)
        return 1

    print("\n  RESULT: PASS")
    print("  conddiff_core's scipy-free reimplementations agree with scipy.ndimage inside the")
    print("  stated tolerances, and the ported synth_mask reproduces the paper's mask synthesis.")
    if rep.count("SKIP"):
        print("  NOTE: some checks were SKIPPED (see above). A skip is not a pass -- re-run with")
        print("        --require-real-data on Great Lakes before treating this as a gate.")
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
