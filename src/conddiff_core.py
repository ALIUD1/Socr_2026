"""conddiff_core.py -- the ONE definition of braingen_CondDiffuser_BraTS_v1's image maths.

WHY THIS FILE EXISTS
--------------------
Two completely different programs, running on two completely different machines, have to
produce the SAME image for the same four dropdown choices:

  1. the CLUSTER RENDERER (Great Lakes, GPU, 200 DDIM steps) which pre-renders the finite
     gallery of images the website ships, and
  2. the WEB BACKEND (AWS Lightsail, 0.25 vCPU / 1 GB) which serves them, and which can also
     run the model live if it is ever given a machine that can afford to.

If those two disagree by even one constant -- a different PROB_TH, a differently seeded random
field, a Gaussian blur with a different boundary mode -- then the site is advertising a
parameter it did not actually honour. The gallery would say "Frontal, Large" under a picture
that the live path would have drawn somewhere else, at a different size, and nobody would ever
see an error. That is a correctness failure, not a cosmetic one, and it is completely silent.

The defence is that BOTH programs import their maths from this file, and this file exists at
two paths that are BYTE-IDENTICAL:

    Summer2026/src/conddiff_core.py                          (research repo, cluster side)
    Brain-Image-Generator/backend/inference/conddiff_core.py  (web repo, server side)

Two copies rather than one shared package because the two repos deploy separately: the web
repo is `COPY . .`-ed into a Docker image that has no access to the research repo, and the
research repo runs inside a Slurm job that has no access to the web repo. A git submodule or a
pip package would solve it more elegantly and would also mean the deployment can no longer be
built from a single `docker build` of one repo. So: duplicate the file, and make the
duplication CHECKABLE.

    # verify the two copies are identical before trusting any rendered gallery
    sha256sum Summer2026/src/conddiff_core.py \
              Brain-Image-Generator/backend/inference/conddiff_core.py

    # PowerShell (forward slashes on purpose -- a backslash inside this docstring would be a
    # Python escape sequence, and PowerShell accepts "/" as a path separator anyway)
    Get-FileHash Summer2026/src/conddiff_core.py,
                 Brain-Image-Generator/backend/inference/conddiff_core.py -Algorithm SHA256

The cluster renderer writes its copy's sha256 into the gallery manifest, and the backend can
compare that against the sha256 of the file it is actually running. A divergence is then a
loud, mechanical fact instead of an argument about whether two files "look the same".

RULES FOR EDITING THIS FILE -- ALL THREE ARE LOAD-BEARING
---------------------------------------------------------
1. MODULE SCOPE IMPORTS numpy AND THE STANDARD LIBRARY, AND NOTHING ELSE.
   torch, diffusers, PIL and anything that reaches for a checkpoint are imported LAZILY,
   inside the one function that needs them. Reason: the web backend's image_generation.py
   imports this chain at the TOP of the file, next to the four working GAN modules. In gallery
   mode the server never runs the model at all, and it must be able to import this module
   without torch being exercised -- both because the micro instance cannot afford to page in
   ~341 MB of weights, and because an exception at import time here would take the five
   working GAN models offline with it.
2. NOTHING AT MODULE SCOPE MAY RAISE. No file reads, no env parsing that can throw, no
   asserts on the environment. Same reason as (1).
3. NO PATHS, NO os.getcwd(), NO os.environ, NO REPO LAYOUT ASSUMPTIONS.
   Paths differ between the two repos (the cluster reads /scratch, the container reads /app)
   and are the CALLER's business. Anything path-shaped belongs in conddiff_inference.py or in
   the renderer script, not here. This is what lets the same bytes import cleanly from either
   repo.

WHAT IS DELIBERATELY *NOT* IN HERE
----------------------------------
  * asset discovery, checkpoint loading, the strict=True load and the parameter-count
    assertion  -> conddiff_inference.py (web) / the renderer (cluster); both need paths
  * device selection, thread count, DDIM step count            -> deployment decisions
  * the conditioning bank manifest and slice admissibility     -> the two banks have different
    formats (CSV on the server, JSON on the cluster) and different MIN_BRAIN policies
  * Supabase publishing, the database record                   -> web only
The dividing line is: if changing it changes THE PICTURE, it lives here. If it only changes
where bytes come from or where they go, it does not.

PROVENANCE
----------
Everything numerical below is a faithful port of the research repo's scripts/augment_tumor.py
(mask synthesis, the DDIM loop, the channel indexing, the [-1,1] <-> [0,1] rescaling) and
scripts/06_captions_v2.py (the small/moderate/large and inferior/middle/superior vocabulary).
The only intentional deviations from augment_tumor.py are:
  (a) AREA_FRAC is a parameter rather than a module constant, so the website's Tumour Size
      control can drive it (AREA_FRAC_BY_SIZE below), and
  (b) scipy.ndimage.gaussian_filter and scipy.ndimage.label are reimplemented in pure numpy
      (SECTION 2), so the deployed container does not need scipy. Both reimplementations
      reproduce scipy's DEFAULTS exactly; see their docstrings for the specific defaults and
      why each one matters.
"""

# --- standard library + numpy. NOTHING ELSE AT MODULE SCOPE. See rule (1) above. -------------
import io
import numpy as np


# =============================================================================================
# SECTION 1 -- CONSTANTS
#
# Every constant here is copied verbatim from the research repo. Changing any of them changes
# the science, not the plumbing -- and changes it for BOTH the pre-rendered gallery and the
# live path, which is exactly the point of the file.
# =============================================================================================

# Lobe order is fixed by scripts/03_preprocess.py:10 in the research repo, because that is the
# order the six atlas maps were stacked into channels 3..8 of every preprocessed slice. The
# website's Lobe dropdown maps onto THIS index order; permuting it silently puts the lesion in
# the wrong lobe, with no error and a perfectly plausible-looking picture.
LOBES  = ["frontal", "parietal", "temporal", "occipital", "cerebellum", "insula"]
ATLAS0 = 3          # in the 9-channel stack, atlas lobe k lives at channel 3 + k

# --- gates that decide whether a slice can host a lesion at all (augment_tumor.py:30-32) ----
PROB_TH   = 0.4     # "inside this lobe" = atlas probability above this
MIN_LOBE  = 500     # the target lobe must be genuinely prominent in the chosen slice

# WHY 0.4 AND NOT SOMETHING ELSE: 03_preprocess.py:55 does `atlas = atlas / atlas.max()` where
# `atlas` is the stacked (6, 240, 240, 155) array. That is a SINGLE global scalar across all
# six lobes and all 155 slices -- not a per-lobe max. So only whichever lobe holds the global
# maximum ever reaches 1.0 and the others top out lower. PROB_TH is calibrated against exactly
# that scaling. A bank built with any other atlas normalisation silently changes every lesion
# size and every MIN_LOBE decision.
#
# (MIN_BRAIN -- the "is this a full mid-axial slice or a tiny inferior sliver" gate -- is
# deliberately NOT here. It is the one threshold the two sides are allowed to disagree on: it
# is COSMETIC, it is overridable by an env var on the server, and the cerebellum cells cannot
# be filled without lowering it. It lives with the bank code on each side.)

# --- tumour SIZE is RELATIVE to the lobe, not a fixed radius (augment_tumor.py:35-37) -------
AREA_FRAC_DEFAULT = (0.15, 0.35)  # lesion area as a fraction of the lobe's area IN THIS SLICE
CORE_FRAC         = 0.30          # enhancing core as a fraction of the lesion's area
MIN_AREA          = 60            # px; below this a "tumour" is too small to be meaningful

# --- tumour SHAPE is organic + atlas-conforming (augment_tumor.py:39-41) --------------------
COMPACT     = 0.95   # >1 = looser/more sprawling blob, <1 = tighter around the seed
NOISE_SIGMA = 6.0    # smoothing of the random field: larger = smoother, blobbier edges
NOISE_AMP   = 0.9    # 0 = perfect ellipse, ~1 = strongly irregular outline

# --- the paper's published vocabulary (scripts/06_captions_v2.py:42,45) ---------------------
# 1 px = 1 mm^2 at BraTS's 1 mm isotropic resolution, so these thresholds are ~mm^2.
SMALL_MAX, LARGE_MIN       = 500, 2000
INFERIOR_MAX, SUPERIOR_MIN = 60, 100

# --- mapping the website's four dropdowns onto the algorithm --------------------------------
# Note the web repo's existing convention, established at cgan_inference_v2.py:19: the params
# KEY is British ("tumour") but the VALUE is American ("With Tumor"). These dicts are keyed by
# the EXACT strings the frontend POSTs, so they must not be "tidied".
UI_TUMOUR = {"With Tumor": True, "Without Tumor": False}
UI_LOBE   = {"Frontal": "frontal", "Parietal": "parietal", "Temporal": "temporal",
             "Occipital": "occipital", "Cerebellum": "cerebellum", "Insula": "insula"}
UI_LEVEL  = {"Inferior": "inferior", "Middle": "middle", "Superior": "superior"}
UI_SIZE   = {"Small": "small", "Moderate": "moderate", "Large": "large"}

# The canonical orderings, derived from the dicts above rather than retyped, so a new dropdown
# option cannot be added in one place and forgotten in the other. Used by iter_cells() to
# enumerate the gallery and by the renderer to lay out its progress table.
LEVELS = ["inferior", "middle", "superior"]
SIZES  = ["small", "moderate", "large"]

# Tumour Size modulates AREA_FRAC -- the fraction of the LOBE's in-slice area the lesion
# occupies. Keeping it relative (rather than an absolute pixel target) is what makes the
# control robust: a small lobe simply cannot host an oversized lesion, so the hard containment
# constraint in synth_mask can never be violated by an ambitious size request.
# "moderate" is the paper's published default, AREA_FRAC = (0.15, 0.35).
AREA_FRAC_BY_SIZE = {
    "small":    (0.05, 0.12),
    "moderate": (0.15, 0.35),
    "large":    (0.40, 0.60),
}


# =============================================================================================
# SECTION 2 -- SCIPY-FREE REIMPLEMENTATIONS OF THE TWO scipy.ndimage CALLS
#
# augment_tumor.py:21 imports gaussian_filter and label from scipy.ndimage. Those are the ONLY
# two scipy calls anywhere on the inference path. Rather than add scipy to the web container
# (~35 MB, plus a dependency-resolver run against numpy==1.24.3 in a shared image that five
# working models depend on) they are reimplemented here in pure numpy.
#
# THE REASON THEY LIVE IN THE *SHARED* FILE RATHER THAN ONLY IN THE BACKEND: if the cluster
# renderer kept calling scipy while the backend called these, the pre-rendered gallery and the
# live path would be computing two different random fields from the same seed. The images
# would both look fine and they would not be the same image. So the cluster gives up scipy too
# and both sides run this code. Both functions are written to reproduce scipy's DEFAULTS
# exactly, which is what makes that substitution safe.
# =============================================================================================

def _gaussian_blur(img, sigma):
    """Isotropic Gaussian blur == scipy.ndimage.gaussian_filter(img, sigma) with defaults.

    WHY IT IS ONLY ~15 LINES: a 2-D Gaussian is SEPARABLE, i.e.
        G_2d(x, y) = G_1d(x) * G_1d(y)
    so blurring along axis 0 and then along axis 1 with the same 1-D kernel gives exactly the
    same result as a true 2-D convolution, at O(2r) work per pixel instead of O(r^2).

    Matching scipy's defaults, which is the whole point of writing this by hand:
      * radius = int(truncate * sigma + 0.5) with truncate = 4.0  -> for sigma=6 that is 24,
        i.e. a 49-tap kernel. Four standard deviations captures >99.99% of the mass.
      * kernel weights exp(-0.5 * (x/sigma)^2), then normalised to sum to 1 (so a constant
        image blurs to itself and the output keeps the input's overall scale).
      * boundary handling `mode='reflect'`, which in scipy means (d c b a | a b c d) -- the
        edge sample is DUPLICATED. numpy's np.pad calls that 'symmetric'; numpy's own
        'reflect' is scipy's 'mirror' and is NOT the same thing. Getting this backwards
        changes the outermost ~24 pixels, which for us is the zero frame around the brain, so
        it would be invisible in a picture and wrong in the numbers.

    img   : (H, W) float array
    sigma : float, standard deviation in pixels
    returns (H, W) float32
    """
    radius = int(4.0 * sigma + 0.5)                              # 24 for sigma = 6.0
    x = np.arange(-radius, radius + 1, dtype=np.float64)         # (2r+1,) tap offsets
    k = np.exp(-0.5 * (x / sigma) ** 2)                          # (2r+1,) unnormalised weights
    k /= k.sum()                                                 # normalise -> sums to 1.0

    out = np.asarray(img, dtype=np.float64)                      # (H, W); accumulate in f64
    for axis in (0, 1):                                          # separable: one pass per axis
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (radius, radius)
        padded = np.pad(out, pad_width, mode="symmetric")        # (H+2r, W) then (H, W+2r)
        acc = np.zeros_like(out)                                 # (H, W)
        n = out.shape[axis]
        for i, w in enumerate(k):                                # slide the kernel, tap by tap
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + n)                           # window of length n at offset i
            acc += w * padded[tuple(sl)]
        out = acc
    return out.astype(np.float32)                                # (H, W) float32


def _largest_connected_component(blob):
    """Keep only the largest 4-connected component of a boolean mask.

    Replaces the scipy idiom at augment_tumor.py:95-98:
        lab, n = label(blob)
        if n > 1:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            blob = lab == sizes.argmax()

    WHY 4-CONNECTIVITY: scipy.ndimage.label's default structuring element is
    generate_binary_structure(2, 1), the plus-shape -- pixels touching only at a corner are
    NOT the same component. We use the same four neighbours (up, down, left, right).

    WHY THIS IS DONE WITH AN EXPLICIT STACK RATHER THAN VECTORISED numpy: the loop only ever
    visits pixels that are True, and the caller has already capped that at n_les (a few
    hundred to a few thousand pixels), so this is microseconds. A vectorised
    label-propagation would need O(blob diameter) full-array passes and is both slower and
    harder to read.

    NOTE ON TIE-BREAKING, because it is the kind of thing that quietly diverges: when two
    components are exactly the same size we keep the FIRST one encountered, and the encounter
    order is np.nonzero's row-major order. `>` (not `>=`) in the comparison below is what
    fixes that. scipy's own label+argmax idiom also keeps the lowest label, which is likewise
    row-major first, so the two agree on ties as well as on the common case.

    blob : (H, W) bool
    returns (H, W) bool -- the single largest component (or an all-False array if blob is empty)
    """
    h, w = blob.shape
    seen = np.zeros((h, w), dtype=bool)          # pixels already assigned to some component
    best = np.zeros((h, w), dtype=bool)          # the biggest component found so far
    best_size = 0

    ys, xs = np.nonzero(blob)                    # only True pixels can start a component
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        # --- flood fill this component with an explicit LIFO stack (no recursion: Python's
        # --- default recursion limit is 1000 and a lesion can be thousands of pixels) -------
        stack = [(sy, sx)]
        seen[sy, sx] = True
        component = [(sy, sx)]
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and blob[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
                    component.append((ny, nx))
        if len(component) > best_size:
            best_size = len(component)
            best = np.zeros((h, w), dtype=bool)
            cy_arr = np.fromiter((p[0] for p in component), dtype=np.intp, count=len(component))
            cx_arr = np.fromiter((p[1] for p in component), dtype=np.intp, count=len(component))
            best[cy_arr, cx_arr] = True
    return best


# =============================================================================================
# SECTION 3 -- THE PAPER'S VOCABULARY AND THE UI -> ALGORITHM MAPPING
# =============================================================================================

def size_word(n):
    """Paper's Small/Moderate/Large buckets by lesion AREA in px (06_captions_v2.py:57)."""
    return "small" if n < SMALL_MAX else ("large" if n > LARGE_MIN else "moderate")


def level_word(z):
    """Paper's Inferior/Middle/Superior buckets by slice index z (06_captions_v2.py:61)."""
    return "inferior" if z < INFERIOR_MAX else ("superior" if z > SUPERIOR_MIN else "middle")


def params_to_spec(tumour, lobe, slice_location, tumour_size):
    """Turn the four raw dropdown strings into a validated generation specification.

    THIS IS THE FUNNEL. Both the cluster renderer and the web backend go through it, so the
    renderer's idea of "cell (Frontal, Middle, Large)" and the backend's idea of the same
    request are the same object, produced by the same code. cell_id() below then names it.

    WHY THIS IS TOLERANT RATHER THAN STRICT: the frontend has five separate registration sites
    and it is entirely possible to ship a build where the model is selectable but one of its
    params is still an empty string. Every other model in the web backend does bare
    `params["tumour"]` indexing, which raises KeyError -> image_generation.py catches it ->
    the API returns HTTP 200 with an empty list -> the user sees a green "Success" toast and a
    blank viewer. Substituting a documented default and RECORDING the substitution is strictly
    more useful than that silence.

    Returns a dict:
        with_tumour : bool
        lobe        : str | None   -- canonical lowercase lobe name, None when no tumour
        lobe_idx    : int  | None  -- index into LOBES, so the atlas channel is ATLAS0+lobe_idx
        level       : str          -- "inferior" | "middle" | "superior"
        size        : str  | None  -- "small" | "moderate" | "large", None when no tumour
        substituted : list[str]    -- names of any fields that fell back to a default
    """
    substituted = []

    with_tumour = UI_TUMOUR.get(tumour)
    if with_tumour is None:
        with_tumour = True                                 # the site's own default
        substituted.append("tumour")

    level = UI_LEVEL.get(slice_location)
    if level is None:
        # sliceLocation is shared state with the cGAN models, whose Sagittal/Coronal
        # orientations can leave it at "Left"/"Anterior"/etc. Those are meaningless here (this
        # model is axial-only), so coerce to the middle of the brain rather than fail.
        level = "middle"
        substituted.append("slice_location")

    if not with_tumour:
        # Lobe and size are genuinely meaningless without a tumour: there is no lesion to
        # place and nothing to size. Return None rather than a misleading default so the
        # recorded params cannot claim we honoured a control we ignored. This is also what
        # collapses the parameter space from 2*6*3*3 = 108 to 57 real cells (see iter_cells).
        return {"with_tumour": False, "lobe": None, "lobe_idx": None,
                "level": level, "size": None, "substituted": substituted}

    lobe_key = UI_LOBE.get(lobe)
    if lobe_key is None:
        lobe_key = "frontal"
        substituted.append("lobe")

    size_key = UI_SIZE.get(tumour_size)
    if size_key is None:
        size_key = "moderate"
        substituted.append("tumour_size")

    return {"with_tumour": True, "lobe": lobe_key, "lobe_idx": LOBES.index(lobe_key),
            "level": level, "size": size_key, "substituted": substituted}


# =============================================================================================
# SECTION 4 -- CELL IDENTITY: THE NAME A REQUEST AND A PRE-RENDERED IMAGE SHARE
#
# The whole gallery design rests on one observation: the user-visible parameter space is
# FINITE and small. Conditioning does not come from the user, it comes from a fixed shipped
# bank, so the only things a visitor can vary are the four dropdowns. That is
#
#     Tumour(2) x Lobe(6) x Level(3) x Size(3) = 108 nominal combinations
#
# but Lobe and Size are meaningless when Tumour = "Without Tumor" (there is no lesion to place
# and nothing to size -- params_to_spec returns None for both), so those 54 nominal
# combinations collapse into 3 real ones, one per level. The real space is
#
#     6*3*3 = 54 with-tumour cells  +  3 without-tumour cells  =  57 cells
#
# 57 cells x K seed variants of a 256x256 grayscale PNG is a few megabytes. Rendering them
# once on a GPU at the paper's full 200 steps and shipping the pictures is strictly better
# than rendering them live on 0.25 vCPU at a degraded 50 steps -- better images, no torch at
# request time, no 341 MB checkpoint in a 1 GB container, and no multi-minute request blocking
# the uvicorn event loop while the health check times out.
#
# For that to work the renderer and the server must agree on the NAME of each cell, forever
# and across machines. That is this function.
# =============================================================================================

def cell_id(spec):
    """Stable filename-safe key for one cell of the parameter space.

    FORMAT (this is a contract; the gallery on disk is keyed by these strings):

        with tumour     "with_<lobe>_<level>_<size>"     e.g. "with_frontal_middle_moderate"
        without tumour  "without_<level>"                e.g. "without_middle"

    where <lobe> is one of LOBES, <level> one of LEVELS and <size> one of SIZES -- all
    lowercase ASCII, all already canonicalised by params_to_spec. The pieces are joined with
    "_" and no piece contains "_", so the string also parses back unambiguously by splitting.

    PROPERTIES THAT MATTER, AND WHY:
      * PURE. It is a function of the four user choices and nothing else -- no seed, no
        timestamp, no bank filename, no dict iteration order, no hash of anything. Rendering
        the gallery twice, on two machines, in two years, produces the same 57 names. A
        content hash (say sha256 of the spec) would also be stable but would be unreadable in
        a directory listing and would change if a field were ever added to the spec dict.
      * TOTAL over the space the UI can express: params_to_spec canonicalises unknown input to
        a default rather than raising, so every request maps to exactly one of the 57 names.
      * FILENAME-SAFE on every platform (lowercase letters and underscores only), because it
        is used directly as a directory name on Windows, Linux and inside the container, and
        as a JSON object key in the gallery manifest.
      * The "without" form deliberately OMITS lobe and size rather than writing
        "without_none_middle_none". Omitting them is the honest encoding: those controls were
        not honoured, so they are not in the name, and the same three files are correctly
        reused no matter what the greyed-out dropdowns happened to be showing.

    spec : the dict returned by params_to_spec()
    returns str
    """
    if not spec["with_tumour"]:
        return "without_{}".format(spec["level"])
    return "with_{}_{}_{}".format(spec["lobe"], spec["level"], spec["size"])


def iter_cells():
    """Enumerate all 57 cells, in a fixed order, as (cell_id, ui_choices) pairs.

    WHY IT IS HERE AND NOT IN THE RENDERER: the renderer uses it to decide what to draw and
    the backend uses it to report which cells the shipped gallery is missing. If each side had
    its own loop they could disagree about how many cells exist -- which is precisely the
    "advertises a parameter it did not honour" failure this file was created to prevent.

    `ui_choices` is a dict of the ORIGINAL dropdown strings (American-spelled values, exactly
    what the frontend POSTs), so the renderer can feed them straight back through
    params_to_spec and the manifest can record what a cell means without decoding its name.

    Order: without-tumour cells first (3, by level), then with-tumour (54, lobe-major, then
    level, then size). Fixed order = a diffable manifest and a deterministic render log.

    yields (str, dict)
    """
    # Reverse the UI dicts once so we can go canonical -> the exact string the frontend sends.
    lobe_ui  = {v: k for k, v in UI_LOBE.items()}
    level_ui = {v: k for k, v in UI_LEVEL.items()}
    size_ui  = {v: k for k, v in UI_SIZE.items()}

    for level in LEVELS:
        # Lobe and size are recorded as the site's defaults purely so the dict has all four
        # keys; params_to_spec throws them away for "Without Tumor" and cell_id never sees
        # them. Do NOT read meaning into them.
        yield ("without_{}".format(level),
               {"tumour": "Without Tumor", "lobe": "Frontal",
                "slice_location": level_ui[level], "tumour_size": "Moderate"})

    for lobe in LOBES:
        for level in LEVELS:
            for size in SIZES:
                yield ("with_{}_{}_{}".format(lobe, level, size),
                       {"tumour": "With Tumor", "lobe": lobe_ui[lobe],
                        "slice_location": level_ui[level], "tumour_size": size_ui[size]})


# =============================================================================================
# SECTION 5 -- THE ATLAS-GUIDED MASK SYNTHESISER
# Ported from scripts/augment_tumor.py:43-110. The algorithm is unchanged; the only edits are
# (a) AREA_FRAC becomes a parameter so the website's Tumour Size control can drive it, and
# (b) the two scipy calls are replaced by the numpy versions in SECTION 2.
# =============================================================================================

def synth_mask(atlas_lobe, brain, rng, area_frac=AREA_FRAC_DEFAULT):
    """Grow an organic tumour that CONFORMS to the lobe and is SIZED relative to it.

    Instead of stamping a fixed circle, we score every pixel by
        atlas probability  x  compactness-around-a-seed  x  smooth random field
    and take the top-N pixels, where N is a fraction of the lobe's area in this slice.
    The atlas term makes the lesion hug the lobe's real shape; the random field makes the
    outline irregular like a real tumour; the compactness term keeps it a single mass.

    atlas_lobe : (256,256) float32, this slice's probability map for the CHOSEN lobe, in [0,1]
    brain      : (256,256) bool, the brain footprint -- the caller computes it from the T1
                 channel as `stack[1] > 0.05`, NOT from FLAIR and NOT from the mask
    rng        : numpy.random.Generator; the entire shape is a pure function of its state
    area_frac  : (lo, hi) fraction-of-lobe-area range to draw the target lesion size from

    Returns (mask, (c0, c1)) or (None, None) on any of three failure conditions.
    `mask` is (256,256) float32 whose values are ONLY {0, 2, 3}:
        0 = background, 2 = edema (the bright FLAIR halo), 3 = enhancing tumour core.
    Label 1 (necrotic) is NEVER emitted -- synthetic lesions are edema+core only, unlike real
    BraTS masks. These are RAW BraTS label values: channel 2 of the stack is the one channel
    that is NOT in [0,1] (03_preprocess.py:72 puts the segmentation in unnormalised), so
    rescaling it to [0,1] would put the model badly off-distribution.
    """
    lobe = (atlas_lobe > PROB_TH) & brain            # the lobe's footprint in THIS slice
    n_lobe = int(lobe.sum())
    if n_lobe < MIN_LOBE:
        # The atlas may claim this lobe, but the patient's slice does not contain enough of
        # it to host a believable lesion. Caller decides what to do.
        return None, None

    # --- 1. seed: a point well inside the lobe (sampled by atlas probability) ---------------
    # NOT uniform. Weighting by the atlas probability makes seeds nucleate in the lobe's core
    # rather than on its ragged margin, so the lesion does not habitually straddle a boundary.
    ys, xs = np.nonzero(lobe)                                  # (n_lobe,), (n_lobe,)
    w = atlas_lobe[ys, xs].astype(np.float64)                  # float64 and renormalised
    w = w / w.sum()                                            # because rng.choice(p=...)
    j = int(rng.choice(len(ys), p=w))                          # demands an EXACT sum of 1
    c0, c1 = int(ys[j]), int(xs[j])                            # seed pixel (axis0, axis1)

    # --- 2. target size: a fraction of THIS lobe's area -> big lobe, big tumour -------------
    n_les = int(rng.uniform(*area_frac) * n_lobe)
    if n_les < MIN_AREA:
        return None, None

    # --- 3. score field = atlas x compactness x irregularity --------------------------------
    idx0 = np.arange(256, dtype=np.float32)[:, None]      # (256,1) row index
    idx1 = np.arange(256, dtype=np.float32)[None, :]      # (1,256) col index
    dist = np.sqrt((idx0 - c0) ** 2 + (idx1 - c1) ** 2)   # (256,256) distance from the seed
    r_eq = np.sqrt(n_les / np.pi)                         # radius a circle of that area would have
    compact = np.exp(-(dist / (COMPACT * r_eq)) ** 2)     # smooth falloff -> keeps it one mass

    # A smoothed random field is what makes the outline organic. sigma=6 px sets the
    # wavelength of the wobble; NOISE_AMP=0.9 scales it to a ~[0.55, 1.45] multiplier, so it
    # perturbs the boundary without ever being able to flip the sign of the score.
    noise = _gaussian_blur(rng.standard_normal((256, 256)).astype(np.float32), NOISE_SIGMA)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)   # -> [0,1]
    irregular = 1.0 + NOISE_AMP * (noise - 0.5)           # ~[0.55,1.45] multiplier

    score = atlas_lobe.astype(np.float32) * compact * irregular   # (256,256)
    score[~lobe] = 0.0                                    # HARD constraint: never leave the lobe

    # --- 4. take the top-N scoring pixels = the lesion --------------------------------------
    # argpartition is O(n) (it only guarantees the top n_les are in the last n_les slots, in no
    # particular order), which is all we need and much cheaper than a full sort. Taking a fixed
    # COUNT rather than thresholding the score means the area is exact by construction instead
    # of being whatever a threshold happened to produce.
    flat = score.ravel()                                  # (65536,)
    n_les = min(n_les, int((flat > 0).sum()))             # cannot want more pixels than exist
    if n_les < MIN_AREA:
        return None, None
    top = np.argpartition(flat, -n_les)[-n_les:]          # indices of the N highest scores
    blob = np.zeros(256 * 256, bool); blob[top] = True
    blob = blob.reshape(256, 256)                         # (256,256) bool

    # --- 5. keep only the largest connected piece (no scattered islands) ---------------------
    # The random field can carve the top-N set into a main mass plus a few specks. A real
    # lesion is one object. NOTE this step can shrink the lesion BELOW the requested n_les --
    # discarded islands are not replaced -- which is why the realised area is measured and
    # reported rather than assumed.
    blob = _largest_connected_component(blob)

    # --- 6. core = the highest-scoring pixels INSIDE the lesion ------------------------------
    m = np.zeros((256, 256), np.float32)
    m[blob] = 2                                           # label 2 = edema (bright FLAIR halo)
    inside = score * blob                                 # scores, zeroed outside the lesion
    n_core = max(1, int(CORE_FRAC * blob.sum()))
    core_idx = np.argpartition(inside.ravel(), -n_core)[-n_core:]
    core = np.zeros(256 * 256, bool); core[core_idx] = True
    # `& blob` re-guards the assignment: argpartition breaks ties arbitrarily and could hand
    # back an index whose score is 0 outside the lesion.
    m[core.reshape(256, 256) & blob] = 3                  # label 3 = enhancing tumour core

    m[~brain] = 0                                         # never spill outside the brain

    # Cheap invariant, expensive bug: the model was trained on raw BraTS labels and channel 2
    # is the only channel not in [0,1]. If this ever fails, something rescaled the mask.
    assert set(np.unique(m).tolist()) <= {0.0, 2.0, 3.0}, "mask must carry only labels {0,2,3}"
    return m, (c0, c1)


def synth_mask_sized(atlas_lobe, brain, rng, size_key, max_tries=12):
    """synth_mask with the website's Small/Moderate/Large control wired in.

    Two things are going on here.

    (1) THE PRIMARY MAPPING is size -> AREA_FRAC band (AREA_FRAC_BY_SIZE). This keeps the
        paper's relative-to-lobe design, which is what guarantees the lesion always fits
        inside the lobe.

    (2) A BOUNDED REJECTION LOOP then tries to make the website's word mean the same thing the
        paper's captions mean. 06_captions_v2.py defines small/moderate/large by ABSOLUTE
        area in px (<500 / 500-2000 / >2000), so we redraw up to `max_tries` times until the
        REALISED area lands in that band. If it never does -- which happens legitimately, e.g.
        a large lesion cannot fit in a small lobe -- we keep the attempt whose area was
        closest and set size_in_band=False. An honest recorded mismatch is much better than a
        500 error, and much better than silently mislabelling.

    THE `stats` DICT IS PART OF THE HONESTY CONTRACT. Whatever comes back here is what the web
    backend records in the database as what actually happened, and it is what the cluster
    renderer writes into the gallery manifest so that a pre-rendered image can still report
    its true realised lesion area months later. Do not add a field here that the caller cannot
    verify, and do not remove one -- the gallery manifest schema mirrors it.

    Returns (mask, (c0,c1), stats). Raises ValueError only if EVERY attempt returned
    (None, None), i.e. the lobe is genuinely absent from this slice -- which the caller's
    slice-admissibility check should already have excluded, so it means the conditioning bank
    and its manifest disagree.
    """
    lo_hi = AREA_FRAC_BY_SIZE[size_key]

    # The paper's absolute band for this word, as a half-open interval in pixels.
    if size_key == "small":
        band = (MIN_AREA, SMALL_MAX)          # [60, 500)
    elif size_key == "large":
        band = (LARGE_MIN + 1, np.inf)        # (2000, inf)
    else:
        band = (SMALL_MAX, LARGE_MIN + 1)     # [500, 2000]

    best = None          # (distance_from_band, mask, centre, lesion_px)
    tries = 0
    for tries in range(1, max_tries + 1):
        m, centre = synth_mask(atlas_lobe, brain, rng, area_frac=lo_hi)
        if m is None:
            continue
        lesion_px = int((m > 0).sum())
        if band[0] <= lesion_px < band[1]:
            best = (0.0, m, centre, lesion_px)
            break
        # Distance to the nearest edge of the band, so "closest attempt" is well defined.
        dist = band[0] - lesion_px if lesion_px < band[0] else lesion_px - (band[1] - 1)
        if best is None or dist < best[0]:
            best = (float(dist), m, centre, lesion_px)

    if best is None:
        raise ValueError(
            "could not grow a synthetic lesion on this slice -- the chosen lobe is not "
            "sufficiently present in it. This means the conditioning bank's manifest "
            "disagrees with the slice data; rebuild the bank."
        )

    dist, m, centre, lesion_px = best
    n_lobe = int(((atlas_lobe > PROB_TH) & brain).sum())
    stats = {
        "lesion_px":      lesion_px,
        "core_px":        int((m == 3).sum()),
        "lobe_px":        n_lobe,
        # what fraction of the lobe the lesion actually ended up covering
        "coverage":       round(lesion_px / n_lobe, 4) if n_lobe else 0.0,
        "centre_axis0":   int(centre[0]),
        "centre_axis1":   int(centre[1]),
        "tries":          int(tries),
        "size_requested": size_key,
        "size_realized":  size_word(lesion_px),
        "size_in_band":   bool(dist == 0.0),
    }
    return m, centre, stats


# =============================================================================================
# SECTION 6 -- THE CONDITIONING TENSOR AND THE DDIM REVERSE LOOP
#
# torch is imported INSIDE these two functions, never at module scope. In gallery mode the web
# backend imports this file and never calls either of them, so the container never pays for
# torch -- see rule (1) in the module docstring.
# =============================================================================================

def build_cond(stack, mask, device="cpu"):
    """Assemble the 8-channel conditioning tensor the U-Net expects.

    THE OFF-BY-ONE THAT BITES EVERYONE: the model's input is
        torch.cat([noisy_flair, cond], dim=1)
    where `cond = stack[1:9]`. That slice re-indexes everything by -1, so INSIDE cond:

        cond index 0  ==  stack channel 1  ==  T1
        cond index 1  ==  stack channel 2  ==  tumour mask     <- NOT cond[2]
        cond index 2+k == stack channel 3+k == atlas lobe k

    This is why sample.py:29 and eval_grid.py:51 in the research repo read the mask as cond[1].
    After the concatenation the model's own channel numbering lines back up with the stack's
    (0=noisy FLAIR, 1=T1, 2=mask, 3..8=atlas), which is what in_channels=9 was trained on.

    stack  : (9,256,256) float32, straight from the conditioning bank
    mask   : (256,256) float32 with labels {0,2,3}, or an all-zero array for "Without Tumor"
    device : torch device string -- "cpu" on the web container (torch==2.0.1+cpu), "cuda" on
             Great Lakes. Passed in rather than decided here because device selection is a
             DEPLOYMENT decision and this file must not make any (rule 3), and because reading
             torch.cuda.is_available() at module scope would import torch at module scope.

    Returns a torch tensor of shape (1,8,256,256) on `device`.

    NOTE THE CONDITIONING IS NOT RESCALED. train.py:33-34 maps only the FLAIR TARGET to
    [-1,1] (`target = target * 2 - 1`); the conditioning gets `cond = cond.to(DEVICE)` with no
    arithmetic at all. Feeding a 2*cond-1 tensor here would be a silent, extremely
    plausible-looking bug -- the output would still look like a brain, just a worse one.
    """
    import torch                                           # lazy: see the section header

    out = stack[1:9].copy()                                # (8,256,256) -- copy, never mutate
    out[1] = mask.astype(np.float32)                       # cond index 1 == stack ch 2 == mask
    assert out.shape == (8, 256, 256) and out.dtype == np.float32
    return torch.from_numpy(out).unsqueeze(0).to(device)   # (1,8,256,256)


def render_flair(model, sched, cond, seed):
    """The DDIM reverse-diffusion loop. This is where the image is actually made.

    Transcribed from augment_tumor.py:151-156 with the shapes spelled out.

    The idea: start from pure Gaussian noise in the FLAIR channel, and repeatedly ask the
    network "given this noisy image and this conditioning, what noise do you think is in
    here?", then have the scheduler subtract a calibrated share of that prediction. After
    `steps` iterations the noise is gone and a FLAIR is left.

    model : the UNet2DModel, in eval mode, already on the same device as `cond`
    sched : the DDIMScheduler, already set_timesteps(steps)
    cond  : (1,8,256,256) float32 -- CONSTANT for the whole loop. Its .device is what the
            initial noise is drawn on, so there is no device argument to get out of sync with
            it; a mismatch would be a torch error rather than a wrong picture.
    seed  : int; seeds the initial noise so a generation is exactly reproducible

    Returns (256,256) float32 in [0,1].

    STEP COUNT IS THE CALLER'S: it is baked into `sched` by set_timesteps(). The cluster
    renderer uses the paper's 200; a live web deployment would use fewer. That is the ONE
    quality knob the two paths are allowed to differ on, and it is recorded per image
    (params["ddim_steps"]) precisely so the difference is never silent.
    """
    import torch                                           # lazy: see the section header

    # One seed in, both randomness sources derived from it. augment_tumor.py gets this half
    # right and half wrong for a web service: line 116 hard-seeds the MASK rng to 0 (so every
    # call would draw the identical tumour) while line 151 leaves the diffusion noise
    # UNSEEDED (so no image can ever be reproduced). Here the caller derives the numpy
    # Generator and this torch Generator from the same integer, and that integer is recorded
    # with the image -- which is what makes any image on the site regenerable.
    device = cond.device
    g = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(cond.shape[0], 1, 256, 256, generator=g, device=device)   # (1,1,256,256)

    for t in sched.timesteps:                              # 0-dim int64 tensors, descending
        with torch.no_grad():                              # no autograd graph -> less memory
            model_in = torch.cat([x, cond], dim=1)         # (1,1,..) + (1,8,..) -> (1,9,256,256)
            # `.sample` unwraps the UNet2DOutput dataclass -- model(...) does NOT return a
            # raw tensor. Same idea for `.prev_sample` on DDIMSchedulerOutput below.
            npred = model(model_in, t).sample               # (1,1,256,256) predicted epsilon
        x = sched.step(npred, t, x).prev_sample             # (1,1,256,256) one step less noisy

    # Undo training's target scaling. train.py:33 does `target = target * 2 - 1` to map the
    # [0,1] FLAIR into [-1,1] for the model, so the exact inverse is (x + 1) / 2. The clip
    # catches the small overshoot DDIM can leave at the extremes.
    return np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1).astype(np.float32)   # (256,256)


# =============================================================================================
# SECTION 7 -- PNG ENCODING
#
# This is in the shared file for the same reason everything else is: the gallery PNG is
# encoded ONCE, on the cluster, and served for months. If the live path encoded with a
# different orientation or a different intensity mapping, the same request would produce a
# visibly different picture depending on which mode the server happened to be in.
# =============================================================================================

def to_png_bytes(arr, kind="flair"):
    """Encode a (256,256) array as PNG bytes, in the research repo's display orientation.

    ORIENTATION. Every figure in the research repo is drawn as
        ax.imshow(arr.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    (sample.py:47, eval_grid.py:84, augment_tumor.py:165, check_atlas_fit.py:79). `origin=
    "lower"` puts row 0 at the BOTTOM, whereas PIL puts row 0 at the top -- so the equivalent
    array for PIL is np.flipud(arr.T). Skipping this yields a 90-degree-rotated brain relative
    to every figure in the paper. Array axis 0 is therefore the image's horizontal axis and
    axis 1 the vertical. (Related, if a Left/Right control is ever added: the convention is
    radiological, MIDLINE_AXIS0 = 128 and image-left is patient-RIGHT.)
    KNOWN DIVERGENCE: the four GAN models on the site hand raw arrays to PIL with no such
    transform, so this model's output will be oriented differently from theirs in the same
    viewer. That is a deliberate choice in favour of matching the paper; it is worth a note in
    the model's UI copy.

    INTENSITY. We return BYTES, not a numpy array, and that is load-bearing.
    supabase_storage.py:87-95 takes the numpy branch and does a PER-IMAGE MIN-MAX STRETCH
    before encoding, which destroys absolute FLAIR intensity and can exaggerate or flatten
    lesion contrast unpredictably from one generation to the next. Passing bytes takes the
    `else` branch at supabase_storage.py:96-98, which uploads them untouched. So the site shows
    exactly the fixed vmin=0, vmax=1 mapping the paper uses, and two generations are
    comparable. Do not "simplify" this into passing an array.

    arr  : (256,256) float32
    kind : "flair" -> [0,1] intensities mapped linearly onto 0..255
           "seg"   -> BraTS labels {0,2,3} mapped onto {0,170,255} by an explicit lookup, so
                      the mask renders identically whether or not a core is present (a min-max
                      stretch of a core-less mask would map edema to pure white).

    Returns PNG bytes. The cluster renderer writes them straight to a .png file; the web
    backend hands them to Supabase. Same bytes either way.
    """
    from PIL import Image                                  # lazy, to keep module scope numpy-only

    a = np.asarray(arr)
    if kind == "seg":
        out = np.zeros(a.shape, dtype=np.uint8)
        lab = np.rint(a).astype(np.int16)                  # labels are stored as floats
        out[lab == 2] = 170                                # edema      -> mid grey
        out[lab == 3] = 255                                # enhancing  -> white
    else:
        out = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)

    # np.ascontiguousarray because PIL needs a C-contiguous buffer and .T produces a view with
    # transposed strides.
    img = Image.fromarray(np.ascontiguousarray(np.flipud(out.T)), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
