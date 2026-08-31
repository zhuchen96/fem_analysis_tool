import glob as _glob, re as _re

SAMPLE_ID  = 2
SPACING_XY = 0.325  # micron per pixel — straight from the microscope / acquisition metadata

MESH_DIR   = f"vtu_files_sample{SAMPLE_ID}"
TIF_DIR    = f"tif_files_sample{SAMPLE_ID}"
RESULT_DIR = f"result_sample{SAMPLE_ID}"

def _first_index(pattern):
    nums = [int(_re.search(r'(\d+)', p.rsplit('/', 1)[-1]).group(1))
            for p in _glob.glob(pattern)]
    return min(nums) if nums else 1

TIF_START = _first_index(f"{TIF_DIR}/T_*.tif")   # first TIFF file number (usually 1)
VTU_START = _first_index(f"{MESH_DIR}/sim_*.vtu") # first VTU file number (may differ, e.g. 0)

# --- setup: landmark tracking (used by setup/ scripts only) -----------
#
# Two points get template-matched backward through time, starting from a
# late frame where the neuromast/reference structure is clearly split in
# two, all the way back toward frame 1. Their midpoint becomes the
# "midline" that separates neuromast from reference tissue; the second
# point alone becomes the "apical point" that further splits the
# neuromast into a front half and a back half.
#
# To find good values: run
#     python setup/pick_seed_points.py 204
# (try a few candidate frame numbers - somewhere in the back third of the
# series usually works, once the two-lobed shape is obviously visible) and
# open the PNG it saves. It's the max-projection with a pixel grid drawn on
# top, so you can just read off two (x, y) pixel coordinates.
#
# Order matters: point 0 must be the one further toward the reference side
# (larger x), point 1 the one further toward the neuromast side (smaller
# x) - 04_build_midline.py checks this and will warn you if it looks wrong.
SEED_FRAME = 206
SEED_POINTS_XY = [
    (884, 183),  # reference-side landmark
    (601, 189),  # neuromast-side landmark -> becomes the apical point
]

# the earliest frame to trust the tracked landmarks at all. Early in a
# sequence the two points usually haven't separated enough yet for their
# midpoint to mean anything - run 03_track_landmarks.py first, look at how
# far back it actually tracked, then eyeball the early frames of that range
# (e.g. with pick_seed_points.py) and push T_START later if they still look
# too close together / jittery.
T_START = 55

# once the tracker runs out of frames (past SEED_FRAME, since it only ever
# tracks backward), the midline and apical point get extrapolated forward
# with a straight line fit to their last FIT_WINDOW tracked frames. This is
# a smoothing knob more than a dataset property - 20 is a reasonable
# default and rarely needs touching.
FIT_WINDOW = 20

# --- vector video axis limits -----------------------------------------------
#
# Each axis (x, y, z) gets its own symmetric limit for the arrow plots.
# None = auto-compute from the current sample's data.
# Set a float to lock the scale across all samples — run both samples first,
# note the printed "computed limits" line, and enter the larger value here so
# all videos share the same scale.
#
# Separate tables for the two fields since their magnitudes are very different.
AXIS_LIMITS = {
    "solution": {
        "x": None,
        "y": None,
        "z": None,
    },
    "pk1_stess_3": {
        "x": None,
        "y": None,
        "z": None,
    },
}
