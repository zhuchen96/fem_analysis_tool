"""
This is the one place that decides "which region is this mesh point in".
Everything downstream (the overlay video, the rose histograms, the vector
arrows) imports get_regions() from here instead of copy-pasting the split
logic five different times - if the split rule ever changes again we only
have to fix it in one spot.

The four regions, in terms of a point's current (deformed) x position:

    neuromast_right  - between the apical point and the midline (apical_x ≤ x < mid_x)
    neuromast_left   - same width as the right part, mirrored left of the apical point
                       (2*apical_x - mid_x ≤ x < apical_x)
    front            - the rightmost FRONT_WIDTH_UM (set in config.py) of the reference tissue,
                       anchored to the optical tissue tip (mask tip):
                           boundary = mask_tip_x − REF_FRONT_UM
                       where mask_tip_x is the rightmost foreground pixel in the
                       plain segmentation mask × SPACING_XY.
                       When the mask tip has moved beyond the FEM mesh domain
                       (mask_tip_x > mesh_tip_x), the boundary is extrapolated by
                       fitting a linear speed to all frames where mask_tip_x ≤ mesh_tip_x.
                       The same boundary is used for expanded=True and expanded=False.
    middle           - the rest of the reference (mid_x + GAP_UM ≤ x < front_boundary)

The left part is capped so it is never wider than the right part: its
left boundary is apical_x - (mid_x - apical_x) = 2*apical_x - mid_x.

The dead zone is one-sided: only on the reference side of the midline
(mid_x to mid_x + GAP_UM, ~6.5 µm). The neuromast side has no buffer —
neuromast_right extends all the way up to mid_x. There's no equivalent gap
at the apical point - that split is a plain cutoff.

A point only counts as "in" one of these regions at all if it also falls
inside the segmented embryo outline. Two flavors of that test are used
around the project: the plain segmented mask, or the same mask grown out
by 5 micron first (catches points that sit just outside the outline, which
happens more than you'd think near the tail). Pass expanded=True to
get_regions() for the second one.
"""
import numpy as np
from skimage.morphology import binary_dilation, disk

from config import RESULT_DIR, SPACING_XY, TIF_START, FRONT_WIDTH_UM

GAP_UM = 20 * SPACING_XY          # ~6.5um, the midline dead zone
REF_FRONT_UM = FRONT_WIDTH_UM     # width of the "front" tip sub-region (µm)
MARGIN_UM = 5.0
MARGIN_PX = int(round(MARGIN_UM / SPACING_XY))

_masks = np.load(f"{RESULT_DIR}/smoothed_boundary.npz")["smoothed_masks"]
IMG_H, IMG_W = _masks.shape[1:]

_mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
_mid_x_by_frame = dict(zip(_mid["frames"].astype(int).tolist(), _mid["mid_x_um"].astype(float).tolist()))

_apical = np.load(f"{RESULT_DIR}/groups_by_midline/apical_point.npz")
_apical_x_by_frame = dict(zip(_apical["frames"].astype(int).tolist(), _apical["apical_x_um"].astype(float).tolist()))

_dilated_mask_cache = {}

# Precompute mask tip x for every frame (rightmost foreground column × SPACING_XY)
_mask_tip_x = {}
for _t in _mid["frames"].astype(int):
    _mask = _masks[int(_t) - TIF_START]
    _cols = np.where(_mask.any(axis=0))[0]
    if len(_cols):
        _mask_tip_x[int(_t)] = float(_cols.max()) * SPACING_XY

# Accumulated boundaries for frames where mask_tip ≤ mesh_tip (mask-mode frames).
# Populated as get_regions() is called in frame order; used to extrapolate beyond.
_valid_boundaries = {}

# Scale factor applied to the fitted boundary speed during extrapolation.
# 1.0 = use the raw linear-fit speed; increase if the extrapolation looks too slow.
EXTRAP_SPEED_SCALE = 1.5


def foreground_mask(t, expanded):
    mask = _masks[t - TIF_START]
    if not expanded:
        return mask
    if t not in _dilated_mask_cache:
        _dilated_mask_cache[t] = binary_dilation(mask, disk(MARGIN_PX))
    return _dilated_mask_cache[t]


def get_regions(t, xy, expanded=False):
    """
    xy is an (n, 2) array of every mesh node's current position, in micron.
    Returns four arrays of node indices: neuromast_left, neuromast_right,
    front, middle. A node missing from all four is either outside the embryo
    or sitting in the gap band around the midline.

    The front/middle boundary is mask_tip_x − REF_FRONT_UM when the mask tip
    is within the FEM domain; otherwise it is extrapolated from the speed
    measured over mask-mode frames. The same boundary applies for both
    expanded=True and expanded=False.
    """
    mid_x = _mid_x_by_frame[t]
    apical_x = _apical_x_by_frame[t]

    # Mesh tip: rightmost FEM node in the reference x-range (no mask filter).
    is_ref_all = xy[:, 0] >= mid_x + GAP_UM
    mesh_tip_x = float(xy[is_ref_all, 0].max()) if is_ref_all.any() else None

    mask_tip_x = _mask_tip_x.get(t)

    # Determine front/middle boundary.
    if mask_tip_x is not None and mesh_tip_x is not None and mask_tip_x <= mesh_tip_x:
        # Mask tip is within the FEM domain: use it directly.
        front_boundary = mask_tip_x - REF_FRONT_UM
        _valid_boundaries[t] = front_boundary
    elif len(_valid_boundaries) >= 2:
        # Mask tip has moved beyond the FEM domain: extrapolate at constant speed.
        ts_arr = np.array(sorted(_valid_boundaries), dtype=float)
        bvals = np.array([_valid_boundaries[t_] for t_ in ts_arr])
        slope_raw, _ = np.polyfit(ts_arr, bvals, 1)
        slope = slope_raw * EXTRAP_SPEED_SCALE
        # Anchor to the last measured boundary so there is no discontinuity.
        last_t = float(ts_arr[-1])
        last_b = float(bvals[-1])
        front_boundary = last_b + slope * (t - last_t)
    elif len(_valid_boundaries) == 1:
        front_boundary = next(iter(_valid_boundaries.values()))
    else:
        front_boundary = None

    # Apply mask (plain or dilated) to decide which nodes count.
    col = (xy[:, 0] / SPACING_XY).round().astype(int).clip(0, IMG_W - 1)
    row = (xy[:, 1] / SPACING_XY).round().astype(int).clip(0, IMG_H - 1)
    inside = foreground_mask(t, expanded)[row, col]
    node_id = np.where(inside)[0]
    pos = xy[inside]

    is_reference = pos[:, 0] >= mid_x + GAP_UM
    is_neuromast = pos[:, 0] < mid_x
    half_width = mid_x - apical_x
    is_right = is_neuromast & (pos[:, 0] >= apical_x)
    is_left = is_neuromast & (pos[:, 0] < apical_x) & (pos[:, 0] >= apical_x - half_width)

    is_front = np.zeros(len(pos), dtype=bool)
    is_middle = np.zeros(len(pos), dtype=bool)
    if is_reference.any() and front_boundary is not None:
        is_front = is_reference & (pos[:, 0] >= front_boundary)
        is_middle = is_reference & (pos[:, 0] < front_boundary)

    return node_id[is_left], node_id[is_right], node_id[is_front], node_id[is_middle]
