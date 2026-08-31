"""
This is the one place that decides "which region is this mesh point in".
Everything downstream (the overlay video, the rose histograms, the vector
arrows) imports get_regions() from here instead of copy-pasting the split
logic five different times - if the split rule ever changes again we only
have to fix it in one spot.

The three regions, in terms of a point's current (deformed) x position:

    neuromast_right  - between the apical point and the midline (apical_x ≤ x < mid_x)
    neuromast_left   - same width as the right part, mirrored left of the apical point
                       (2*apical_x - mid_x ≤ x < apical_x)
    reference        - behind the midline, starting at mid_x + GAP_UM

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

from config import RESULT_DIR, SPACING_XY, TIF_START

GAP_UM = 20 * SPACING_XY  # ~6.5um, the midline dead zone (20px, chosen early on and kept ever since)
MARGIN_UM = 5.0
MARGIN_PX = int(round(MARGIN_UM / SPACING_XY))

_masks = np.load(f"{RESULT_DIR}/smoothed_boundary.npz")["smoothed_masks"]
IMG_H, IMG_W = _masks.shape[1:]

_mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
_mid_x_by_frame = dict(zip(_mid["frames"].astype(int).tolist(), _mid["mid_x_um"].astype(float).tolist()))

_apical = np.load(f"{RESULT_DIR}/groups_by_midline/apical_point.npz")
_apical_x_by_frame = dict(zip(_apical["frames"].astype(int).tolist(), _apical["apical_x_um"].astype(float).tolist()))

_dilated_mask_cache = {}


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
    Returns three arrays of node indices: neuromast_left, neuromast_right,
    reference. A node missing from all three is either outside the embryo
    or sitting in the gap band around the midline.
    """
    mask = foreground_mask(t, expanded)
    col = (xy[:, 0] / SPACING_XY).round().astype(int).clip(0, IMG_W - 1)
    row = (xy[:, 1] / SPACING_XY).round().astype(int).clip(0, IMG_H - 1)
    inside = mask[row, col]

    node_id = np.where(inside)[0]
    pos = xy[inside]

    mid_x = _mid_x_by_frame[t]
    apical_x = _apical_x_by_frame[t]

    is_reference = pos[:, 0] >= mid_x + GAP_UM
    is_neuromast = pos[:, 0] < mid_x
    half_width = mid_x - apical_x
    is_right = is_neuromast & (pos[:, 0] >= apical_x)
    is_left = is_neuromast & (pos[:, 0] < apical_x) & (pos[:, 0] >= apical_x - half_width)

    return node_id[is_left], node_id[is_right], node_id[is_reference]
