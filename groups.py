"""
This is the one place that decides "which region is this mesh point in".
Everything downstream (the overlay video, the rose histograms, the vector
arrows) imports get_regions() from here instead of copy-pasting the split
logic five different times - if the split rule ever changes again we only
have to fix it in one spot.

The three regions, in terms of a point's current (deformed) x position:

    neuromast_left   - in front of the tracked apical constriction point
    neuromast_right  - between the apical point and the midline
    reference        - behind the midline

We leave a small dead zone straddling the midline (about 6.5 micron, i.e.
+/- 20 pixels) so points don't hop back and forth between neuromast and
reference from one frame to the next just from mesh jitter. There's no
equivalent gap at the apical point - that split is a plain cutoff.

A point only counts as "in" one of these regions at all if it also falls
inside the segmented embryo outline. Two flavors of that test are used
around the project: the plain segmented mask, or the same mask grown out
by 5 micron first (catches points that sit just outside the outline, which
happens more than you'd think near the tail). Pass expanded=True to
get_regions() for the second one.
"""
import numpy as np
from skimage.morphology import binary_dilation, disk

SPACING_XY = 0.325  # micron per pixel - the value the whole pipeline is built around
GAP_UM = 20 * SPACING_XY  # ~6.5um, the midline dead zone (20px, chosen early on and kept ever since)
MARGIN_UM = 5.0
MARGIN_PX = int(round(MARGIN_UM / SPACING_XY))

_masks = np.load("smoothed_boundary.npz")["smoothed_masks"]
IMG_H, IMG_W = _masks.shape[1:]

_mid = np.load("groups_by_midline/midline.npz")
_mid_x_by_frame = dict(zip(_mid["frames"].astype(int).tolist(), _mid["mid_x_um"].astype(float).tolist()))

_apical = np.load("groups_by_midline/apical_point.npz")
_apical_x_by_frame = dict(zip(_apical["frames"].astype(int).tolist(), _apical["apical_x_um"].astype(float).tolist()))

_dilated_mask_cache = {}


def foreground_mask(t, expanded):
    mask = _masks[t - 1]
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
    is_neuromast = pos[:, 0] < mid_x - GAP_UM
    is_left = is_neuromast & (pos[:, 0] < apical_x)
    is_right = is_neuromast & (pos[:, 0] >= apical_x)

    return node_id[is_left], node_id[is_right], node_id[is_reference]
