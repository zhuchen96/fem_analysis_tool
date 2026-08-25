"""
The actual method behind 02_smooth_boundary.py. Split out into its own file
just because it's a chunk of numeric code that's easier to read on its own,
away from the "load this, save that" bits.

Why this exists: frame-by-frame segmentation (01_segment_masks.py) is noisy
at the edges, and a simple "average the outline radius per angle" smoother
can't represent a shape that splits into two lobes (which this structure
does partway through most sequences) - it just cuts across the notch or
bulges past the real edge.

Instead: turn each frame's mask into a signed distance field (positive
inside, negative outside), re-center every frame on a fixed point first so
the structure's own bulk translation doesn't get smeared into the temporal
average, smooth the aligned SDF stack through time, then shift the smoothed
result back to each frame's own position and re-threshold at zero. Working
with the full 2D field (not a 1-radius-per-angle summary) is what lets it
keep concave shapes and forks.
"""
import numpy as np
from scipy import ndimage

SIGMA_T = 5  # frames, the temporal smoothing width
# minimum fraction of the temporal smoothing window that needs to be backed
# by an actual detection - below this the "smoothed" mask is mostly just
# extrapolation, not real signal (mainly matters for the very first few
# frames, before the structure has properly formed)
CONFIDENCE_FLOOR = 0.35


def _nan_gaussian_smooth_1d(arr, sigma):
    valid = ~np.isnan(arr)
    if valid.sum() == 0:
        return arr
    filled = np.where(valid, arr, 0.0)
    num = ndimage.gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    den = ndimage.gaussian_filter1d(valid.astype(float), sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out


def _raw_centroids(raw_masks):
    n = len(raw_masks)
    cy = np.full(n, np.nan)
    cx = np.full(n, np.nan)
    for i, m in enumerate(raw_masks):
        if m.sum() == 0:
            continue
        ys, xs = np.nonzero(m)
        cy[i] = ys.mean()
        cx[i] = xs.mean()
    return cy, cx


def _mask_to_sdf(mask):
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return (inside - outside).astype(np.float32)


def smooth_masks_through_time(raw_masks, sigma_t=SIGMA_T):
    n, h, w = raw_masks.shape
    cy, cx = _raw_centroids(raw_masks)
    cy_s = _nan_gaussian_smooth_1d(cy, sigma_t)
    cx_s = _nan_gaussian_smooth_1d(cx, sigma_t)

    cy0, cx0 = h / 2.0, w / 2.0  # everything aligns to this fixed canonical center

    aligned = np.full((n, h, w), np.nan, dtype=np.float32)
    for t in range(n):
        if np.isnan(cy[t]):
            continue
        sdf = _mask_to_sdf(raw_masks[t])
        dy, dx = cy0 - cy[t], cx0 - cx[t]
        aligned[t] = ndimage.shift(sdf, shift=(dy, dx), order=1, mode="nearest")

    valid = ~np.isnan(aligned)
    filled = np.where(valid, aligned, 0.0)
    num = ndimage.gaussian_filter1d(filled, sigma=sigma_t, axis=0, mode="nearest")
    den = ndimage.gaussian_filter1d(valid.astype(np.float32), sigma=sigma_t, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed_aligned = num / den
    smoothed_aligned[den < 1e-6] = -1.0  # nothing nearby in time -> treat as outside

    den_1d = den[:, 0, 0]  # this validity fraction only varies in t, uniform across the frame

    smoothed_masks = np.zeros((n, h, w), dtype=bool)
    for t in range(n):
        if np.isnan(cy_s[t]) or den_1d[t] < CONFIDENCE_FLOOR:
            continue
        dy, dx = cy_s[t] - cy0, cx_s[t] - cx0
        back = ndimage.shift(smoothed_aligned[t], shift=(dy, dx), order=1, mode="nearest")
        smoothed_masks[t] = back > 0

    return smoothed_masks, cy_s, cx_s
