"""
Step 1 of 6: get a rough per-frame mask of the structure out of the raw
microscopy stack. This is deliberately quick and dirty, frame by frame -
02_smooth_boundary.py does the actual temporal cleanup afterward, so small
mistakes here (a flickering pixel or two at the edge) don't matter much.

Method: gaussian-smooth the max-projection, Otsu-threshold it, keep the
single blob that looks most like real structure (solid, not too small, not
a thin fiber streak), then grow it out a bit with a looser hysteresis
threshold to catch the dimmer-but-real halo around it.

Run from the repo root:
    python setup/01_segment_masks.py

Writes raw_masks.npy to the repo root.
"""
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, closing, disk
from skimage.measure import label, regionprops

from _dataset import tif_frame_count, tif_image_shape

N_FRAMES = tif_frame_count()
IMG_H, IMG_W = tif_image_shape()

# below this Otsu threshold, a frame just doesn't have real structure in it
# yet (early frames especially) - Otsu on pure background noise still picks
# *some* split point, so we need this floor to catch that and skip it
OTSU_FLOOR = 50

# the hysteresis low threshold, as a fraction of the frame's own Otsu (high)
# threshold - expands the bright seed blob out to the dimmer-but-real
# halo/neck around it, while the connectivity requirement in hysteresis
# thresholding keeps it from also picking up unrelated background specks
# sitting at a similar brightness
LOW_FRAC = 0.55


def segment_frame(mip):
    smooth = gaussian(mip.astype(float), sigma=3, preserve_range=True)
    thresh = threshold_otsu(smooth)
    if thresh < OTSU_FLOOR:
        return np.zeros(mip.shape, dtype=bool)

    mask = smooth > thresh
    mask = closing(mask, disk(5))
    mask = remove_small_objects(mask, min_size=300)
    mask = ndimage.binary_fill_holes(mask)

    lbl = label(mask)
    if lbl.max() == 0:
        return np.zeros_like(mask)

    # of everything that survived, pick the one blob that actually looks
    # like the real structure - compact (not a thin streak) and reasonably
    # sized, then break ties by picking the brightest
    best = None
    best_score = -1
    for r in regionprops(lbl, intensity_image=smooth):
        if r.solidity < 0.5 or r.area < 300:
            continue
        score = r.area * r.intensity_mean
        if score > best_score:
            best_score = score
            best = r.label
    if best is None:
        return np.zeros_like(mask)
    seed = lbl == best

    # now grow that seed out to the dimmer halo around it, but only the
    # part of the halo actually touching the seed
    hyst = apply_hysteresis_threshold(smooth, thresh * LOW_FRAC, thresh)
    hyst = ndimage.binary_fill_holes(hyst)
    hyst_lbl = label(hyst)
    touching = set(np.unique(hyst_lbl[seed])) - {0}
    if not touching:
        return seed
    return np.isin(hyst_lbl, list(touching))


if __name__ == "__main__":
    print(f"segmenting {N_FRAMES} frames ({IMG_H}x{IMG_W} px each)...")
    raw_masks = np.zeros((N_FRAMES, IMG_H, IMG_W), dtype=bool)
    for i in range(1, N_FRAMES + 1):
        stack = tifffile.imread(f"Time series_flipped/T_{i}.tif")
        mip = stack.max(axis=0)
        raw_masks[i - 1] = segment_frame(mip)
        if i % 20 == 0 or i == 1:
            print(f"  frame {i}: {raw_masks[i - 1].sum()} px detected")

    np.save("raw_masks.npy", raw_masks)
    print("saved raw_masks.npy - run setup/02_smooth_boundary.py next")
