"""
Step 2 of 6: settle down the frame-by-frame masks from step 1 through time.
Segmentation on its own flickers a pixel or ten at the edges frame to frame;
this cleans that up without erasing genuine shape changes (see _smoothing.py
for how).

Run from the repo root:
    python setup/02_smooth_boundary.py

Needs raw_masks.npy (from step 1). Writes smoothed_boundary.npz to the repo
root - this is one of the files the main pipeline (groups.py) reads.
"""
import numpy as np
from _smoothing import smooth_masks_through_time

if __name__ == "__main__":
    raw_masks = np.load("raw_masks.npy")
    smoothed_masks, cy_s, cx_s = smooth_masks_through_time(raw_masks)

    print("frames with a raw detection:", sum(m.sum() > 0 for m in raw_masks), "/", len(raw_masks))
    print("frames with a smoothed mask:", sum(m.sum() > 0 for m in smoothed_masks), "/", len(smoothed_masks))

    np.savez("smoothed_boundary.npz", smoothed_masks=smoothed_masks, cy=cy_s, cx=cx_s)
    print("saved smoothed_boundary.npz - run setup/03_track_landmarks.py next")
