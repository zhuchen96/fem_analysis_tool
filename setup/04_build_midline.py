"""
Step 4 of 6: build the midline - the moving vertical line that separates
"neuromast" tissue from "reference" tissue - out of the two tracked
landmarks from step 3.

Two things happen here:
  1. Wherever both landmarks are still tracked, the midline is just their
     midpoint in x.
  2. The tracker only reaches back from SEED_FRAME (it tracks backward,
     never forward), so there's nothing tracked after that. Past
     SEED_FRAME the midline gets extrapolated forward in a straight line,
     fit to the landmarks' average speed over the last FIT_WINDOW tracked
     frames - the tissue moves smoothly enough on that timescale for a
     linear extrapolation to hold up over the remaining frames.

Run from the repo root:
    python setup/04_build_midline.py

Needs tracked_points_backward_v2.npz (step 3). Writes
groups_by_midline/midline.npz, covering config.T_START through the last
frame that has mesh data.
"""
import os
import numpy as np

from config import SPACING_XY, T_START, FIT_WINDOW
from _dataset import vtu_frame_count

OUT_DIR = "groups_by_midline"
os.makedirs(OUT_DIR, exist_ok=True)

LAST_FRAME = vtu_frame_count()

d = np.load("tracked_points_backward_v2.npz")
traj_xy = d["traj_xy"]  # (2, seed_frame, 2) px, indexed [t-1]
seed_frame = int(d["seed_frame"])

# point 0 should be the reference-side (larger x) landmark and point 1 the
# neuromast-side one - sanity check that, since 05_build_apical_point.py
# assumes it too and a swapped config.py would quietly wreck the split
both_tracked = ~np.isnan(traj_xy[0, :, 0]) & ~np.isnan(traj_xy[1, :, 0])
if both_tracked.any() and not (traj_xy[0, both_tracked, 0] > traj_xy[1, both_tracked, 0]).all():
    print("WARNING: landmark 0 isn't consistently to the right of landmark 1 - "
          "double-check the order of SEED_POINTS_XY in config.py")

valid = ~np.isnan(traj_xy[:, :, 0]).any(axis=0)  # frame valid if BOTH points have a position
tracked_frames = [t for t in range(T_START, seed_frame + 1) if valid[t - 1]]
print(f"tracked midline covers {tracked_frames[0]}..{tracked_frames[-1]} "
      f"({len(tracked_frames)} frames; skipped "
      f"{seed_frame - T_START + 1 - len(tracked_frames)} where a landmark had already dropped out)")

tracked_mid_x = np.array([
    (traj_xy[0, t - 1, 0] + traj_xy[1, t - 1, 0]) / 2.0 * SPACING_XY
    for t in tracked_frames
])

fit_frames = [t for t in tracked_frames if t >= (seed_frame - FIT_WINDOW + 1)]
fit_x = np.array([
    (traj_xy[0, t - 1, 0] + traj_xy[1, t - 1, 0]) / 2.0 * SPACING_XY
    for t in fit_frames
])
slope, intercept = np.polyfit(np.array(fit_frames, dtype=float), fit_x, 1)
print(f"extrapolating past frame {seed_frame} at {slope:.4f} um/frame "
      f"(fit over frames {fit_frames[0]}..{seed_frame})")

extrap_frames = list(range(seed_frame + 1, LAST_FRAME + 1))
extrap_mid_x = np.array([slope * t + intercept for t in extrap_frames])

all_frames = np.array(tracked_frames + extrap_frames)
all_mid_x = np.concatenate([tracked_mid_x, extrap_mid_x])
is_extrapolated = np.concatenate([
    np.zeros(len(tracked_frames), dtype=bool),
    np.ones(len(extrap_frames), dtype=bool),
])

np.savez(f"{OUT_DIR}/midline.npz",
         frames=all_frames, mid_x_um=all_mid_x, is_extrapolated=is_extrapolated,
         fit_slope=slope, fit_intercept=intercept, fit_seed_frame=seed_frame)
print(f"saved {OUT_DIR}/midline.npz, covering {all_frames.min()}..{all_frames.max()} "
      f"({(~is_extrapolated).sum()} tracked, {is_extrapolated.sum()} extrapolated)")
print("run setup/05_build_apical_point.py next")
