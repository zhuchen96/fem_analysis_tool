"""
Step 5 of 6: pull out the apical constriction point trajectory - the
neuromast-side landmark (point 1) on its own, not the midpoint of both
(that's the midline, from step 4). This is what further splits the
neuromast into a front half and a back half in groups.py.

Same approach as step 4: use the tracked x-position where we have it, then
extrapolate forward past SEED_FRAME with a straight-line fit to the last
FIT_WINDOW tracked frames.

Run from the repo root:
    python setup/05_build_apical_point.py

Needs tracked_points_backward_v2.npz (step 3). Writes
groups_by_midline/apical_point.npz.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import SPACING_XY, T_START, FIT_WINDOW, RESULT_DIR, TIF_START, VTU_START
from _dataset import vtu_frame_count, tif_frame_count

OUT_DIR = f"{RESULT_DIR}/groups_by_midline"
os.makedirs(OUT_DIR, exist_ok=True)

LAST_FRAME = min(tif_frame_count(), vtu_frame_count() + TIF_START - VTU_START)

d = np.load(f"{RESULT_DIR}/tracked_points_backward_v2.npz")
traj_xy = d["traj_xy"]
seed_frame = int(d["seed_frame"])

apical_px_x = traj_xy[1, :, 0]  # point 1 = the neuromast-side landmark, see config.py
valid = ~np.isnan(apical_px_x)
tracked_frames = [t for t in range(T_START, seed_frame + 1) if valid[t - 1]]
tracked_x_um = np.array([apical_px_x[t - 1] * SPACING_XY for t in tracked_frames])
print(f"apical point tracked {tracked_frames[0]}..{tracked_frames[-1]} ({len(tracked_frames)} frames)")

fit_frames = [t for t in tracked_frames if t >= (seed_frame - FIT_WINDOW + 1)]
fit_x = np.array([apical_px_x[t - 1] * SPACING_XY for t in fit_frames])
slope, intercept = np.polyfit(np.array(fit_frames, dtype=float), fit_x, 1)
print(f"extrapolating past frame {seed_frame} at {slope:.4f} um/frame "
      f"(fit over frames {fit_frames[0]}..{seed_frame})")

extrap_frames = list(range(seed_frame + 1, LAST_FRAME + 1))
extrap_x_um = np.array([slope * t + intercept for t in extrap_frames])

all_frames = np.array(tracked_frames + extrap_frames)
all_x_um = np.concatenate([tracked_x_um, extrap_x_um])
is_extrapolated = np.concatenate([
    np.zeros(len(tracked_frames), dtype=bool),
    np.ones(len(extrap_frames), dtype=bool),
])

np.savez(f"{OUT_DIR}/apical_point.npz",
         frames=all_frames, apical_x_um=all_x_um, is_extrapolated=is_extrapolated,
         fit_slope=slope, fit_intercept=intercept, fit_seed_frame=seed_frame)
print(f"saved {OUT_DIR}/apical_point.npz, covering {all_frames.min()}..{all_frames.max()} "
      f"({(~is_extrapolated).sum()} tracked, {is_extrapolated.sum()} extrapolated)")
print("run setup/06_migration_direction.py next")
