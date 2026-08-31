"""
Step 6 of 6 (last one): work out a single fixed "which way is the tissue
migrating" direction in the xy plane, used as the reference axis for the
rose histograms in the main pipeline instead of the raw lab-frame x-axis
(which has no reason to line up with the actual migration).

Method: track the combined neuromast+reference centroid's (x, y) position
across every frame and fit a straight line to it over time - the slope of
that fit is the migration direction. We use the centroid *position* trend
rather than averaging the mesh's displacement field directly, because that
field is a noisy, high-frequency elastic-relaxation signal frame to frame
and gave inconsistent directions when tried; the centroid position itself
is clean and close to monotonic, which matches a real net migration.

Run from the repo root, after steps 1, 2, 4 and 5 are done (this needs
smoothed_boundary.npz, midline.npz and apical_point.npz already in place,
since it reuses groups.py to work out which mesh points count as tissue):

    python setup/06_migration_direction.py

Writes migration_direction.npz to the repo root.
"""
import os
import sys
import numpy as np
import pyvista as pv

# groups.py lives one folder up, at the repo root, not inside setup/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from groups import get_regions
from config import MESH_DIR, RESULT_DIR, TIF_START, VTU_START  # root config.py

ref_mesh = pv.read(f"{MESH_DIR}/sim_{VTU_START}.vtu")
ref_xy = ref_mesh.points[:, :2]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
frames = mid["frames"].astype(int)

cx, cy = [], []
for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t - TIF_START + VTU_START}.vtu")
    cur_xy = ref_xy + np.asarray(mesh.point_data["growth"])[:, :2] + np.asarray(mesh.point_data["solution"])[:, :2]

    left, right, ref = get_regions(t, cur_xy, expanded=False)
    tissue = np.concatenate([left, right, ref])
    cx.append(cur_xy[tissue, 0].mean())
    cy.append(cur_xy[tissue, 1].mean())

cx = np.array(cx)
cy = np.array(cy)
t = frames.astype(float)

A = np.vstack([t, np.ones_like(t)]).T
sx, _ = np.linalg.lstsq(A, cx, rcond=None)[0]
sy, _ = np.linalg.lstsq(A, cy, rcond=None)[0]

vec = np.array([sx, sy])
unit = vec / np.linalg.norm(vec)
angle = np.degrees(np.arctan2(vec[1], vec[0]))
print(f"migration direction: {angle:.1f} deg, unit vector {unit}")

# quick sanity check - plain start-to-end displacement should point roughly
# the same way as the fitted trend
vec2 = np.array([cx[-1] - cx[0], cy[-1] - cy[0]])
angle2 = np.degrees(np.arctan2(vec2[1], vec2[0]))
print(f"endpoint-to-endpoint angle for comparison: {angle2:.1f} deg")

np.savez(f"{RESULT_DIR}/migration_direction.npz", angle_deg=angle, unit=unit, method="centroid_position_linregress")
print(f"saved {RESULT_DIR}/migration_direction.npz - setup is done, the make_*_video.py scripts in the repo root are ready to run")
