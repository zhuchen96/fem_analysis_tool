"""
Overlay video: plots the neuromast/reference split on top of the actual
microscopy images (max-projected), frame by frame, then stitches everything
into an mp4. Colors are blue = front half of the neuromast, orange = back
half, red = reference tissue, yellow = the fixed top/bottom strip we keep
around as a stationary control region.

This is the plain-mask version - a point only counts as "inside the embryo"
if it lands within the segmented outline exactly as drawn. See
make_overlay_video_expanded_mask.py for the version that pads the outline
out by 5um first before testing.

Needs: sim_*.vtu (deformation field) + T_*.tif (raw images) + the tracked
midline/apical-point/segmentation files that groups.py loads.
"""
import os
import numpy as np
import pyvista as pv
import tifffile
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from groups import get_regions, GAP_UM, SPACING_XY
from config import MESH_DIR, TIF_DIR, RESULT_DIR, T_START, SEED_FRAME

FRAME_DIR = f"{RESULT_DIR}/overlay_groups_by_midline"
VIDEO_OUT = f"{RESULT_DIR}/overlay_groups_by_midline.mp4"
EXPANDED = False

# the fixed control strip (top/bottom edges of the tissue) - same node set
# for every frame, only its position moves as the mesh deforms
CONTROL_Y_LOW = 40.0
CONTROL_Y_HIGH = 90.0

os.makedirs(FRAME_DIR, exist_ok=True)

ref_mesh = pv.read(f"{MESH_DIR}/sim_1.vtu")
ref_xy = ref_mesh.points[:, :2]
control_ids = np.where((ref_xy[:, 1] < CONTROL_Y_LOW) | (ref_xy[:, 1] > CONTROL_Y_HIGH))[0]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
_all_frames = mid["frames"].astype(int)
mid_x_of = dict(zip(_all_frames.tolist(), mid["mid_x_um"].astype(float).tolist()))
_extrap = mid["is_extrapolated"] if "is_extrapolated" in mid.files else np.zeros(len(_all_frames), dtype=bool)
mid_extrap_of = dict(zip(_all_frames.tolist(), _extrap.tolist()))
frames = _all_frames[(_all_frames >= T_START) & (_all_frames <= SEED_FRAME)]

apical = np.load(f"{RESULT_DIR}/groups_by_midline/apical_point.npz")
apical_x_of = dict(zip(apical["frames"].astype(int).tolist(), apical["apical_x_um"].astype(float).tolist()))
apical_extrap_of = dict(zip(apical["frames"].astype(int).tolist(), apical["is_extrapolated"].tolist()))

writer = imageio.get_writer(VIDEO_OUT, fps=12)

for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t}.vtu")
    cur_xy = ref_xy + np.asarray(mesh.point_data["growth"])[:, :2] + np.asarray(mesh.point_data["solution"])[:, :2]

    left, right, ref = get_regions(t, cur_xy, expanded=EXPANDED)
    control_xy = cur_xy[control_ids]

    stack = tifffile.imread(f"{TIF_DIR}/T_{t}.tif")
    mip = stack.max(axis=0)
    h, w = mip.shape
    extent = [0, w * SPACING_XY, 0, h * SPACING_XY]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.imshow(mip, cmap="gray", origin="lower", extent=extent)
    ax.scatter(control_xy[:, 0], control_xy[:, 1], s=1.0, c="yellow", alpha=0.25, linewidths=0)
    ax.scatter(cur_xy[left, 0], cur_xy[left, 1], s=1.5, c="dodgerblue", alpha=0.8, linewidths=0)
    ax.scatter(cur_xy[right, 0], cur_xy[right, 1], s=1.5, c="darkorange", alpha=0.8, linewidths=0)
    ax.scatter(cur_xy[ref, 0], cur_xy[ref, 1], s=1.5, c="crimson", alpha=0.8, linewidths=0)

    mid_x = mid_x_of[t]
    ax.axvspan(mid_x - GAP_UM, mid_x + GAP_UM, color="magenta", alpha=0.2, zorder=0.5)
    ax.axvline(mid_x - GAP_UM, color="magenta", linewidth=0.9, linestyle=":")
    ax.axvline(mid_x + GAP_UM, color="magenta", linewidth=0.9, linestyle=":")
    ax.axvline(mid_x, color=("magenta" if mid_extrap_of[t] else "white"), linewidth=1.2, linestyle="--")

    apical_x = apical_x_of[t]
    ax.axvline(apical_x, color=("yellow" if apical_extrap_of[t] else "cyan"), linewidth=1.2, linestyle="--")

    ax.axhline(CONTROL_Y_LOW, color="lime", linewidth=0.6, linestyle=":")
    ax.axhline(CONTROL_Y_HIGH, color="lime", linewidth=0.6, linestyle=":")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (micron)")
    ax.set_ylabel("y (micron)")
    ax.set_title(
        f"t={t}  |  apical point at {apical_x:.1f}um (cyan)  |  midline at {mid_x:.1f} +/- {GAP_UM:.1f}um (white)\n"
        f"neuromast_left={len(left)}  neuromast_right={len(right)}  reference={len(ref)}  control={len(control_ids)}",
        fontsize=9,
    )
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=130)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: neuromast_left={len(left)} neuromast_right={len(right)} reference={len(ref)}")

writer.close()
print(f"wrote {VIDEO_OUT}")
