"""
Same idea as make_overlay_video.py, but this one shows the tissue displaced
by the "growth" field instead of "solution" - i.e. what you'd see if you
only ran ParaView's first WarpByVector stage (growth) and skipped the
second one (solution). See Simulation_membrane_overlay.pvsm for the actual
two-stage pipeline this is standing in for.

Also simpler than the main overlay: just a plain neuromast/reference split
down the midline, no apical-point sub-split and no gap. That's how this
one's always been done, so left it as-is rather than bolting the newer
3-way split onto a field it was never meant for.
"""
import os
import numpy as np
import pyvista as pv
import tifffile
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, disk

from config import MESH_DIR, TIF_DIR, RESULT_DIR, T_START, SEED_FRAME

FRAME_DIR = f"{RESULT_DIR}/overlay_groups_by_midline_growth"
VIDEO_OUT = f"{RESULT_DIR}/overlay_groups_by_midline_growth.mp4"

SPACING_XY = 0.325
MARGIN_UM = 5.0
MARGIN_PX = int(round(MARGIN_UM / SPACING_XY))
CONTROL_Y_LOW = 40.0
CONTROL_Y_HIGH = 90.0

os.makedirs(FRAME_DIR, exist_ok=True)

ref_mesh = pv.read(f"{MESH_DIR}/sim_1.vtu")
ref_xy = ref_mesh.points[:, :2]
control_ids = np.where((ref_xy[:, 1] < CONTROL_Y_LOW) | (ref_xy[:, 1] > CONTROL_Y_HIGH))[0]

masks = np.load(f"{RESULT_DIR}/smoothed_boundary.npz")["smoothed_masks"]
img_h, img_w = masks.shape[1:]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
_all_frames = mid["frames"].astype(int)
mid_x_of = dict(zip(_all_frames.tolist(), mid["mid_x_um"].astype(float).tolist()))
_extrap = mid["is_extrapolated"] if "is_extrapolated" in mid.files else np.zeros(len(_all_frames), dtype=bool)
mid_extrap_of = dict(zip(_all_frames.tolist(), _extrap.tolist()))
frames = _all_frames[(_all_frames >= T_START) & (_all_frames <= SEED_FRAME)]

writer = imageio.get_writer(VIDEO_OUT, fps=12)

for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t}.vtu")
    growth = np.asarray(mesh.point_data["growth"])[:, :2]
    cur_xy = ref_xy + growth

    col = (cur_xy[:, 0] / SPACING_XY).round().astype(int).clip(0, img_w - 1)
    row = (cur_xy[:, 1] / SPACING_XY).round().astype(int).clip(0, img_h - 1)
    mask = binary_dilation(masks[t - 1], disk(MARGIN_PX))
    inside = np.where(mask[row, col])[0]

    mid_x = mid_x_of[t]
    left = inside[cur_xy[inside, 0] < mid_x]
    right = inside[cur_xy[inside, 0] >= mid_x]
    control_xy = cur_xy[control_ids]

    stack = tifffile.imread(f"{TIF_DIR}/T_{t}.tif")
    mip = stack.max(axis=0)
    h, w = mip.shape
    extent = [0, w * SPACING_XY, 0, h * SPACING_XY]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.imshow(mip, cmap="gray", origin="lower", extent=extent)
    ax.scatter(control_xy[:, 0], control_xy[:, 1], s=1.0, c="yellow", alpha=0.25, linewidths=0)
    ax.scatter(cur_xy[left, 0], cur_xy[left, 1], s=1.5, c="dodgerblue", alpha=0.8, linewidths=0)
    ax.scatter(cur_xy[right, 0], cur_xy[right, 1], s=1.5, c="orangered", alpha=0.8, linewidths=0)
    ax.axvline(mid_x, color=("orange" if mid_extrap_of[t] else "white"), linewidth=1.2, linestyle="--")
    ax.axhline(CONTROL_Y_LOW, color="lime", linewidth=0.6, linestyle=":")
    ax.axhline(CONTROL_Y_HIGH, color="lime", linewidth=0.6, linestyle=":")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (micron)")
    ax.set_ylabel("y (micron)")
    ax.set_title(
        f"[growth] t={t}  |  midline at {mid_x:.1f}um  |  "
        f"neuromast={len(left)}  reference={len(right)}  control={len(control_ids)}"
    )
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=130)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: neuromast={len(left)} reference={len(right)}")

writer.close()
print(f"wrote {VIDEO_OUT}")
