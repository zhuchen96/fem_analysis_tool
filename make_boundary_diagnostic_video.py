"""
Diagnostic video showing the front/middle division line frame by frame.

Two modes:
  GREEN line  — mask mode: mask_tip_x ≤ mesh_tip_x;
                boundary = mask_tip_x − REF_FRONT_UM (20 µm).
  RED line    — extrapolation mode: mask tip has moved beyond the FEM domain;
                boundary is extrapolated at the speed measured over all mask-mode frames.

Additional vertical lines:
  cyan dotted  — mesh tip (rightmost unfiltered FEM node in the reference x-range)
  yellow dotted— mask tip (rightmost foreground pixel)
  white dashed — midline
  magenta dotted— midline + GAP_UM (dead-zone edge)

Dots on the image:
  crimson      — front region mesh nodes
  mediumpurple — middle region mesh nodes

Run from the repo root:
    python make_boundary_diagnostic_video.py
Output: result_sampleN/boundary_diagnostic.mp4
"""
import os
import numpy as np
import pyvista as pv
import tifffile
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from groups import (get_regions, foreground_mask,
                    GAP_UM, REF_FRONT_UM, SPACING_XY, IMG_W, IMG_H,
                    _mask_tip_x, _valid_boundaries, _mid_x_by_frame, EXTRAP_SPEED_SCALE)
from config import MESH_DIR, TIF_DIR, RESULT_DIR, T_START, SEED_FRAME, TIF_START, VTU_START

FRAME_DIR = f"{RESULT_DIR}/boundary_diagnostic"
VIDEO_OUT = f"{RESULT_DIR}/boundary_diagnostic.mp4"

os.makedirs(FRAME_DIR, exist_ok=True)

ref_mesh = pv.read(f"{MESH_DIR}/sim_{VTU_START}.vtu")
ref_xy = ref_mesh.points[:, :2]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
_all_frames = mid["frames"].astype(int)
frames = _all_frames[(_all_frames >= T_START) & (_all_frames <= SEED_FRAME)]


def _fmt(v, d=1):
    return f"{v:.{d}f}" if v is not None else "N/A"


print("=== first pass: computing boundaries ===")
frame_data = {}

for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t - TIF_START + VTU_START}.vtu")
    cur_xy = (ref_xy
              + np.asarray(mesh.point_data["growth"])[:, :2]
              + np.asarray(mesh.point_data["solution"])[:, :2])

    mid_x = _mid_x_by_frame[t]

    # Mesh tip from unfiltered nodes (same as groups.py step)
    is_ref_all = cur_xy[:, 0] >= mid_x + GAP_UM
    mesh_tip_x = float(cur_xy[is_ref_all, 0].max()) if is_ref_all.any() else None

    mask_tip_x_val = _mask_tip_x.get(t)

    # Call get_regions — this populates _valid_boundaries as a side effect.
    left, right, front, middle = get_regions(t, cur_xy, expanded=False)

    in_mask_mode = t in _valid_boundaries
    bx = _valid_boundaries.get(t)

    # For extrapolation frames, recompute the extrapolated value.
    if not in_mask_mode and len(_valid_boundaries) >= 2:
        ts_arr = np.array(sorted(_valid_boundaries), dtype=float)
        bvals = np.array([_valid_boundaries[t_] for t_ in ts_arr])
        slope_raw, _ = np.polyfit(ts_arr, bvals, 1)
        slope = slope_raw * EXTRAP_SPEED_SCALE
        last_t = float(ts_arr[-1])
        last_b = float(bvals[-1])
        bx = last_b + slope * (t - last_t)

    frame_data[t] = {
        "front_xy":   cur_xy[front],
        "middle_xy":  cur_xy[middle],
        "boundary_x": bx,
        "mesh_tip_x": mesh_tip_x,
        "mask_tip_x": mask_tip_x_val,
        "mode":       "mask" if in_mask_mode else "extrapolation",
        "mid_x":      mid_x,
    }

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        mode_tag = "MASK  " if in_mask_mode else "EXTRAP"
        print(f"  t={t}  {mode_tag}  boundary={_fmt(bx)}  "
              f"mesh_tip={_fmt(mesh_tip_x)}  mask_tip={_fmt(mask_tip_x_val)}")

# Measure speed using the same window as groups.py
speed_slope = None
if len(_valid_boundaries) >= 2:
    ts_f = np.array(sorted(_valid_boundaries), dtype=float)
    bvals_f = np.array([_valid_boundaries[t_] for t_ in ts_f])
    speed_slope, _ = np.polyfit(ts_f, bvals_f, 1)
    speed_slope = speed_slope * EXTRAP_SPEED_SCALE
    n_mask = len(_valid_boundaries)
    n_ext = len(frames) - n_mask
    print(f"\nmeasured boundary speed: {speed_slope:.4f} µm/frame  "
          f"({n_mask} mask-mode frames, {n_ext} extrapolation frames)")
else:
    print("\ntoo few mask-mode frames to measure speed")

print("\n=== second pass: drawing ===")
writer = imageio.get_writer(VIDEO_OUT, fps=12)

for t in frames:
    t = int(t)
    d = frame_data[t]
    bx      = d["boundary_x"]
    mt      = d["mesh_tip_x"]
    msk     = d["mask_tip_x"]
    mode    = d["mode"]
    mid_x   = d["mid_x"]

    stack = tifffile.imread(f"{TIF_DIR}/T_{t}.tif")
    mip = stack.max(axis=0)
    h, w = mip.shape
    extent = [0, w * SPACING_XY, 0, h * SPACING_XY]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.imshow(mip, cmap="gray", origin="lower", extent=extent)

    if len(d["front_xy"]):
        ax.scatter(d["front_xy"][:, 0], d["front_xy"][:, 1],
                   s=1.5, c="crimson", alpha=0.8, linewidths=0, label="front")
    if len(d["middle_xy"]):
        ax.scatter(d["middle_xy"][:, 0], d["middle_xy"][:, 1],
                   s=1.5, c="mediumpurple", alpha=0.8, linewidths=0, label="middle")

    bline_color = "limegreen" if mode == "mask" else "tomato"
    if bx is not None:
        ax.axvline(bx, color=bline_color, linewidth=2.2, linestyle="-",
                   label=f"boundary [{mode}]")

    if mt is not None:
        ax.axvline(mt,  color="cyan",   linewidth=1.0, linestyle=":", label="mesh tip")
    if msk is not None:
        ax.axvline(msk, color="yellow", linewidth=1.0, linestyle=":", label="mask tip")

    ax.axvline(mid_x,          color="white",   linewidth=0.9, linestyle="--")
    ax.axvline(mid_x + GAP_UM, color="magenta", linewidth=0.7, linestyle=":")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    speed_str = ""
    if mode == "extrapolation" and speed_slope is not None:
        speed_str = f"  speed={speed_slope:.3f} µm/fr"
    ax.set_title(
        f"t={t}  │  mode: {mode}{speed_str}\n"
        f"boundary={_fmt(bx)} µm    mesh_tip={_fmt(mt)} µm    mask_tip={_fmt(msk)} µm",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper left", markerscale=3)
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=130)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"  drew t={t}")

writer.close()
print(f"wrote {VIDEO_OUT}")
