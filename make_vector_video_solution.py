"""
A coarser, easier-to-read alternative to the rose histogram: for each
region, just average the "solution" displacement over every point in it,
then draw the result as three little 1D arrows (x, y, z each get their own
number line and their own row... er, column). One arrow per region per
axis, nine panels total per frame.

Axis limits are fixed for the whole video (computed from every frame up
front) but x, y and z each get their own scale - z tends to be an order of
magnitude bigger than x or y here, so sharing one scale would flatten x/y
to nothing.

Plain segmentation mask - see the _expanded_mask twin for the padded one.
"""
import os
import numpy as np
import pyvista as pv
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from groups import get_regions
from config import MESH_DIR, RESULT_DIR

FRAME_DIR = f"{RESULT_DIR}/vector_arrows_solution"
VIDEO_OUT = f"{RESULT_DIR}/vector_arrows_solution.mp4"
FIELD = "solution"
EXPANDED = False

REGIONS = ["neuromast_left", "neuromast_right", "reference"]
REGION_LABEL = {
    "neuromast_left": "neuromast, front half",
    "neuromast_right": "neuromast, back half",
    "reference": "reference tissue",
}
AXES = ["x", "y", "z"]

os.makedirs(FRAME_DIR, exist_ok=True)

ref_mesh = pv.read(f"{MESH_DIR}/sim_1.vtu")
ref_xy = ref_mesh.points[:, :2]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
frames = mid["frames"].astype(int)

# first pass - walk every frame once and just record the mean vector per region
mean_vec = {r: {a: [] for a in AXES} for r in REGIONS}
for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t}.vtu")
    cur_xy = ref_xy + np.asarray(mesh.point_data["solution"])[:, :2]
    field = np.asarray(mesh.point_data[FIELD])

    left, right, ref = get_regions(t, cur_xy, expanded=EXPANDED)
    ids_of = {"neuromast_left": left, "neuromast_right": right, "reference": ref}

    for r in REGIONS:
        ids = ids_of[r]
        avg = field[ids].mean(axis=0) if len(ids) else (np.nan, np.nan, np.nan)
        for i, a in enumerate(AXES):
            mean_vec[r][a].append(avg[i])

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: n = " + " ".join(f"{r}={len(ids_of[r])}" for r in REGIONS))

# one shared limit per axis, pooled across all three regions
axis_limit = {}
for a in AXES:
    vals = np.concatenate([np.array(mean_vec[r][a]) for r in REGIONS])
    vals = vals[np.isfinite(vals)]
    axis_limit[a] = np.nanmax(np.abs(vals)) * 1.15 if len(vals) else 1.0

# second pass - drawing only, no need to touch the vtu files again
writer = imageio.get_writer(VIDEO_OUT, fps=12)
for i, t in enumerate(frames):
    t = int(t)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for row, r in enumerate(REGIONS):
        for col, a in enumerate(AXES):
            ax = axes[row, col]
            v = mean_vec[r][a][i]
            limit = axis_limit[a]
            ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
            if np.isfinite(v) and v != 0:
                # cap the head length at the shaft itself - otherwise a
                # short vector ends up all arrowhead and no shaft
                head_len = min(limit * 0.08, abs(v) * 0.9)
                ax.arrow(0, 0, v, 0, length_includes_head=True, head_width=0.12,
                          head_length=head_len, width=0.02, color="darkorange", linewidth=0)
                ax.set_title(f"{REGION_LABEL[r]}, {a}  (v={v:.3g})", fontsize=8.5)
            elif v == 0:
                ax.plot(0, 0, "o", color="darkorange", markersize=4)
                ax.set_title(f"{REGION_LABEL[r]}, {a}  (v=0)", fontsize=8.5)
            else:
                ax.set_title(f"{REGION_LABEL[r]}, {a}  (no points)", fontsize=8.5)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-0.3, 0.3)
            ax.set_yticks([])
            ax.set_xlabel(a)
            ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"solution field, t={t} - average displacement per region, one axis at a time")
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=110)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

writer.close()
print(f"wrote {VIDEO_OUT}")
