"""
For each frame in [T_START, SEED_FRAME], retrieves the x/y/z field values for
every mesh node in each region, then produces:

  Per-field outputs (e.g. "solution"):
    result_sampleN/curves_solution/
        neuromast_left.csv    -- one row per (frame, point): frame,point_id,x,y,z
        neuromast_right.csv
        whole_neuromast.csv
        reference.csv
    result_sampleN/curves_solution.png
        -- 3-panel figure (x / y / z vs time), four curves per panel
           (one per region), each curve is mean ± shaded std

Repeated for both "solution" and "pk1_stess_3".
Uses the plain (non-expanded) segmentation mask.
"""
import os
import csv
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from groups import get_regions
from config import (MESH_DIR, RESULT_DIR, T_START, SEED_FRAME,
                    TIF_START, VTU_START)

FIELDS = ["solution", "pk1_stess_3"]
REGIONS = ["neuromast_left", "neuromast_right", "whole_neuromast", "reference"]
REGION_LABEL = {
    "neuromast_left":  "neuromast, left",
    "neuromast_right": "neuromast, right",
    "whole_neuromast": "whole neuromast",
    "reference":       "reference tissue",
}
REGION_COLOR = {
    "neuromast_left":  "dodgerblue",
    "neuromast_right": "darkorange",
    "whole_neuromast": "green",
    "reference":       "crimson",
}
AXES_NAMES = ["x", "y", "z"]
EXPANDED = False

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
frames = mid["frames"].astype(int)
frames = frames[(frames >= T_START) & (frames <= SEED_FRAME)]

ref_mesh = pv.read(f"{MESH_DIR}/sim_{VTU_START}.vtu")
ref_xy = ref_mesh.points[:, :2]

for field in FIELDS:
    out_dir = f"{RESULT_DIR}/curves_{field}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {field} ===")

    # mean and std accumulators: region -> axis -> list (one entry per frame)
    means = {r: {ax: [] for ax in AXES_NAMES} for r in REGIONS}
    stds  = {r: {ax: [] for ax in AXES_NAMES} for r in REGIONS}
    frame_list = []

    # open per-region CSV files up front
    csv_handles = {}
    writers = {}
    for r in REGIONS:
        fh = open(f"{out_dir}/{r}.csv", "w", newline="")
        csv_handles[r] = fh
        writers[r] = csv.writer(fh)
        writers[r].writerow(["frame", "point_id", "x", "y", "z"])

    for t in frames:
        t = int(t)
        mesh = pv.read(f"{MESH_DIR}/sim_{t - TIF_START + VTU_START}.vtu")
        cur_xy = (ref_xy
                  + np.asarray(mesh.point_data["growth"])[:, :2]
                  + np.asarray(mesh.point_data["solution"])[:, :2])
        field_vals = np.asarray(mesh.point_data[field])   # (n_points, 3)

        left, right, ref = get_regions(t, cur_xy, expanded=EXPANDED)
        ids_of = {
            "neuromast_left":  left,
            "neuromast_right": right,
            "whole_neuromast": np.concatenate([left, right]),
            "reference":       ref,
        }

        frame_list.append(t)

        for r in REGIONS:
            ids = ids_of[r]
            vals = field_vals[ids]    # (n_pts_in_group, 3)

            for pid, v in zip(ids, vals):
                writers[r].writerow([t, int(pid), v[0], v[1], v[2]])

            if len(vals):
                m = vals.mean(axis=0)
                s = vals.std(axis=0)
            else:
                m = s = np.full(3, np.nan)

            for i, ax in enumerate(AXES_NAMES):
                means[r][ax].append(float(m[i]))
                stds[r][ax].append(float(s[i]))

        if t % 20 == 0 or t == frames[0] or t == frames[-1]:
            print(f"  t={t}: " + " ".join(f"{r}={len(ids_of[r])}" for r in REGIONS))

    for fh in csv_handles.values():
        fh.close()

    # --- figure: 3 rows (x / y / z), 4 curves per panel ---
    frame_arr = np.array(frame_list)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for row, ax_name in enumerate(AXES_NAMES):
        ax = axs[row]
        for r in REGIONS:
            m = np.array(means[r][ax_name])
            s = np.array(stds[r][ax_name])
            color = REGION_COLOR[r]
            ax.plot(frame_arr, m, color=color, label=REGION_LABEL[r], linewidth=1.5)
            ax.fill_between(frame_arr, m - s, m + s, color=color, alpha=0.18)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylabel(ax_name)
        ax.grid(alpha=0.25)

    axs[0].legend(fontsize=9, loc="upper left")
    axs[2].set_xlabel("frame")
    fig.suptitle(f"{field}  —  mean ± std per region, lab frame x / y / z")
    fig.tight_layout()

    png_path = f"{RESULT_DIR}/curves_{field}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  figure  → {png_path}")
    print(f"  CSVs    → {out_dir}/")

print("\ndone")
