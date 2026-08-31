"""
Computes the magnitude of the pk1_stess_3 vector (|stress|) at every mesh
node for each frame in [T_START, SEED_FRAME], then plots mean ± std over
time for all four regions.

Outputs:
  result_sampleN/curves_pk1_stess_3/
      magnitude_neuromast_left.csv    frame, point_id, magnitude
      magnitude_neuromast_right.csv
      magnitude_whole_neuromast.csv
      magnitude_reference.csv
  result_sampleN/curves_pk1_stess_3_magnitude.png
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

FIELD = "pk1_stess_3"
EXPANDED = False

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

out_dir = f"{RESULT_DIR}/curves_{FIELD}"
os.makedirs(out_dir, exist_ok=True)

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
frames = mid["frames"].astype(int)
frames = frames[(frames >= T_START) & (frames <= SEED_FRAME)]

ref_mesh = pv.read(f"{MESH_DIR}/sim_{VTU_START}.vtu")
ref_xy = ref_mesh.points[:, :2]

means = {r: [] for r in REGIONS}
stds  = {r: [] for r in REGIONS}
frame_list = []

csv_handles = {}
writers = {}
for r in REGIONS:
    fh = open(f"{out_dir}/magnitude_{r}.csv", "w", newline="")
    csv_handles[r] = fh
    writers[r] = csv.writer(fh)
    writers[r].writerow(["frame", "point_id", "magnitude"])

for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t - TIF_START + VTU_START}.vtu")
    cur_xy = (ref_xy
              + np.asarray(mesh.point_data["growth"])[:, :2]
              + np.asarray(mesh.point_data["solution"])[:, :2])
    field_vals = np.asarray(mesh.point_data[FIELD])          # (n_points, 3)
    magnitudes = np.linalg.norm(field_vals, axis=1)          # (n_points,)

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
        mags = magnitudes[ids]

        for pid, mag in zip(ids, mags):
            writers[r].writerow([t, int(pid), float(mag)])

        if len(mags):
            means[r].append(float(mags.mean()))
            stds[r].append(float(mags.std()))
        else:
            means[r].append(np.nan)
            stds[r].append(np.nan)

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: " + " ".join(f"{r}={len(ids_of[r])}" for r in REGIONS))

for fh in csv_handles.values():
    fh.close()

frame_arr = np.array(frame_list)
fig, ax = plt.subplots(figsize=(12, 5))

for r in REGIONS:
    m = np.array(means[r])
    s = np.array(stds[r])
    color = REGION_COLOR[r]
    ax.plot(frame_arr, m, color=color, label=REGION_LABEL[r], linewidth=1.5)
    ax.fill_between(frame_arr, m - s, m + s, color=color, alpha=0.18)

ax.set_xlabel("frame")
ax.set_ylabel("|pk1_stess_3|")
ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
ax.grid(alpha=0.25)
ax.legend(fontsize=9)
fig.suptitle("pk1_stess_3 magnitude — mean ± std per region over time")
fig.tight_layout()

png_path = f"{RESULT_DIR}/curves_{FIELD}_magnitude.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"figure  → {png_path}")
print(f"CSVs    → {out_dir}/magnitude_*.csv")
