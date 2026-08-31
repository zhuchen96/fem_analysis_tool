"""
Rose-histogram video for the "solution" field (cumulative displacement from
the first frame). For each of the three regions, bins the direction each
point has moved in, viewed in three planes of the migration-aligned frame:

    M = along the direction of migration
    P = sideways, perpendicular to migration, still in-plane
    Z = out of plane

Plain (non-expanded) segmentation mask - see the _expanded_mask twin for
that variant.
"""
import os
import numpy as np
import pyvista as pv
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from groups import get_regions
from config import MESH_DIR, RESULT_DIR, T_START, SEED_FRAME, TIF_START, VTU_START

FRAME_DIR = f"{RESULT_DIR}/angle_histograms_solution"
VIDEO_OUT = f"{RESULT_DIR}/angle_histograms_solution.mp4"
FIELD = "solution"
EXPANDED = False
N_BINS = 24

REGIONS = ["neuromast_left", "neuromast_right", "reference", "whole_neuromast"]
REGION_LABEL = {
    "neuromast_left": "neuromast, left",
    "neuromast_right": "neuromast, right",
    "reference": "reference tissue",
    "whole_neuromast": "whole neuromast",
}
PLANES = [("M vs P", 0, 1), ("M vs Z", 0, 2), ("P vs Z", 1, 2)]

os.makedirs(FRAME_DIR, exist_ok=True)

ref_mesh = pv.read(f"{MESH_DIR}/sim_{VTU_START}.vtu")
ref_xy = ref_mesh.points[:, :2]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
frames = mid["frames"].astype(int)
frames = frames[(frames >= T_START) & (frames <= SEED_FRAME)]

mig = np.load(f"{RESULT_DIR}/migration_direction.npz")
ux, uy = mig["unit"]
px, py = -uy, ux  # perpendicular to the migration direction, in-plane


def to_migration_frame(vecs):
    m = vecs[:, 0] * ux + vecs[:, 1] * uy
    p = vecs[:, 0] * px + vecs[:, 1] * py
    return np.stack([m, p, vecs[:, 2]], axis=1)


writer = imageio.get_writer(VIDEO_OUT, fps=12)

for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t - TIF_START + VTU_START}.vtu")
    cur_xy = ref_xy + np.asarray(mesh.point_data["growth"])[:, :2] + np.asarray(mesh.point_data["solution"])[:, :2]
    field_mig = to_migration_frame(np.asarray(mesh.point_data[FIELD]))

    left, right, ref = get_regions(t, cur_xy, expanded=EXPANDED)
    cur_ids = {"neuromast_left": np.array(sorted(left), dtype=int),
               "neuromast_right": np.array(sorted(right), dtype=int),
               "reference": np.array(sorted(ref), dtype=int)}
    cur_ids["whole_neuromast"] = np.concatenate([cur_ids["neuromast_left"], cur_ids["neuromast_right"]])

    fig, axes = plt.subplots(4, 3, subplot_kw={"projection": "polar"}, figsize=(12, 15))
    for row, r in enumerate(REGIONS):
        ids = cur_ids[r]
        vecs = field_mig[ids] if len(ids) else np.zeros((0, 3))
        for col, (plane_name, a, b) in enumerate(PLANES):
            ax = axes[row, col]
            if len(ids):
                angles = np.degrees(np.arctan2(vecs[:, b], vecs[:, a]))
                counts, edges = np.histogram(angles, bins=N_BINS, range=(-180, 180))
            else:
                counts = np.zeros(N_BINS)
                edges = np.linspace(-180, 180, N_BINS + 1)
            theta = np.radians((edges[:-1] + edges[1:]) / 2)
            ax.bar(theta, counts, width=np.radians(360 / N_BINS),
                   color="darkorange", edgecolor="k", linewidth=0.3)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_title(f"{REGION_LABEL[r]} - {plane_name} (n={len(ids)})", fontsize=8.5)
    fig.suptitle(f"solution field, t={t} - direction of cumulative displacement, migration frame")
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=110)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: n = " + " ".join(f"{r}={len(cur_ids[r])}" for r in REGIONS))

writer.close()
print(f"wrote {VIDEO_OUT}")
