"""
pk1_stess_3 version of make_histogram_video_solution_expanded_mask.py -
expanded (padded) segmentation mask, everything else the same as
make_histogram_video_stress.py.
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

FRAME_DIR = f"{RESULT_DIR}/angle_histograms_pk1_stess_3_expanded_mask"
VIDEO_OUT = f"{RESULT_DIR}/angle_histograms_pk1_stess_3_expanded_mask.mp4"
FIELD = "pk1_stess_3"
EXPANDED = True
N_BINS = 24

REGIONS = ["neuromast_left", "neuromast_right", "reference"]
REGION_LABEL = {
    "neuromast_left": "neuromast, front half",
    "neuromast_right": "neuromast, back half",
    "reference": "reference tissue",
}
PLANES = [("M vs P", 0, 1), ("M vs Z", 0, 2), ("P vs Z", 1, 2)]

os.makedirs(FRAME_DIR, exist_ok=True)

ref_mesh = pv.read(f"{MESH_DIR}/sim_1.vtu")
ref_xy = ref_mesh.points[:, :2]

mid = np.load(f"{RESULT_DIR}/groups_by_midline/midline.npz")
frames = mid["frames"].astype(int)

mig = np.load(f"{RESULT_DIR}/migration_direction.npz")
ux, uy = mig["unit"]
px, py = -uy, ux


def to_migration_frame(vecs):
    m = vecs[:, 0] * ux + vecs[:, 1] * uy
    p = vecs[:, 0] * px + vecs[:, 1] * py
    return np.stack([m, p, vecs[:, 2]], axis=1)


writer = imageio.get_writer(VIDEO_OUT, fps=12)
prev_ids = None

for t in frames:
    t = int(t)
    mesh = pv.read(f"{MESH_DIR}/sim_{t}.vtu")
    cur_xy = ref_xy + np.asarray(mesh.point_data["solution"])[:, :2]
    field_mig = to_migration_frame(np.asarray(mesh.point_data[FIELD]))

    left, right, ref = get_regions(t, cur_xy, expanded=EXPANDED)
    cur_ids = {"neuromast_left": set(left), "neuromast_right": set(right), "reference": set(ref)}

    if prev_ids is None:
        stable = {r: np.array([], dtype=int) for r in REGIONS}
    else:
        stable = {r: np.array(sorted(cur_ids[r] & prev_ids[r]), dtype=int) for r in REGIONS}

    fig, axes = plt.subplots(3, 3, subplot_kw={"projection": "polar"}, figsize=(12, 11.5))
    for row, r in enumerate(REGIONS):
        ids = stable[r]
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
    fig.suptitle(f"pk1_stess_3 field, t={t} - current stress direction, migration frame (expanded mask)")
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=110)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

    prev_ids = cur_ids
    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: stable points - "
              + " ".join(f"{r}={len(stable[r])}" for r in REGIONS))

writer.close()
print(f"wrote {VIDEO_OUT}")
