import os
import glob
import re
import numpy as np
import pyvista as pv
import tifffile
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MESH_DIR, TIF_DIR, RESULT_DIR

FRAME_DIR = f"{RESULT_DIR}/overlay_plain"
VIDEO_OUT = f"{RESULT_DIR}/overlay_plain.mp4"
SPACING_XY = 0.325  # micron per pixel

os.makedirs(FRAME_DIR, exist_ok=True)

frames = sorted(
    int(re.search(r"(\d+)", os.path.basename(p)).group(1))
    for p in glob.glob(f"{MESH_DIR}/sim_*.vtu")
)
frames = [t for t in frames if t >= 1]

ref_mesh = pv.read(f"{MESH_DIR}/sim_1.vtu")
ref_xy = ref_mesh.points[:, :2]

writer = imageio.get_writer(VIDEO_OUT, fps=12)

for t in frames:
    mesh = pv.read(f"{MESH_DIR}/sim_{t}.vtu")
    cur_xy = ref_xy + np.asarray(mesh.point_data["growth"])[:, :2] + np.asarray(mesh.point_data["solution"])[:, :2]

    stack = tifffile.imread(f"{TIF_DIR}/T_{t}.tif")
    mip = stack.max(axis=0)
    h, w = mip.shape
    extent = [0, w * SPACING_XY, 0, h * SPACING_XY]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.imshow(mip, cmap="gray", origin="lower", extent=extent)
    ax.scatter(cur_xy[:, 0], cur_xy[:, 1], s=1.0, c="white", alpha=0.4, linewidths=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (micron)")
    ax.set_ylabel("y (micron)")
    ax.set_title(f"t={t}", fontsize=9)
    fig.tight_layout()

    frame_path = f"{FRAME_DIR}/frame_{t:03d}.png"
    fig.savefig(frame_path, dpi=130)
    plt.close(fig)
    writer.append_data(imageio.imread(frame_path))

    if t % 20 == 0 or t == frames[0] or t == frames[-1]:
        print(f"t={t}: {len(cur_xy)} points")

writer.close()
print(f"wrote {VIDEO_OUT}")
