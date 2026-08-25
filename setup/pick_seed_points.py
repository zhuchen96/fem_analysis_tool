"""
Not part of the actual pipeline run - just a helper for filling in
config.py's SEED_FRAME / SEED_POINTS_XY on a new sequence.

Point it at a candidate late frame (one where the structure has clearly
split into a front lobe and a back lobe):

    python setup/pick_seed_points.py 204

It saves setup/seed_frame_204.png - the max-projection of that frame with a
light pixel grid drawn over it, so you can read off two (x, y) pixel
coordinates without needing a separate image viewer. Pick one point toward
the reference side and one toward the neuromast side, and drop both plus
the frame number into config.py. Try a couple of different frame numbers if
the first one you check doesn't look clean.
"""
import sys
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    sys.exit("usage: python setup/pick_seed_points.py <frame number>")

t = int(sys.argv[1])
stack = tifffile.imread(f"Time series_flipped/T_{t}.tif")
mip = stack.max(axis=0)
h, w = mip.shape

fig, ax = plt.subplots(figsize=(w / 80, h / 80))
ax.imshow(mip, cmap="gray")
ax.set_xticks(np.arange(0, w, 50))
ax.set_yticks(np.arange(0, h, 50))
ax.tick_params(labelsize=6)
ax.grid(color="lime", alpha=0.4, linewidth=0.5)
ax.set_title(f"T_{t}.tif max projection - read (x, y) pixel coords off the grid")
fig.tight_layout()

out_path = f"setup/seed_frame_{t}.png"
fig.savefig(out_path, dpi=130)
print(f"wrote {out_path}")
print("pick a reference-side point and a neuromast-side point (in that order)")
print("and put them, plus this frame number, into config.py")
