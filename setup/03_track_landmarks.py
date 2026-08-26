"""
Step 3 of 6: track the two landmark points (config.py's SEED_POINTS_XY)
backward through time from SEED_FRAME, one frame at a time. Each step
nudges the point onto the local intensity centroid, then finds where that
patch best matches in the previous frame (normalized cross-correlation, in
a small search window around the current position).

If a point's match confidence ever drops too low - it's wandered off the
real structure and onto background - tracking for that point just stops
there rather than freezing at a stale position for all earlier frames.

Run from the repo root:
    python setup/03_track_landmarks.py

Writes tracked_points_backward_v2.npz to the repo root.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tifffile
from skimage.feature import match_template

from config import SEED_FRAME, SEED_POINTS_XY, TIF_DIR, RESULT_DIR

TEMPLATE_RADIUS = 20
SEARCH_MARGIN = 20
SCORE_FLOOR = 0.4  # match confidence below this means the point has drifted off the real structure


def load_mip(t):
    stack = tifffile.imread(f"{TIF_DIR}/T_{t}.tif")
    return stack.max(axis=0).astype(np.float32)


def extract_patch(img, cx, cy, r):
    H, W = img.shape
    x0, x1 = cx - r, cx + r + 1
    y0, y1 = cy - r, cy + r + 1
    patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=img.dtype)
    xs0, xs1 = max(x0, 0), min(x1, W)
    ys0, ys1 = max(y0, 0), min(y1, H)
    patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    return patch


def centroid_refine(img, cx, cy, r):
    H, W = img.shape
    x0, x1 = cx - r, cx + r + 1
    y0, y1 = cy - r, cy + r + 1
    xs0, xs1 = max(x0, 0), min(x1, W)
    ys0, ys1 = max(y0, 0), min(y1, H)
    patch = img[ys0:ys1, xs0:xs1].astype(np.float64)
    ys, xs = np.mgrid[ys0:ys1, xs0:xs1]
    total = patch.sum()
    if total <= 0:
        return float(cx), float(cy)
    rx = (xs * patch).sum() / total
    ry = (ys * patch).sum() / total
    return rx, ry


def track_backward(seed_x, seed_y, images):
    cx, cy = float(seed_x), float(seed_y)
    positions = {SEED_FRAME: (cx, cy)}
    scores = {}
    stopped_at = None

    for t in range(SEED_FRAME, 1, -1):
        img_cur = images[t]

        rx, ry = centroid_refine(img_cur, int(round(cx)), int(round(cy)), TEMPLATE_RADIUS)
        cx, cy = rx, ry
        positions[t] = (cx, cy)

        template = extract_patch(img_cur, int(round(cx)), int(round(cy)), TEMPLATE_RADIUS).astype(np.float32)

        img_prev = images[t - 1]
        H, W = img_prev.shape
        margin = TEMPLATE_RADIUS + SEARCH_MARGIN
        cxi, cyi = int(round(cx)), int(round(cy))
        x0, x1 = cxi - margin, cxi + margin + 1
        y0, y1 = cyi - margin, cyi + margin + 1
        xs0, xs1 = max(x0, 0), min(x1, W)
        ys0, ys1 = max(y0, 0), min(y1, H)
        window = img_prev[ys0:ys1, xs0:xs1]

        if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
            stopped_at = t - 1
            break

        result = match_template(window, template)
        best = np.unravel_index(np.argmax(result), result.shape)
        score = float(result[best])

        if score < SCORE_FLOOR:
            stopped_at = t - 1
            break

        match_cy = ys0 + best[0] + TEMPLATE_RADIUS
        match_cx = xs0 + best[1] + TEMPLATE_RADIUS
        scores[t - 1] = score
        cx, cy = float(match_cx), float(match_cy)
        positions[t - 1] = (cx, cy)

    return positions, scores, stopped_at


if __name__ == "__main__":
    print(f"loading MIPs 1..{SEED_FRAME}")
    images = {t: load_mip(t) for t in range(1, SEED_FRAME + 1)}

    all_positions = []
    all_scores = []
    stop_frames = []
    for (sx, sy) in SEED_POINTS_XY:
        positions, scores, stopped_at = track_backward(sx, sy, images)
        all_positions.append(positions)
        all_scores.append(scores)
        stop_frames.append(stopped_at)
        last_frame = min(positions.keys())
        print(f"seed ({sx},{sy}): tracked frames {SEED_FRAME}->{last_frame}"
              + (f" (stopped: low match score below frame {stopped_at + 1})" if stopped_at is not None else " (reached frame 1)"))

    n_points = len(SEED_POINTS_XY)
    traj_xy = np.full((n_points, SEED_FRAME, 2), np.nan)
    traj_score = np.full((n_points, SEED_FRAME), np.nan)
    for p, positions in enumerate(all_positions):
        for t, (x, y) in positions.items():
            traj_xy[p, t - 1] = (x, y)
    for p, scores in enumerate(all_scores):
        for t, s in scores.items():
            traj_score[p, t - 1] = s

    np.savez(f"{RESULT_DIR}/tracked_points_backward_v2.npz",
             seed_frame=SEED_FRAME, seed_points_xy=np.array(SEED_POINTS_XY),
             traj_xy=traj_xy, traj_score=traj_score,
             stop_frame=np.array([s if s is not None else -1 for s in stop_frames]))
    print(f"saved {RESULT_DIR}/tracked_points_backward_v2.npz - run setup/04_build_midline.py next")
