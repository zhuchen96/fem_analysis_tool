# neuromast region-split pipeline

This folder makes eleven videos out of a zebrafish neuromast simulation: the
tissue at every timepoint gets split into three regions (front half of the
neuromast, back half of the neuromast, and the reference tissue behind it),
and each script visualizes some aspect of that split — where the regions
sit on the actual microscopy image, which way points inside each region are
moving, or how stress is distributed across them.

Everything here reads straight from the simulation output (`.vtu` mesh
files) and the raw microscopy stack (`.tif` files) and writes a finished
`.mp4`. No intermediate CSVs, no multi-step build process — one script in,
one video out.

## The three regions, briefly

For a given frame, a mesh point's *current* (deformed) x-position decides
where it lands:

- **neuromast_left** — in front of the tracked apical constriction point
- **neuromast_right** — between the apical point and the midline
- **reference** — behind the midline

There's a small dead zone straddling the midline (~6.5 µm) so points don't
flicker back and forth between neuromast and reference from one frame to
the next just because of mesh jitter. There's no such gap at the apical
point — that split is a plain cutoff. All of this logic lives in one place,
`groups.py`, which every other script imports.

A point only counts as being in a region at all if it also falls inside the
segmented embryo outline — either the plain outline, or the same outline
padded out by 5 µm first (the "expanded mask" variants, which catch points
that sit just outside the segmentation, more common near the tail than
you'd expect).

## What's in this folder

| script | what it makes |
|---|---|
| `groups.py` | not a video — the shared helper the other 11 scripts import, decides which region a point is in |
| `make_overlay_video.py` | the three regions plotted on top of the actual microscopy image, frame by frame |
| `make_overlay_video_expanded_mask.py` | same, but using the padded (expanded) mask |
| `make_overlay_video_growth.py` | same idea but for the `growth` field and the simpler original 2-region split (neuromast / reference, no front/back, no gap) |
| `make_histogram_video_solution.py` | rose histograms of movement direction (the `solution` displacement field), per region, in the migration-aligned frame |
| `make_histogram_video_solution_expanded_mask.py` | same, padded mask |
| `make_histogram_video_stress.py` | same rose-histogram layout, but for the `pk1_stess_3` stress field |
| `make_histogram_video_stress_expanded_mask.py` | same, padded mask |
| `make_vector_video_solution.py` | simpler summary than the rose histogram — one arrow per region per axis (x/y/z), showing the average `solution` displacement |
| `make_vector_video_solution_expanded_mask.py` | same, padded mask |
| `make_vector_video_stress.py` | same arrow layout, averaging `pk1_stess_3` instead |
| `make_vector_video_stress_expanded_mask.py` | same, padded mask |

Each `make_*_video.py` script is standalone — you can run any one of them
on its own, in any order.

## 1. Setting up on a Mac

You'll need conda (this uses `pyvista`/`vtk` to read the mesh files, which
is much easier to install through conda than plain pip). If you don't have
it already, install [Miniforge](https://github.com/conda-forge/miniforge)
(works on both Apple Silicon and Intel Macs — grab the installer for your
chip from that page, or via Homebrew: `brew install miniforge`).

Then, from inside this folder:

```bash
conda env create -f environment.yml
conda activate neuromast_pipeline
```

That installs numpy, scipy, scikit-image, matplotlib, pyvista/vtk,
tifffile, and imageio (with the ffmpeg plugin needed to write `.mp4`
files — it ships its own ffmpeg binary, you don't need to separately
`brew install ffmpeg`).

## 2. Getting the data in place

The scripts expect this layout, relative to this folder:

```
neuromast_pipeline/
├── wildtyp_sample2_with_growth/     <- you provide this (simulation output)
│   ├── sim_1.vtu
│   ├── sim_2.vtu
│   └── ... sim_230.vtu
├── Time series_flipped/             <- you provide this (raw microscopy stack)
│   ├── T_1.tif
│   ├── T_2.tif
│   └── ... T_232.tif
├── smoothed_boundary.npz            <- you provide this (segmentation mask stack)
├── migration_direction.npz          <- already included
└── groups_by_midline/
    ├── midline.npz                  <- already included
    └── apical_point.npz             <- already included
```

The three "already included" files are tiny (a few KB each) — they're
per-frame measurements specific to this sample (the tracked midline
position, the tracked apical-constriction point, and the overall migration
direction), so they're just committed straight into the repo.

The raw data is the opposite story — about 12 GB of `.tif` stacks and
`.vtu` meshes, plus a 73 MB mask file — way too big to put in git. Copy
those three items in from wherever your data currently lives (or symlink
them, that works fine too):

```bash
ln -s /path/to/your/wildtyp_sample2_with_growth  wildtyp_sample2_with_growth
ln -s /path/to/your/"Time series_flipped"        "Time series_flipped"
ln -s /path/to/your/smoothed_boundary.npz         smoothed_boundary.npz
```

A couple of notes on what those files actually are, in case you're
regenerating them rather than copying them:

- `smoothed_boundary.npz` needs one array, `smoothed_masks`, a boolean
  stack of shape `(n_frames, height, width)` — one segmentation mask per
  frame, indexed so frame `t` lives at `smoothed_masks[t - 1]`.
- The scripts only ever look at frames 55 through 230 (that's the range
  covered by the tracked midline/apical-point data), so you don't strictly
  need earlier frames of the tif/vtu series for these videos, but it
  doesn't hurt to keep them.

## 3. Running it

Once the environment is active and the data is linked in, just run
whichever script you want from inside this folder:

```bash
python make_overlay_video.py
```

Each script prints its progress (region point-counts every 20 frames or
so) and, when it finishes, writes:

- a folder of individual PNG frames (e.g. `overlay_groups_by_midline/`)
- the finished video next to it (e.g. `overlay_groups_by_midline.mp4`)

You can run all eleven back to back if you want everything at once:

```bash
for f in make_overlay_video.py make_overlay_video_expanded_mask.py make_overlay_video_growth.py \
         make_histogram_video_solution.py make_histogram_video_solution_expanded_mask.py \
         make_histogram_video_stress.py make_histogram_video_stress_expanded_mask.py \
         make_vector_video_solution.py make_vector_video_solution_expanded_mask.py \
         make_vector_video_stress.py make_vector_video_stress_expanded_mask.py; do
    echo "=== $f ==="
    python "$f"
done
```

Each one reads all 176 mesh frames, so budget a few minutes per script on
a laptop — the vector-arrow scripts read the meshes twice (once to figure
out sensible fixed axis scales, once to draw), so those run a bit slower
than the others.

## Troubleshooting

- **`ModuleNotFoundError`** — make sure you actually ran
  `conda activate neuromast_pipeline` in this shell session, not just
  `conda env create`.
- **Script can't find `sim_55.vtu` / `T_55.tif` / etc.** — check the
  symlinks (or copies) from step 2 landed in this folder, not one level up
  or down. Run `ls wildtyp_sample2_with_growth | head` to sanity check.
- **`KeyError` on `smoothed_masks`** — you've probably got a different
  version of that file (e.g. `raw_masks.npy` instead of the smoothed one).
  You need the one with the `smoothed_masks` array specifically, not the
  raw unsmoothed masks.
- **No sound/video, but a `.mp4` file with 0 bytes** — this usually means
  `imageio-ffmpeg` didn't install correctly. Try
  `pip install --force-reinstall imageio-ffmpeg` inside the activated
  environment.
