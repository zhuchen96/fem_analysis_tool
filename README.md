# fem analysis pipeline

This folder makes eleven videos out of the fem simulation files: the
tissue at every timepoint gets split into three regions (front half of the
neuromast, back half of the neuromast, and the reference tissue)

Everything here reads straight from the simulation output (`.vtu` mesh
files) and the raw microscopy stack (`.tif` files) and writes a finished
`.mp4`. No intermediate CSVs, no multi-step build process — one script in,
one video out.

## The three regions, briefly

For a given frame, a mesh point's *current* (deformed) x-position decides
where it lands:

- **neuromast_left**
- **neuromast_right** 
- **reference**

There's a small zone at the midline (~6.5 µm) so points don't
flicker back and forth between neuromast and reference from one frame to
the next just because of mesh jitter. There's no such gap at the apical
point — that split is a plain cutoff. All of this logic lives in one place,
`groups.py`, which every other script imports.

A point only counts as being in a region at all if it also falls inside the
segmented embryo outline — either the plain outline, or the same outline
padded out by 5 µm first (the "expanded mask" variants).

## What's in this folder

| script | what it makes |
|---|---|
| `groups.py` | the shared helper the other 11 scripts import, decides which region a point is in |
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
├── smoothed_boundary.npz            <- generated (see below)
├── migration_direction.npz          <- generated, or already included for this sample
└── groups_by_midline/
    ├── midline.npz                  <- generated, or already included for this sample
    └── apical_point.npz             <- generated, or already included for this sample
```

**If you're working with the same sample this repo was built from**, the
three generated files are already committed (they're tiny, a few KB each —
per-frame measurements specific to this sample, not the bulky kind of
data). You just need to link in the raw data itself — about 12 GB of
`.tif` stacks and `.vtu` meshes, plus a 73 MB mask file, all too big for
git:

```bash
ln -s /path/to/your/wildtyp_sample2_with_growth  wildtyp_sample2_with_growth
ln -s /path/to/your/"Time series_flipped"        "Time series_flipped"
ln -s /path/to/your/smoothed_boundary.npz         smoothed_boundary.npz
```

**If this is a brand new sequence** (different embryo, different imaging
session), none of the generated files exist yet — see section 4 below,
"Setting up a new sequence", which walks through producing all four of
them from just the raw `.tif`/`.vtu` data.

Either way, note the scripts only ever look at frames 55 through 230 for
this sample (whatever range is covered by the tracked midline/apical-point
data) — you don't strictly need earlier frames of the tif/vtu series for
the videos, but it doesn't hurt to keep them.

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

## 4. Setting up a new sequence

The four generated files from section 2 (`smoothed_boundary.npz`,
`migration_direction.npz`, `groups_by_midline/midline.npz`,
`groups_by_midline/apical_point.npz`) aren't hand-editable data — they're
each produced by a script, out of the raw `.tif`/`.vtu` files, in the
`setup/` folder. For a new sample, run through `setup/` once and the four
scripts in section 3 will work exactly the same way they do here.

| script | what it makes |
|---|---|
| `setup/config.py` | not a script — the handful of numbers that need a human to look at the images and decide (see below) |
| `setup/01_segment_masks.py` | raw microscopy → a rough per-frame mask of the structure, `raw_masks.npy` |
| `setup/02_smooth_boundary.py` | cleans that mask up through time → `smoothed_boundary.npz` |
| `setup/03_track_landmarks.py` | tracks two landmark points backward through time from a chosen frame → `tracked_points_backward_v2.npz` |
| `setup/04_build_midline.py` | the two landmarks' midpoint, extended across all frames → `groups_by_midline/midline.npz` |
| `setup/05_build_apical_point.py` | one of the two landmarks on its own → `groups_by_midline/apical_point.npz` |
| `setup/06_migration_direction.py` | net migration direction, from the tissue centroid's trajectory → `migration_direction.npz` |
| `setup/pick_seed_points.py` | not part of the pipeline — a helper for finding the landmark points `config.py` needs |
| `setup/_dataset.py`, `setup/_smoothing.py` | not scripts you run — small shared helpers the numbered scripts import (frame counting, the mask-smoothing math) |

Most of that runs unattended. The one part that doesn't: `config.py` needs
two points clicked on an image by a person, since there's no automatic way
to know where this particular structure's front and back landmarks are.
Open it and read through the comments — it explains what each value means
and how to find it, in particular:

```bash
python setup/pick_seed_points.py 204
```

pick a late frame (try a few numbers — anywhere the structure has clearly
split into a front lobe and a back lobe works), which saves a PNG with a
pixel grid over the image so you can read off two `(x, y)` coordinates
without a separate viewer.

Once `config.py` is filled in, run the whole chain from the repo root:

```bash
bash setup/run_all.sh
```

or run the six numbered scripts one at a time if you want to check the
output of each step before moving on — each one prints its own progress
and tells you which script to run next. `01` and `03` are the slow ones
(reading every raw frame); the rest are quick.

One easy-to-miss detail: `setup/config.py` has its own copy of `SPACING_XY`
(micron per pixel), and so does `groups.py` at the repo root. If a new
sequence was imaged at a different resolution, update it in **both**
places — they're kept separate on purpose (you shouldn't need `setup/` at
all if you already have the four generated files), but that means nothing
will warn you if they drift apart.

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
- **`setup/04_build_midline.py` prints a WARNING about landmark order** —
  `config.py`'s `SEED_POINTS_XY` has the two points backward. Point 0 needs
  to be the one further toward the reference side (larger x).
- **`setup/06_migration_direction.py` fails to import `groups`** — it needs
  to be run from the repo root (`python setup/06_migration_direction.py`),
  not from inside `setup/`.
