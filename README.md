# fem analysis pipeline

This folder makes twelve videos out of the fem simulation files: the
tissue at every timepoint gets split into three regions (front half of the
neuromast, back half of the neuromast, and the reference)

Everything here reads straight from the simulation output (`.vtu` mesh
files) and the raw microscopy stack (`.tif` files) and writes a finished
`.mp4`. No intermediate CSVs, no multi-step build process.

## Quick start

**1. Activate the environment**
```bash
conda activate fem_analysis
```

**2. Put your data in place**

Name your folders following the `sampleN` convention and set `SAMPLE_ID` in `config.py`:
```bash
ln -s /path/to/vtu_files   vtu_files_sample1
ln -s /path/to/tif_files   tif_files_sample1
```
```python
# config.py
SAMPLE_ID = 1
```

**3. Run the plain overlay video**

This overlays all mesh nodes on the raw microscopy images with no region split — use it to check that the mesh and images are aligned before going further:
```bash
python make_overlay_video_plain.py
# output: result_sample1/overlay_plain.mp4
```

**4. Pick a seed frame for landmark tracking**

Choose a late frame where the neuromast has clearly split into two lobes. Generate a pixel-grid image to read off coordinates:
```bash
python setup/pick_seed_points.py 204
# saves setup/seed_frame_204.png — open it and read off two (x, y) coordinates
```
Set `SEED_FRAME` in `config.py` to the frame number you chose.

**5. Set the landmark coordinates in config**

Open `setup/seed_frame_N.png`. Pick two points:
- point 0: the **reference-side** landmark (larger x)
- point 1: the **apical constriction point** toward the neuromast side (smaller x)

```python
# config.py
SEED_POINTS_XY = [
    (884, 183),  # reference-side landmark
    (601, 189),  # apical constriction point
]
```

**6. Set the start frame**

This is the earliest frame where the two landmarks are clearly separated. Look at the plain overlay video from step 3 to judge — if unsure, run setup step 03 first and check how far back it tracked.
```python
# config.py
T_START = 55
```

**7. Run the setup chain**
```bash
bash setup/run_all.sh
# produces result_sample1/smoothed_boundary.npz, groups_by_midline/*.npz, migration_direction.npz
```

**8. Generate all videos**
```bash
for f in make_overlay_video_plain.py \
         make_overlay_video.py make_overlay_video_expanded_mask.py make_overlay_video_growth.py \
         make_histogram_video_solution.py make_histogram_video_solution_expanded_mask.py \
         make_histogram_video_stress.py make_histogram_video_stress_expanded_mask.py \
         make_vector_video_solution.py make_vector_video_solution_expanded_mask.py \
         make_vector_video_stress.py make_vector_video_stress_expanded_mask.py; do
    echo "=== $f ==="
    python "$f"
done
# all outputs in result_sample1/
```

---

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
| `config.py` | not a script — set `SAMPLE_ID` and sample-specific parameters here; used by all scripts |
| `groups.py` | the shared helper the other 12 scripts import, decides which region a point is in |
| `make_overlay_video_plain.py` | all mesh nodes as white dots on the raw microscopy image, no region split |
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
conda activate fem_analysis
```

That installs numpy, scipy, scikit-image, matplotlib, pyvista/vtk,
tifffile, and imageio (with the ffmpeg plugin needed to write `.mp4`
files — it ships its own ffmpeg binary, you don't need to separately
`brew install ffmpeg`).

## 2. Getting the data in place

### Setting the sample ID

Open `config.py` at the repo root and set `SAMPLE_ID` to whichever sample
you want to work with:

```python
SAMPLE_ID = 1
```

This controls three things at once: which data directories the scripts read
from, and where they write their outputs:

| `SAMPLE_ID` | mesh input | microscopy input | all outputs |
|---|---|---|---|
| `1` | `vtu_files_sample1/` | `tif_files_sample1/` | `result_sample1/` |
| `2` | `vtu_files_sample2/` | `tif_files_sample2/` | `result_sample2/` |
| … | … | … | … |

If your data directories are named differently, just override `MESH_DIR` and
`TIF_DIR` directly in `config.py` instead of relying on the derived names.

### Expected layout

```
fem_analysis_pipeline/
├── config.py                              <- set SAMPLE_ID here
├── vtu_files_sample1/                     <- you provide (simulation output)
│   ├── sim_1.vtu
│   └── ... sim_230.vtu
├── tif_files_sample1/                     <- you provide (raw microscopy stack)
│   ├── T_1.tif
│   └── ... T_232.tif
└── result_sample1/                        <- generated by setup/ and the video scripts
    ├── smoothed_boundary.npz
    ├── migration_direction.npz
    └── groups_by_midline/
        ├── midline.npz
        └── apical_point.npz
```

Link the data in from wherever you keep it:

```bash
ln -s /path/to/your/vtu_files_sample1  vtu_files_sample1
ln -s /path/to/your/tif_files_sample1  tif_files_sample1
```

**If the setup files for this sample already exist** (i.e. `result_sample1/`
is already populated), you can skip straight to section 3.

**If this is a brand new sequence**, none of the generated files exist yet —
see section 4, "Setting up a new sequence".

## 3. Running it

Once the environment is active and the data is linked in, just run
whichever script you want from inside this folder:

```bash
python make_overlay_video.py
```

Each script prints its progress (region point-counts every 20 frames or
so) and, when it finishes, writes into `result_sample{N}/`:

- a subfolder of individual PNG frames (e.g. `result_sample1/overlay_groups_by_midline/`)
- the finished video next to it (e.g. `result_sample1/overlay_groups_by_midline.mp4`)

You can run all twelve back to back if you want everything at once:

```bash
for f in make_overlay_video_plain.py \
         make_overlay_video.py make_overlay_video_expanded_mask.py make_overlay_video_growth.py \
         make_histogram_video_solution.py make_histogram_video_solution_expanded_mask.py \
         make_histogram_video_stress.py make_histogram_video_stress_expanded_mask.py \
         make_vector_video_solution.py make_vector_video_solution_expanded_mask.py \
         make_vector_video_stress.py make_vector_video_stress_expanded_mask.py; do
    echo "=== $f ==="
    python "$f"
done
```

Each one reads all mesh frames, so budget a few minutes per script on
a laptop — the vector-arrow scripts read the meshes twice (once to figure
out sensible fixed axis scales, once to draw), so those run a bit slower
than the others.

## 4. Setting up a new sequence

The generated files under `result_sample{N}/` (`smoothed_boundary.npz`,
`migration_direction.npz`, `groups_by_midline/midline.npz`,
`groups_by_midline/apical_point.npz`) aren't hand-editable data — they're
each produced by a script, out of the raw `.tif`/`.vtu` files, in the
`setup/` folder. For a new sample, run through `setup/` once and the
video scripts in section 3 will work exactly the same way.

| script | what it makes |
|---|---|
| `setup/01_segment_masks.py` | raw microscopy → a rough per-frame mask of the structure, `result_sampleN/raw_masks.npy` |
| `setup/02_smooth_boundary.py` | cleans that mask up through time → `result_sampleN/smoothed_boundary.npz` |
| `setup/03_track_landmarks.py` | tracks two landmark points backward through time from a chosen frame → `result_sampleN/tracked_points_backward_v2.npz` |
| `setup/04_build_midline.py` | the two landmarks' midpoint, extended across all frames → `result_sampleN/groups_by_midline/midline.npz` |
| `setup/05_build_apical_point.py` | one of the two landmarks on its own → `result_sampleN/groups_by_midline/apical_point.npz` |
| `setup/06_migration_direction.py` | net migration direction, from the tissue centroid's trajectory → `result_sampleN/migration_direction.npz` |
| `setup/pick_seed_points.py` | not part of the pipeline — a helper for finding the landmark points `setup/config.py` needs |
| `setup/_dataset.py`, `setup/_smoothing.py` | not scripts you run — small shared helpers the numbered scripts import (frame counting, the mask-smoothing math) |

Before running setup for a new sample, edit `config.py` (repo root — the
only config file): set `SAMPLE_ID`, and fill in `SEED_FRAME`,
`SEED_POINTS_XY`, and `T_START` for the new sample.

`config.py` needs two points clicked on an image by a person, since
there's no automatic way to know where this particular structure's front and
back landmarks are. Open it and read through the comments, then run:

```bash
python setup/pick_seed_points.py 204
```

Pick a late frame (try a few numbers — anywhere the structure has clearly
split into a front lobe and a back lobe works), which saves a PNG with a
pixel grid over the image so you can read off two `(x, y)` coordinates
without a separate viewer.

Once `setup/config.py` is filled in, run the whole chain from the repo root:

```bash
bash setup/run_all.sh
```

or run the six numbered scripts one at a time if you want to check the
output of each step before moving on — each one prints its own progress
and tells you which script to run next. `01` and `03` are the slow ones
(reading every raw frame); the rest are quick.

All parameters live in a single `config.py` at the repo root — there is no
separate `setup/config.py` any more.

## Troubleshooting

- **`ModuleNotFoundError`** — make sure you actually ran
  `conda activate fem_analysis` in this shell session, not just
  `conda env create`.
- **Script can't find `sim_55.vtu` / `T_55.tif` / etc.** — check that the
  symlinks (or copies) match the names in `config.py`. Run
  `ls vtu_files_sample1 | head` (or whatever `MESH_DIR` is set to) to
  sanity-check.
- **Script can't find `result_sampleN/smoothed_boundary.npz`** — the setup
  chain hasn't been run for this sample yet. Run `bash setup/run_all.sh`
  first (after filling in `setup/config.py`).
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
