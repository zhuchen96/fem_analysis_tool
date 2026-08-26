#!/bin/bash
# Runs the whole setup chain in order, for a sequence whose config.py you've
# already filled in (SPACING_XY, SEED_FRAME, SEED_POINTS_XY, T_START).
#
# Run from the repo root, not from inside setup/:
#     bash setup/run_all.sh
set -e

python setup/01_segment_masks.py
python setup/02_smooth_boundary.py
python setup/03_track_landmarks.py
python setup/04_build_midline.py
python setup/05_build_apical_point.py
python setup/06_migration_direction.py

echo
echo "all set - outputs are in result_sampleN/ (see config.py for SAMPLE_ID)."
echo "you can now run any of the make_*_video.py scripts in the repo root."
