"""
Small shared helper, used by a couple of the setup scripts - figures out how
many frames exist and what size they are directly from the data folders, so
nobody has to hand-count files or type image dimensions into config.py (and
then forget to update them if the sequence length changes).
"""
import glob
import re
import tifffile

from config import MESH_DIR, TIF_DIR


def _max_index(pattern):
    numbers = [int(re.search(r"(\d+)", path.rsplit("/", 1)[-1]).group(1))
               for path in glob.glob(pattern)]
    if not numbers:
        raise FileNotFoundError(f"no files matched {pattern!r} - check the data is in place")
    return max(numbers)


def tif_frame_count():
    return _max_index(f"{TIF_DIR}/T_*.tif")


def vtu_frame_count():
    return _max_index(f"{MESH_DIR}/sim_*.vtu")


def tif_image_shape():
    stack = tifffile.imread(f"{TIF_DIR}/T_1.tif")
    return stack.max(axis=0).shape
