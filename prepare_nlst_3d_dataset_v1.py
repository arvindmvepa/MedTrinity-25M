"""
preprocess_nlst.py  –  convert NLST DICOM series to fixed-size .npy volumes

Example
-------
python preprocess_nlst.py \
    --input_dir  /local/amvepa91/nlst_vqa_subset/manifest-1743585557797/NLST \
    --output_dir /local/amvepa91/nlst_npy \
    --min_slices 10 \
    --skip_existing
"""
import argparse, os, sys, traceback
from multiprocessing import Pool

import numpy as np
import pydicom
import monai.transforms as mtf
from tqdm import tqdm
import SimpleITK as sitk

# ------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",  required=True, help="root of raw NLST tree")
parser.add_argument("--output_dir", required=True, help="root where *.npy go")
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--min_slices", type=int, default=20,
                    help="skip series with fewer slices than this (0 = keep all)")
parser.add_argument("--skip_existing", action="store_true",
                    help="do NOT overwrite .npy files that already exist")
args = parser.parse_args()

in_root  = os.path.abspath(args.input_dir)
out_root = os.path.abspath(args.output_dir)
os.makedirs(out_root, exist_ok=True)


# ------------------------------------------------------------------------- #
# Utility: read one DICOM series ➜ np.ndarray  [D, H, W]  (float32)
# ------------------------------------------------------------------------- #
def load_dicom_series(series_dir: str, min_slices=20) -> np.ndarray:
    """Read all *.dcm in *series_dir*, sort, return stacked volume."""
    dcm_files = [os.path.join(series_dir, f)
                 for f in os.listdir(series_dir)
                 if f.lower().endswith(".dcm")]
    if not dcm_files:
        raise RuntimeError(f"No DICOM files in {series_dir}")
    if len(dcm_files) < min_slices:
        raise RuntimeError(f"Too few DICOM files in {series_dir} (found {len(dcm_files)})")

    reader = sitk.ImageSeriesReader()
    # Get series file names (handles weird ordering)
    dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
    reader.SetFileNames(dicom_names)
    # Read the volume
    image = reader.Execute()

    img_np = sitk.GetArrayFromImage(image)  
    return img_np


# ------------------------------------------------------------------------- #
# Worker
# ------------------------------------------------------------------------- #
def process_series(series_dir: str):
    rel_path = os.path.relpath(series_dir, in_root)          # e.g. 123/1.../3...
    out_path = os.path.join(out_root, rel_path) + ".npy"     # keep tree, add .npy
    if os.path.exists(out_path) and args.skip_existing:
        #print(f"Skipping existing {out_path}", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        vol = load_dicom_series(series_dir, min_slices=args.min_slices)                  # [D, H, W]
        np.save(out_path, vol.astype(np.float32))
    except Exception as e:
        error_msg = traceback.format_exc()
        if "Too few" not in error_msg:
            print(f"Failed on {series_dir}\n{error_msg}", file=sys.stderr)

# ------------------------------------------------------------------------- #
# Enumerate *leaf* dirs that contain DICOM files and queue them
# ------------------------------------------------------------------------- #
leaf_series_dirs = []
for root, dirs, files in os.walk(in_root):
    # treat directory as a series iff it contains *.dcm files
    if any(f.lower().endswith(".dcm") for f in files):
        leaf_series_dirs.append(root)

print(f"Found {len(leaf_series_dirs)} series.")

# ------------------------------------------------------------------------- #
# Multiprocessing
# ------------------------------------------------------------------------- #
with Pool(processes=args.workers) as pool, \
     tqdm(total=len(leaf_series_dirs), desc="Processing") as pbar:
    for _ in pool.imap_unordered(process_series, leaf_series_dirs):
        pbar.update(1)
