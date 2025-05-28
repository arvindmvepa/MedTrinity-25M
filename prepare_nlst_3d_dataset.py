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
# MONAI transform:  [C, D, H, W]  →  [C, 32, 256, 256]
# ------------------------------------------------------------------------- #
xform = mtf.Compose([
    mtf.CropForeground(),                        # tight-crop non-zero region
    mtf.Resize(spatial_size=[32, 256, 256],
               mode="bilinear",
               align_corners=True)
])

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

    # sort slices – prefer ImagePositionPatient (z) then InstanceNumber
    slices = []
    sort_keys = []
    for fp in dcm_files:
        ds = pydicom.dcmread(fp, force=True)
        slices.append(ds.pixel_array.astype(np.float32))
        if "ImagePositionPatient" in ds:
            z_pos = float(ds.ImagePositionPatient[2])
        elif "InstanceNumber" in ds:
            z_pos = float(ds.InstanceNumber)
        else:                                  # fallback: filename order
            z_pos = 0.0
        sort_keys.append(z_pos)

    order = np.argsort(sort_keys)  # indices that sort by z-pos
    volume = np.stack([slices[i] for i in order], axis=0)
    return volume


# ------------------------------------------------------------------------- #
# Worker
# ------------------------------------------------------------------------- #
def process_series(series_dir: str):
    rel_path = os.path.relpath(series_dir, in_root)          # e.g. 123/1.../3...
    out_path = os.path.join(out_root, rel_path) + ".npy"     # keep tree, add .npy
    if os.path.exists(out_path) and args.skip_existing:
        print(f"Skipping existing {out_path}", file=sys.stderr)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        vol = load_dicom_series(series_dir, min_slices=args.min_slices)                  # [D, H, W]
        vol = vol[np.newaxis, ...]                           # [C=1, D, H, W]

        # min-max normalise (CTs occasionally have unusual bit-depths)
        vol -= vol.min()
        vol /= max(vol.max(), 1e-8)

        vol = xform(vol)                                     # [1, 32, 256, 256]
        np.save(out_path, vol.astype(np.float32))
    except Exception as e:
        print(f"Failed on {series_dir}\n{traceback.format_exc()}", file=sys.stderr)

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
