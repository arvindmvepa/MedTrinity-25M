#!/usr/bin/env python

import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ---- 1. Reference volume (use your provided path, but T1c instead of seg) ----
ref_paths = [
    "/local2/shared_data/BraTS2024-BraTS-GoAT/"
    "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/"
    "BraTS-GoAT-00000/BraTS-GoAT-00000-t1c.nii.gz",
    "/local2/shared_data/BraTS2024-BraTS-GoAT/"
    "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/"
    "BraTS-GoAT-00000/BraTS-GoAT-00001-t1c.nii.gz",
    "/local2/shared_data/BraTS2024-BraTS-GoAT/"
    "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/"
    "BraTS-GoAT-00000/BraTS-GoAT-00002-t1c.nii.gz",
    "/local2/shared_data/BraTS2024-BraTS-GoAT/"
    "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/"
    "BraTS-GoAT-00000/BraTS-GoAT-00003-t1c.nii.gz",
    "/local2/shared_data/BraTS2024-BraTS-GoAT/"
    "MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/"
    "BraTS-GoAT-00000/BraTS-GoAT-00004-t1c.nii.gz",

]
for ref_path in ref_paths:
    print(f"Reference volume: {ref_path}")

    # Load reference image
    ref_img = nib.load(ref_path)
    ref_data = ref_img.get_fdata()
    ref_shape = ref_img.shape

    # ---- 2. Target volumes to compare against ----
    gli_pattern = "/local2/shared_data/BraTS2023_2017_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/*/*t1c.nii.gz"
    gli_files = sorted(glob.glob(gli_pattern))

    if not gli_files:
        print("No GLI volumes found with pattern:", gli_pattern)
        exit(1)

    print(f"Found {len(gli_files)} GLI T1c volumes to compare.")

    # ---- 3. Comparison parameters ----
    # You can tweak these if needed
    RTOL = 1e-5
    ATOL = 1e-5

    matches = []

    # ---- 4. Compare each GLI volume to the reference ----
    for fpath in tqdm(gli_files, desc="Comparing volumes"):
        try:
            img = nib.load(fpath)
        except Exception as e:
            print(f"\n[WARN] Failed to load {fpath}: {e}")
            continue

        if img.shape != ref_shape:
            # Different shape → cannot be identical
            continue

        data = img.get_fdata()

        # Check equality with tolerance
        if np.allclose(ref_data, data, rtol=RTOL, atol=ATOL):
            print(f"\n[MATCH] {fpath} matches the reference volume.")
            matches.append(fpath)

    # ---- 5. Report results ----
    print("\n=== Comparison complete ===")
    if matches:
        print("Matching volumes (within tolerance):")
        for m in matches:
            print("  ", m)
    else:
        print("No GLI volumes match the reference within the given tolerance.")
