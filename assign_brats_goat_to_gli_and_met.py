#!/usr/bin/env python
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json
import os
import argparse


def insert_volume_dir_into_mapping(mapping_dict, ref_path, target_path):
    ref_dir = os.path.dirname(ref_path)
    target_dir = os.path.dirname(target_path)
    mapping_dict[ref_dir] = target_dir

def main():
    parser = argparse.ArgumentParser(description='Assign BraTS GoAT volumes to GLI and MET datasets based on T1c volume matching')
    parser.add_argument('--save_file', 
                       default='brats_goat_to_gli_and_met_mapping.json',
                       help='Output JSON file for volume directory mappings')
    parser.add_argument('--brats_goat_volume_string_path', 
                       default='/local2/shared_data/BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/BraTS-GoAT-*/BraTS-GoAT-*-t1c.nii.gz',
                       help='Glob pattern for BraTS GoAT T1c volumes')
    parser.add_argument('--brats_gli_volume_string_path', 
                       default='/local2/shared_data/BraTS2023_2017_GLI/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/*/*t1c.nii.gz',
                       help='Glob pattern for BraTS GLI T1c volumes')
    parser.add_argument('--brats_met_volume_string_path', 
                       default='/local2/shared_data/BraTS2024-BraTS-MET/MICCAI-BraTS2024-MET-Challenge-Training_overall/*/*t1c.nii.gz',
                       help='Glob pattern for BraTS MET T1c volumes')
    parser.add_argument('--rtol', type=float, default=1e-5,
                       help='Relative tolerance for volume comparison')
    parser.add_argument('--atol', type=float, default=1e-5,
                       help='Absolute tolerance for volume comparison')
    
    args = parser.parse_args()
    
    brats_goat_files = sorted(glob.glob(args.brats_goat_volume_string_path))
    print(f"Found {len(brats_goat_files)} BraTS-GoAT T1c volumes to process.")
    
    brats_gli_files = sorted(glob.glob(args.brats_gli_volume_string_path))
    if not brats_gli_files:
        print("No GLI volumes found with pattern:", args.brats_gli_volume_string_path)
        exit(1)
    print(f"Found {len(brats_gli_files)} GLI T1c volumes to compare.")

    brats_met_files = sorted(glob.glob(args.brats_met_volume_string_path))
    if not brats_met_files:
        print("No MET volumes found with pattern:", args.brats_met_volume_string_path)
        exit(1)
    print(f"Found {len(brats_met_files)} MET T1c volumes to compare.")

    mapping_results = {}
    matched_count = 0
    
    for goat_path in tqdm(brats_goat_files, desc="Processing GoAT volumes"):
        # Load reference image
        goat_img = nib.load(goat_path)
        goat_data = goat_img.get_fdata()
        goat_shape = goat_img.shape

        found_match = False
        for target_path in (brats_gli_files + brats_met_files):
            target_img = nib.load(target_path)

            if target_img.shape != goat_shape:
                continue

            target_data = target_img.get_fdata()

            # Check equality with tolerance
            if np.allclose(goat_data, target_data, rtol=args.rtol, atol=args.atol):
                insert_volume_dir_into_mapping(mapping_results, goat_path, target_path)
                if "GLI" in target_path:
                    brats_gli_files.remove(target_path)
                elif "MET" in target_path:
                    brats_met_files.remove(target_path)
                matched_count += 1
                found_match = True
                break
        
        if not found_match:
            print(f"Warning: No match found for GoAT volume: {os.path.basename(goat_path)}")
    
    # Save results
    with open(args.save_file, "w") as f:
        json.dump(mapping_results, f, indent=2)
    
    print(f"\nMatching complete!")
    print(f"Total GoAT volumes processed: {len(brats_goat_files)}")
    print(f"Successful matches found: {matched_count}")
    print(f"Mapping results saved to: {args.save_file}")
    print(f"Remaining unmatched GLI volumes: {len(brats_gli_files)}")
    print(f"Remaining unmatched MET volumes: {len(brats_met_files)}")

if __name__ == "__main__":
    main()