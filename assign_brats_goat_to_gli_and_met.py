#!/usr/bin/env python
import glob
import numpy as np
import nibabel as nib
import json
import os
import argparse
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

def insert_volume_dir_into_mapping(mapping_dict, ref_path, target_path):
    ref_dir = os.path.dirname(ref_path)
    target_dir = os.path.dirname(target_path)
    mapping_dict[ref_dir] = target_dir

def process_single_goat_volume(goat_path, target_files, rtol, atol):
    """
    Process a single GoAT volume and find its match in target files
    Returns: (goat_path, matched_target_path) or (goat_path, None)
    """
    try:
        # Load GoAT image
        goat_img = nib.load(goat_path)
        goat_data = goat_img.get_fdata()
        goat_shape = goat_img.shape

        for target_path in target_files:
            try:
                target_img = nib.load(target_path)

                if target_img.shape != goat_shape:
                    continue

                target_data = target_img.get_fdata()

                # Check equality with tolerance
                if np.allclose(goat_data, target_data, rtol=rtol, atol=atol):
                    return (goat_path, target_path)
                    
            except Exception as e:
                print(f"Error processing target {target_path}: {e}")
                continue

        # No match found
        return (goat_path, None)
        
    except Exception as e:
        print(f"Error processing GoAT volume {goat_path}: {e}")
        return (goat_path, None)

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
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of jobs to use (default: -1 for all CPU cores)')
    parser.add_argument('--remove_matched', action='store_true',
                       help='Remove matched target files to speed up subsequent comparisons')
    
    args = parser.parse_args()
    
    # Determine number of processes
    if args.n_jobs == -1:
        print(f"Using all available CPU cores for parallel processing")
    else:
        print(f"Using {args.n_jobs} processes for parallel processing")
    
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
    unmatched_goat_volumes = []
    
    if args.remove_matched:
        print("Processing sequentially to remove matched files...")
        
        # Process sequentially when removing matched files
        remaining_gli_files = brats_gli_files.copy()
        remaining_met_files = brats_met_files.copy()
        matched_count = 0
        
        for goat_path in tqdm(brats_goat_files, desc="Processing GoAT volumes"):
            all_target_files = remaining_gli_files + remaining_met_files
            
            goat_path, matched_target_path = process_single_goat_volume(
                goat_path, all_target_files, args.rtol, args.atol
            )
            
            if matched_target_path is not None:
                insert_volume_dir_into_mapping(mapping_results, goat_path, matched_target_path)
                matched_count += 1
                
                # Remove the matched file from the appropriate list
                if "GLI" in matched_target_path:
                    remaining_gli_files.remove(matched_target_path)
                elif "MET" in matched_target_path:
                    remaining_met_files.remove(matched_target_path)
            else:
                unmatched_goat_volumes.append(goat_path)
        
        remaining_gli_count = len(remaining_gli_files)
        remaining_met_count = len(remaining_met_files)
        
    else:
        print("Processing in parallel (no file removal)...")
        
        # Combine all target files for parallel processing
        all_target_files = brats_gli_files + brats_met_files
        print(f"Total target files to compare against: {len(all_target_files)}")
        
        # Use joblib with tqdm_joblib for progress bar
        with tqdm_joblib(desc="Processing GoAT volumes", total=len(brats_goat_files)):
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_single_goat_volume)(
                    goat_path, 
                    all_target_files, 
                    args.rtol, 
                    args.atol
                )
                for goat_path in brats_goat_files
            )
        
        # Process results
        matched_count = 0
        matched_target_files = []
        
        for goat_path, matched_target_path in results:
            if matched_target_path is not None:
                insert_volume_dir_into_mapping(mapping_results, goat_path, matched_target_path)
                matched_count += 1
                matched_target_files.append(matched_target_path)
            else:
                unmatched_goat_volumes.append(goat_path)
        
        # Calculate remaining files
        matched_gli = sum(1 for path in matched_target_files if "GLI" in path)
        matched_met = sum(1 for path in matched_target_files if "MET" in path)
        remaining_gli_count = len(brats_gli_files) - matched_gli
        remaining_met_count = len(brats_met_files) - matched_met
    
    # Save results
    with open(args.save_file, "w") as f:
        json.dump(mapping_results, f, indent=2)
    
    print(f"\nMatching complete!")
    print(f"Total GoAT volumes processed: {len(brats_goat_files)}")
    print(f"Successful matches found: {matched_count}")
    print(f"Unmatched GoAT volumes: {len(unmatched_goat_volumes)}")
    print(f"Mapping results saved to: {args.save_file}")
    
    # Show some unmatched volumes if any
    if unmatched_goat_volumes:
        print(f"\nFirst 5 unmatched GoAT volumes:")
        for vol in unmatched_goat_volumes[:5]:
            print(f"  {os.path.basename(vol)}")
    
    print(f"\nTarget file breakdown:")
    print(f"  Original GLI files: {len(brats_gli_files)}")
    print(f"  Original MET files: {len(brats_met_files)}")
    if args.remove_matched:
        print(f"  GLI files matched: {len(brats_gli_files) - remaining_gli_count}")
        print(f"  MET files matched: {len(brats_met_files) - remaining_met_count}")
    else:
        matched_gli = sum(1 for path in matched_target_files if "GLI" in path)
        matched_met = sum(1 for path in matched_target_files if "MET" in path)
        print(f"  GLI files matched: {matched_gli}")
        print(f"  MET files matched: {matched_met}")
    print(f"  Remaining GLI files: {remaining_gli_count}")
    print(f"  Remaining MET files: {remaining_met_count}")

if __name__ == "__main__":
    main()