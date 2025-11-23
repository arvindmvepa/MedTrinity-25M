#!/usr/bin/env python3
"""
Combine GLI and MET auxiliary datasets into a single GLI_MET auxiliary dataset.
This script loads the GLI and MET aux files and combines them with proper ID renumbering.
"""

import json
import argparse
from pathlib import Path

def combine_datasets(gli_file, met_file, output_file, filter_resection_cavity=True):
    """
    Combine GLI and MET auxiliary datasets into a single dataset.
    
    Args:
        gli_file: Path to GLI dataset JSON file
        met_file: Path to MET dataset JSON file  
        output_file: Path to output combined dataset JSON file
        filter_resection_cavity: If True, exclude GLI questions with 'Resection Cavity' label_name
    """
    
    print(f"Loading GLI data from: {gli_file}")
    with open(gli_file, 'r') as f:
        gli_data = json.load(f)
    
    print(f"Loading MET data from: {met_file}")
    with open(met_file, 'r') as f:
        met_data = json.load(f)
    
    print(f"GLI data: {len(gli_data)} volumes")
    print(f"MET data: {len(met_data)} volumes")
    
    # Filter out Resection Cavity questions from GLI data if requested
    if filter_resection_cavity:
        original_gli_count = len(gli_data)
        gli_data = [entry for entry in gli_data if entry.get('label_name') != 'Resection Cavity']
        filtered_count = original_gli_count - len(gli_data)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} Resection Cavity questions from GLI data")
            print(f"GLI data after filtering: {len(gli_data)} volumes")
    
    # Combine the datasets
    combined_data = []
    
    # Add GLI data (keep original IDs)
    combined_data.extend(gli_data)
    
    # Add MET data with renumbered IDs
    id_string = None
    if 'id' in gli_data[0]:
        id_string = 'id'
    elif 'qid' in gli_data[0]:
        id_string = 'qid'
    else:
        raise ValueError("No 'id' or 'qid' field found in GLI dataset entries.")
    gli_max_id = max(entry[id_string] for entry in gli_data) if gli_data else -1
    
    for entry in met_data:
        # Create a copy and update the ID
        combined_entry = entry.copy()
        combined_entry[id_string] = gli_max_id + 1 + entry[id_string]
        combined_data.append(combined_entry)
    
    # Sort by ID to maintain order
    combined_data.sort(key=lambda x: x[id_string])
    
    print(f"Combined data: {len(combined_data)} volumes")
    print(f"ID range: {min(entry[id_string] for entry in combined_data)} to {max(entry[id_string] for entry in combined_data)}")
    
    # Save combined dataset
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined dataset saved to: {output_file}")
    
    # Show some statistics
    gli_count = len(gli_data)
    met_count = len(met_data)
    total_count = len(combined_data)
    
    print(f"\nDataset statistics:")
    print(f"  GLI questions: {gli_count} ({gli_count/total_count*100:.1f}%)")
    print(f"  MET questions: {met_count} ({met_count/total_count*100:.1f}%)")
    print(f"  Total volumes: {total_count}")
    
    # Show sample entries
    print(f"\nSample entries:")
    print(f"First GLI entry (ID {combined_data[0][id_string]}): {Path(combined_data[0].get('seg_file', combined_data[0].get('volume_seg_file')))}")
    if gli_count < total_count:
        first_met_idx = gli_count
        print(f"First MET entry (ID {combined_data[first_met_idx][id_string]}): {Path(combined_data[first_met_idx].get('seg_file', combined_data[first_met_idx].get('volume_seg_file')))}")
    print(f"Last entry (ID {combined_data[-1][id_string]}): {Path(combined_data[-1].get('seg_file', combined_data[-1].get('volume_seg_file')))}")
def main():
    parser = argparse.ArgumentParser(description='Combine GLI and MET datasets')
    parser.add_argument('--gli_file', 
                       default='brats_gli_3d_vqa_subjTrue_train_aux_updated_v11_seed0.json',
                       help='Path to GLI dataset file')
    parser.add_argument('--met_file',
                       default='brats_met_3d_vqa_subjTrue_train_aux_updated_v11_seed0.json', 
                       help='Path to MET dataset file')
    parser.add_argument('--output_file',
                       default='brats_gli_met_3d_vqa_subjTrue_train_aux_combined_v11_seed0.json',
                       help='Output path for combined dataset')
    parser.add_argument('--version',
                       default='v11',
                       help='Version string for output file')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed number for file naming')
    parser.add_argument('--no_filter_resection_cavity', action='store_true', default=False,
                       help='Keep Resection Cavity questions from GLI data (by default they are filtered out)')
    
    args = parser.parse_args()
    
    print(f"Combining datasets:")
    print(f"  GLI file: {args.gli_file}")
    print(f"  MET file: {args.met_file}")
    print(f"  Output file: {args.output_file}")
    print(f"  Filter Resection Cavity: {not args.no_filter_resection_cavity}")
    
    # Check if input files exist
    if not Path(args.gli_file).exists():
        print(f"Error: GLI file not found: {args.gli_file}")
        return
        
    if not Path(args.met_file).exists():
        print(f"Error: MET file not found: {args.met_file}")
        return
    
    combine_datasets(args.gli_file, args.met_file, args.output_file, not args.no_filter_resection_cavity)

if __name__ == "__main__":
    main()