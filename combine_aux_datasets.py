#!/usr/bin/env python3
"""
Combine GLI and MET auxiliary datasets into a single GLI_MET auxiliary dataset.
This script loads the GLI and MET aux files and combines them with proper ID renumbering.
"""

import json
import argparse
from pathlib import Path

def combine_aux_datasets(gli_aux_file, met_aux_file, output_file):
    """
    Combine GLI and MET auxiliary datasets into a single dataset.
    
    Args:
        gli_aux_file: Path to GLI auxiliary dataset JSON file
        met_aux_file: Path to MET auxiliary dataset JSON file  
        output_file: Path to output combined dataset JSON file
    """
    
    print(f"Loading GLI auxiliary data from: {gli_aux_file}")
    with open(gli_aux_file, 'r') as f:
        gli_data = json.load(f)
    
    print(f"Loading MET auxiliary data from: {met_aux_file}")
    with open(met_aux_file, 'r') as f:
        met_data = json.load(f)
    
    print(f"GLI data: {len(gli_data)} volumes")
    print(f"MET data: {len(met_data)} volumes")
    
    # Combine the datasets
    combined_data = []
    
    # Add GLI data (keep original IDs)
    combined_data.extend(gli_data)
    
    # Add MET data with renumbered IDs
    gli_max_id = max(entry['id'] for entry in gli_data) if gli_data else -1
    
    for entry in met_data:
        # Create a copy and update the ID
        combined_entry = entry.copy()
        combined_entry['id'] = gli_max_id + 1 + entry['id']
        combined_data.append(combined_entry)
    
    # Sort by ID to maintain order
    combined_data.sort(key=lambda x: x['id'])
    
    print(f"Combined data: {len(combined_data)} volumes")
    print(f"ID range: {min(entry['id'] for entry in combined_data)} to {max(entry['id'] for entry in combined_data)}")
    
    # Save combined dataset
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined auxiliary dataset saved to: {output_file}")
    
    # Show some statistics
    gli_count = len(gli_data)
    met_count = len(met_data)
    total_count = len(combined_data)
    
    print(f"\nDataset statistics:")
    print(f"  GLI volumes: {gli_count} ({gli_count/total_count*100:.1f}%)")
    print(f"  MET volumes: {met_count} ({met_count/total_count*100:.1f}%)")
    print(f"  Total volumes: {total_count}")
    
    # Show sample entries
    print(f"\nSample entries:")
    print(f"First GLI entry (ID {combined_data[0]['id']}): {Path(combined_data[0]['seg_file']).parent.name}")
    if gli_count < total_count:
        first_met_idx = gli_count
        print(f"First MET entry (ID {combined_data[first_met_idx]['id']}): {Path(combined_data[first_met_idx]['seg_file']).parent.name}")
    print(f"Last entry (ID {combined_data[-1]['id']}): {Path(combined_data[-1]['seg_file']).parent.name}")

def main():
    parser = argparse.ArgumentParser(description='Combine GLI and MET auxiliary datasets')
    parser.add_argument('--gli_aux_file', 
                       default='brats_gli_3d_vqa_subjTrue_train_aux_updated_v11_seed0.json',
                       help='Path to GLI auxiliary dataset file')
    parser.add_argument('--met_aux_file',
                       default='brats_met_3d_vqa_subjTrue_train_aux_v11_seed0.json', 
                       help='Path to MET auxiliary dataset file')
    parser.add_argument('--output_file',
                       default='brats_gli_met_3d_vqa_subjTrue_train_aux_combined_v11_seed0.json',
                       help='Output path for combined auxiliary dataset')
    parser.add_argument('--version',
                       default='v11',
                       help='Version string for output file')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed number for file naming')
    
    args = parser.parse_args()
    
    print(f"Combining auxiliary datasets:")
    print(f"  GLI file: {args.gli_aux_file}")
    print(f"  MET file: {args.met_aux_file}")
    print(f"  Output file: {args.output_file}")
    
    # Check if input files exist
    if not Path(args.gli_aux_file).exists():
        print(f"Error: GLI file not found: {args.gli_aux_file}")
        return
        
    if not Path(args.met_aux_file).exists():
        print(f"Error: MET file not found: {args.met_aux_file}")
        return
    
    combine_aux_datasets(args.gli_aux_file, args.met_aux_file, args.output_file)

if __name__ == "__main__":
    main()