#!/usr/bin/env python3
"""
Analyze distributions of values in answer_vqa_numeric from generated VQA files.
This script examines the five entries in answer_vqa_numeric:
1. Area (scalar)
2. Region (list of indices) 
3. Shape (scalar)
4. Satellite (scalar)
5. Unknown (scalar, if present)
"""

import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import argparse


def load_vqa_files(file_paths):
    """Load VQA data from multiple JSON files"""
    all_data = []
    for file_path in file_paths:
        print(f"Loading: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"  Loaded {len(data)} entries")
            all_data.extend(data)
    print(f"Total entries loaded: {len(all_data)}")
    return all_data


def analyze_numeric_distributions(vqa_data):
    """Analyze distributions of answer_vqa_numeric values"""
    
    # Initialize collectors
    area_values = []
    region_values = []  # Will collect all individual region indices
    shape_values = []
    satellite_values = []
    unknown_values = []
    
    # Collector for region presence per entry
    region_presence = defaultdict(int)  # Count how many entries contain each region index
    
    total_entries = len(vqa_data)
    entries_with_unknown = 0
    
    print("Processing VQA entries...")
    
    for i, entry in enumerate(vqa_data):
        if 'answer_vqa_numeric' not in entry:
            print(f"Warning: Entry {i} missing 'answer_vqa_numeric' key")
            continue
            
        numeric = entry['answer_vqa_numeric']
        
        if len(numeric) < 4:
            print(f"Warning: Entry {i} has insufficient numeric values: {numeric}")
            continue
        
        # Extract values
        area_values.append(numeric[0])
        
        # Handle region list (index 1)
        regions = numeric[1] if isinstance(numeric[1], list) else [numeric[1]]
        region_values.extend(regions)
        
        # Count presence of each region index in this entry
        for region_idx in set(regions):  # Use set to avoid double-counting duplicates in same entry
            region_presence[region_idx] += 1
        
        shape_values.append(numeric[2])
        satellite_values.append(numeric[3])
        
        # Handle unknown (index 4, if present)
        if len(numeric) > 4:
            unknown_values.append(numeric[4])
            entries_with_unknown += 1
    
    print(f"Processed {len(area_values)} valid entries")
    if entries_with_unknown > 0:
        print(f"Found {entries_with_unknown} entries with unknown values")
    
    # Create distributions
    results = {}
    
    # Area distribution
    area_dist = pd.Series(area_values).value_counts().sort_index()
    results['area'] = {
        'distribution': area_dist,
        'total_count': len(area_values),
        'unique_values': len(area_dist)
    }
    
    # Region distribution (individual indices)
    region_dist = pd.Series(region_values).value_counts().sort_index()
    results['region_indices'] = {
        'distribution': region_dist,
        'total_count': len(region_values),
        'unique_values': len(region_dist)
    }
    
    # Region presence (proportion of entries containing each region)
    region_presence_series = pd.Series(region_presence).sort_index()
    region_proportions = (region_presence_series / total_entries * 100).round(2)
    results['region_presence'] = {
        'counts': region_presence_series,
        'proportions': region_proportions,
        'total_entries': total_entries
    }
    
    # Shape distribution
    shape_dist = pd.Series(shape_values).value_counts().sort_index()
    results['shape'] = {
        'distribution': shape_dist,
        'total_count': len(shape_values),
        'unique_values': len(shape_dist)
    }
    
    # Satellite distribution
    satellite_dist = pd.Series(satellite_values).value_counts().sort_index()
    results['satellite'] = {
        'distribution': satellite_dist,
        'total_count': len(satellite_values),
        'unique_values': len(satellite_dist)
    }
    
    # Unknown distribution (if present)
    if unknown_values:
        unknown_dist = pd.Series(unknown_values).value_counts().sort_index()
        results['unknown'] = {
            'distribution': unknown_dist,
            'total_count': len(unknown_values),
            'unique_values': len(unknown_dist)
        }
    
    return results


def print_distribution_summary(results):
    """Print summary of all distributions"""
    
    print("\n" + "="*80)
    print("VQA NUMERIC ANSWER DISTRIBUTIONS")
    print("="*80)
    
    # Area distribution
    print(f"\n1. AREA DISTRIBUTION:")
    print("-" * 40)
    area_dist = results['area']['distribution']
    total_area = results['area']['total_count']
    for value, count in area_dist.items():
        percentage = (count / total_area * 100)
        print(f"Value {value:2d}: {count:6d} ({percentage:5.1f}%)")
    print(f"Total: {total_area:6d} (100.0%)")
    
    # Region indices distribution
    print(f"\n2. REGION INDICES DISTRIBUTION (all occurrences):")
    print("-" * 40)
    region_dist = results['region_indices']['distribution']
    total_region = results['region_indices']['total_count']
    for value, count in region_dist.items():
        percentage = (count / total_region * 100)
        print(f"Region {value:2d}: {count:6d} ({percentage:5.1f}%)")
    print(f"Total: {total_region:6d} (100.0%)")
    
    # Region presence (proportion of entries)
    print(f"\n3. REGION PRESENCE (proportion of entries containing each region):")
    print("-" * 40)
    region_presence = results['region_presence']
    total_entries = region_presence['total_entries']
    proportions = region_presence['proportions']
    counts = region_presence['counts']
    
    for region_idx, proportion in proportions.items():
        count = counts[region_idx]
        print(f"Region {region_idx:2d}: {count:6d} entries ({proportion:5.1f}%)")
    print(f"Total entries: {total_entries}")
    print(f"Note: Sum may exceed 100% as entries can contain multiple regions")
    
    # Shape distribution
    print(f"\n4. SHAPE DISTRIBUTION:")
    print("-" * 40)
    shape_dist = results['shape']['distribution']
    total_shape = results['shape']['total_count']
    for value, count in shape_dist.items():
        percentage = (count / total_shape * 100)
        print(f"Value {value:2d}: {count:6d} ({percentage:5.1f}%)")
    print(f"Total: {total_shape:6d} (100.0%)")
    
    # Satellite distribution
    print(f"\n5. SATELLITE DISTRIBUTION:")
    print("-" * 40)
    satellite_dist = results['satellite']['distribution']
    total_satellite = results['satellite']['total_count']
    for value, count in satellite_dist.items():
        percentage = (count / total_satellite * 100)
        print(f"Value {value:2d}: {count:6d} ({percentage:5.1f}%)")
    print(f"Total: {total_satellite:6d} (100.0%)")
    
    # Unknown distribution (if present)
    if 'unknown' in results:
        print(f"\n6. UNKNOWN DISTRIBUTION:")
        print("-" * 40)
        unknown_dist = results['unknown']['distribution']
        total_unknown = results['unknown']['total_count']
        for value, count in unknown_dist.items():
            percentage = (count / total_unknown * 100)
            print(f"Value {value:2d}: {count:6d} ({percentage:5.1f}%)")
        print(f"Total: {total_unknown:6d} (100.0%)")


def save_distributions_to_csv(results, output_dir):
    """Save distributions to CSV files"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving distributions to: {output_dir}")
    
    # Area distribution
    area_df = pd.DataFrame({
        'value': results['area']['distribution'].index,
        'count': results['area']['distribution'].values,
        'percentage': (results['area']['distribution'] / results['area']['total_count'] * 100).round(2)
    })
    area_df.to_csv(output_dir / 'area_distribution.csv', index=False)
    print(f"  Saved: area_distribution.csv")
    
    # Region indices distribution
    region_indices_df = pd.DataFrame({
        'region_index': results['region_indices']['distribution'].index,
        'count': results['region_indices']['distribution'].values,
        'percentage': (results['region_indices']['distribution'] / results['region_indices']['total_count'] * 100).round(2)
    })
    region_indices_df.to_csv(output_dir / 'region_indices_distribution.csv', index=False)
    print(f"  Saved: region_indices_distribution.csv")
    
    # Region presence
    region_presence_df = pd.DataFrame({
        'region_index': results['region_presence']['counts'].index,
        'entries_containing': results['region_presence']['counts'].values,
        'percentage_of_entries': results['region_presence']['proportions'].values
    })
    region_presence_df.to_csv(output_dir / 'region_presence_distribution.csv', index=False)
    print(f"  Saved: region_presence_distribution.csv")
    
    # Shape distribution
    shape_df = pd.DataFrame({
        'value': results['shape']['distribution'].index,
        'count': results['shape']['distribution'].values,
        'percentage': (results['shape']['distribution'] / results['shape']['total_count'] * 100).round(2)
    })
    shape_df.to_csv(output_dir / 'shape_distribution.csv', index=False)
    print(f"  Saved: shape_distribution.csv")
    
    # Satellite distribution
    satellite_df = pd.DataFrame({
        'value': results['satellite']['distribution'].index,
        'count': results['satellite']['distribution'].values,
        'percentage': (results['satellite']['distribution'] / results['satellite']['total_count'] * 100).round(2)
    })
    satellite_df.to_csv(output_dir / 'satellite_distribution.csv', index=False)
    print(f"  Saved: satellite_distribution.csv")
    
    # Unknown distribution (if present)
    if 'unknown' in results:
        unknown_df = pd.DataFrame({
            'value': results['unknown']['distribution'].index,
            'count': results['unknown']['distribution'].values,
            'percentage': (results['unknown']['distribution'] / results['unknown']['total_count'] * 100).round(2)
        })
        unknown_df.to_csv(output_dir / 'unknown_distribution.csv', index=False)
        print(f"  Saved: unknown_distribution.csv")
    
    # Summary statistics
    summary_data = []
    for task, data in results.items():
        if task == 'region_presence':
            continue  # Skip region presence for summary
        summary_data.append({
            'task': task,
            'total_count': data['total_count'],
            'unique_values': data['unique_values'],
            'min_value': data['distribution'].index.min(),
            'max_value': data['distribution'].index.max(),
            'most_common_value': data['distribution'].index[0],
            'most_common_count': data['distribution'].iloc[0]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'distribution_summary.csv', index=False)
    print(f"  Saved: distribution_summary.csv")


def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description='Analyze VQA numeric answer distributions')
    parser.add_argument('files', nargs='+', help='VQA JSON files to analyze')
    parser.add_argument('--output', '-o', default='vqa_distributions', 
                       help='Output directory for CSV files (default: vqa_distributions)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        vqa_data = load_vqa_files(args.files)
        
        # Analyze distributions
        results = analyze_numeric_distributions(vqa_data)
        
        # Print summary
        print_distribution_summary(results)
        
        # Save to CSV files
        save_distributions_to_csv(results, args.output)
        
        print(f"\nAnalysis complete!")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()