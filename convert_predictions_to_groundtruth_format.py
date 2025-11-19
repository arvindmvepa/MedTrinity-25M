#!/usr/bin/env python3
"""
Convert prediction files with logits to ground truth format.
Converts logits to labels using argmax for area/shape/satellite and threshold>0 for regions.
"""

import json
import numpy as np
from pathlib import Path
import argparse

def get_label_order(dataset_type='gli'):
    """Define the order of labels as they appear in the prediction file"""
    if dataset_type in ['met', 'goat']:
        return [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue"
        ]
    else:  # GLI dataset
        return [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue",
            "Resection Cavity"
        ]

def convert_logits_to_labels(predictions, dataset_type='gli'):
    """
    Convert a list of prediction dictionaries with logits to ground truth format.
    
    For GLI: Each volume has 16 questions (4 labels x 4 question types)
    For MET/GOAT: Each volume has 12 questions (3 labels x 4 question types)
    
    Questions are in order:
    - N area questions (one per label type)
    - N region questions (one per label type)
    - N shape questions (one per label type)
    - N satellite questions (one per label type)
    """
    label_order = get_label_order(dataset_type)
    num_labels = len(label_order)
    questions_per_volume = num_labels * 4  # 4 question types per label
    
    # Group predictions by volume
    num_volumes = len(predictions) // questions_per_volume
    converted_data = []
    
    print(f"Dataset: {dataset_type}")
    print(f"Number of labels: {num_labels}")
    print(f"Questions per volume: {questions_per_volume}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Number of volumes: {num_volumes}")
    
    for vol_idx in range(num_volumes):
        start_idx = vol_idx * questions_per_volume
        vol_predictions = predictions[start_idx:start_idx + questions_per_volume]
        
        # Extract seg_file from first prediction in the volume
        seg_file = vol_predictions[0].get('seg_file', '')
        
        # Initialize the volume entry
        volume_entry = {
            "id": vol_idx,
            "seg_file": seg_file,
            "labels": {}
        }
        
        # Process each label type
        for label_idx, label_name in enumerate(label_order):
            volume_entry["labels"][label_name] = {}
            
            # Area question (questions 0 to num_labels-1)
            area_pred = vol_predictions[label_idx]
            area_logits = area_pred.get('area_logits', [])
            if area_logits:
                area_label = int(np.argmax(area_logits))
                volume_entry["labels"][label_name]["area"] = area_label
            
            # Region question (questions num_labels to 2*num_labels-1)
            region_pred = vol_predictions[num_labels + label_idx]
            region_logits = region_pred.get('region_logits', [])
            if region_logits:
                # Get all indices where logits > 0
                region_labels = [i for i, logit in enumerate(region_logits) if logit > 0]
                volume_entry["labels"][label_name]["region"] = region_labels
            
            # Shape question (questions 2*num_labels to 3*num_labels-1)
            shape_pred = vol_predictions[2 * num_labels + label_idx]
            shape_logits = shape_pred.get('shape_logits', [])
            if shape_logits:
                shape_label = int(np.argmax(shape_logits))
                volume_entry["labels"][label_name]["shape"] = shape_label
            
            # Satellite question (questions 3*num_labels to 4*num_labels-1)
            satellite_pred = vol_predictions[3 * num_labels + label_idx]
            satellite_logits = satellite_pred.get('satellite_logits', [])
            if satellite_logits:
                satellite_label = int(np.argmax(satellite_logits))
                volume_entry["labels"][label_name]["satellite"] = satellite_label
        
        converted_data.append(volume_entry)
    
    return converted_data

def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description='Convert prediction logits to ground truth format')
    parser.add_argument('dataset_type', choices=['gli', 'met', 'goat'], 
                       help='Dataset type: gli, met, or goat')
    parser.add_argument('input_file', help='Input JSON file with predictions and logits')
    parser.add_argument('output_file', help='Output JSON file in ground truth format')
    args = parser.parse_args()
    
    dataset_type = args.dataset_type
    
    print(f"Loading predictions from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        predictions = json.load(f)
    
    num_labels = len(get_label_order(dataset_type))
    questions_per_volume = num_labels * 4
    expected_volumes = len(predictions) // questions_per_volume
    
    print(f"Converting {len(predictions)} predictions for {dataset_type} dataset...")
    print(f"Expected volumes: {expected_volumes}")
    
    converted_data = convert_logits_to_labels(predictions, dataset_type)
    
    print(f"Saving converted data to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"✓ Successfully converted {len(converted_data)} volumes")
    
    # Print sample output
    if converted_data:
        print(f"\nSample output (first volume):")
        print(json.dumps(converted_data[0], indent=2))

if __name__ == "__main__":
    main()
