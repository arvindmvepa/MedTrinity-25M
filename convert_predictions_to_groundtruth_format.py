#!/usr/bin/env python3
"""
Convert prediction files with logits to ground truth format.
Converts logits to labels using argmax for area/shape/satellite and threshold>0 for regions.
"""

import json
import numpy as np
from pathlib import Path

def get_label_order():
    """Define the order of labels as they appear in the prediction file"""
    return [
        "Non-Enhancing Tumor",
        "Surrounding Non-enhancing FLAIR hyperintensity",
        "Enhancing Tissue",
        "Resection Cavity"
    ]

def convert_logits_to_labels(predictions):
    """
    Convert a list of prediction dictionaries with logits to ground truth format.
    
    Each volume has 16 questions in order:
    - 4 area questions (one per label type)
    - 4 region questions (one per label type)
    - 4 shape questions (one per label type)
    - 4 satellite questions (one per label type)
    """
    label_order = get_label_order()
    
    # Group predictions by volume (every 16 questions = 1 volume)
    num_volumes = len(predictions) // 16
    converted_data = []
    
    for vol_idx in range(num_volumes):
        start_idx = vol_idx * 16
        vol_predictions = predictions[start_idx:start_idx + 16]
        
        # Extract seg_file from first prediction in the volume
        seg_file = vol_predictions[0].get('seg_file', '')
        
        # Initialize the volume entry
        volume_entry = {
            "id": vol_idx,
            "seg_file": seg_file,
            "labels": {}
        }
        
        # Process each label type (4 labels x 4 questions = 16)
        for label_idx, label_name in enumerate(label_order):
            volume_entry["labels"][label_name] = {}
            
            # Area question (questions 0-3)
            area_pred = vol_predictions[label_idx]
            area_logits = area_pred.get('area_logits', [])
            if area_logits:
                area_label = int(np.argmax(area_logits)) + 1  # Convert to 1-indexed
                volume_entry["labels"][label_name]["area"] = area_label
            
            # Region question (questions 4-7)
            region_pred = vol_predictions[4 + label_idx]
            region_logits = region_pred.get('region_logits', [])
            if region_logits:
                # Get all indices where logits > 0, convert to 1-indexed
                region_labels = [i + 1 for i, logit in enumerate(region_logits) if logit > 0]
                volume_entry["labels"][label_name]["region"] = region_labels
            
            # Shape question (questions 8-11)
            shape_pred = vol_predictions[8 + label_idx]
            shape_logits = shape_pred.get('shape_logits', [])
            if shape_logits:
                shape_label = int(np.argmax(shape_logits)) + 1  # Convert to 1-indexed
                volume_entry["labels"][label_name]["shape"] = shape_label
            
            # Satellite question (questions 12-15)
            satellite_pred = vol_predictions[12 + label_idx]
            satellite_logits = satellite_pred.get('satellite_logits', [])
            if satellite_logits:
                satellite_label = int(np.argmax(satellite_logits)) + 1  # Convert to 1-indexed
                volume_entry["labels"][label_name]["satellite"] = satellite_label
        
        converted_data.append(volume_entry)
    
    return converted_data

def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert prediction logits to ground truth format')
    parser.add_argument('input_file', help='Input JSON file with predictions and logits')
    parser.add_argument('output_file', help='Output JSON file in ground truth format')
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Converting {len(predictions)} predictions ({len(predictions)//16} volumes)...")
    converted_data = convert_logits_to_labels(predictions)
    
    print(f"Saving converted data to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)
    
    print(f"✓ Successfully converted {len(converted_data)} volumes")
    
    # Print sample output
    if converted_data:
        print("\nSample output (first volume):")
        print(json.dumps(converted_data[0], indent=2))

if __name__ == "__main__":
    main()
