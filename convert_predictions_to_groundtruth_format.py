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
    if dataset_type in ['met', 'goat', 'gli_met']:
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

def find_met_start_index(predictions):
    """Find the index where MET predictions start by looking at seg_file paths"""
    for i, pred in enumerate(predictions):
        seg_file = pred.get('seg_file', '')
        if 'MET' in seg_file:
            return i
    return None

def convert_logits_to_labels(predictions, dataset_type='gli'):
    """
    Convert a list of prediction dictionaries with logits to ground truth format.
    
    For GLI: Each volume has 16 questions (4 labels x 4 question types)
    For MET/GOAT: Each volume has 12 questions (3 labels x 4 question types)
    For GLI_MET: Each volume has 24 questions (12 from GLI without Resection Cavity + 12 from MET)
    
    Questions are in order:
    - N area questions (one per label type)
    - N region questions (one per label type)
    - N shape questions (one per label type)
    - N satellite questions (one per label type)
    """
    
    if dataset_type == 'gli_met':
        # Handle GLI_MET case separately
        met_start_idx = find_met_start_index(predictions)
        if met_start_idx is None:
            raise ValueError("No MET predictions found in the data")
        
        # Extract only non-Resection Cavity GLI questions (12 per volume)
        gli_questions_per_volume = 12
        met_questions_per_volume = 12
        num_gli_volumes = met_start_idx // 16  # GLI has 16 questions but we use 12
        num_met_volumes = (len(predictions) - met_start_idx) // met_questions_per_volume
        
        converted_data = []
        label_order = get_label_order(dataset_type)
        
        # Process GLI volumes
        for vol_idx in range(num_gli_volumes):
            gli_start_idx = vol_idx * 16
            vol_predictions = predictions[gli_start_idx:gli_start_idx + gli_questions_per_volume]
            
            seg_file = vol_predictions[0].get('seg_file', '')
            volume_entry = {"id": vol_idx, "seg_file": seg_file, "labels": {}}
            
            # Process first 3 labels only (skip Resection Cavity)
            for label_idx, label_name in enumerate(label_order):
                volume_entry["labels"][label_name] = {}
                
                # Area (0-2), Region (3-5), Shape (6-8), Satellite (9-11)
                if vol_predictions[label_idx].get('area_logits'):
                    area_label = int(np.argmax(vol_predictions[label_idx]['area_logits']))
                    volume_entry["labels"][label_name]["area"] = area_label
                
                if vol_predictions[3 + label_idx].get('region_logits'):
                    region_logits = vol_predictions[3 + label_idx]['region_logits']
                    region_labels = [i for i, logit in enumerate(region_logits) if logit > 0]
                    volume_entry["labels"][label_name]["region"] = region_labels
                
                if vol_predictions[6 + label_idx].get('shape_logits'):
                    shape_label = int(np.argmax(vol_predictions[6 + label_idx]['shape_logits']))
                    volume_entry["labels"][label_name]["shape"] = shape_label
                
                if vol_predictions[9 + label_idx].get('satellite_logits'):
                    satellite_label = int(np.argmax(vol_predictions[9 + label_idx]['satellite_logits']))
                    volume_entry["labels"][label_name]["satellite"] = satellite_label
            
            converted_data.append(volume_entry)
        
        # Process MET volumes
        for vol_idx in range(num_met_volumes):
            met_vol_start_idx = met_start_idx + vol_idx * met_questions_per_volume
            vol_predictions = predictions[met_vol_start_idx:met_vol_start_idx + met_questions_per_volume]
            
            seg_file = vol_predictions[0].get('seg_file', '')
            volume_entry = {"id": num_gli_volumes + vol_idx, "seg_file": seg_file, "labels": {}}
            
            # Process MET labels (3 labels)
            for label_idx, label_name in enumerate(label_order):
                volume_entry["labels"][label_name] = {}
                
                # Area (0-2), Region (3-5), Shape (6-8), Satellite (9-11)
                if vol_predictions[label_idx].get('area_logits'):
                    area_label = int(np.argmax(vol_predictions[label_idx]['area_logits']))
                    volume_entry["labels"][label_name]["area"] = area_label
                
                if vol_predictions[3 + label_idx].get('region_logits'):
                    region_logits = vol_predictions[3 + label_idx]['region_logits']
                    region_labels = [i for i, logit in enumerate(region_logits) if logit > 0]
                    volume_entry["labels"][label_name]["region"] = region_labels
                
                if vol_predictions[6 + label_idx].get('shape_logits'):
                    shape_label = int(np.argmax(vol_predictions[6 + label_idx]['shape_logits']))
                    volume_entry["labels"][label_name]["shape"] = shape_label
                
                if vol_predictions[9 + label_idx].get('satellite_logits'):
                    satellite_label = int(np.argmax(vol_predictions[9 + label_idx]['satellite_logits']))
                    volume_entry["labels"][label_name]["satellite"] = satellite_label
            
            converted_data.append(volume_entry)
        
        print(f"Converted {num_gli_volumes} GLI volumes and {num_met_volumes} MET volumes")
        return converted_data
    
    # Original logic for other datasets
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
    parser.add_argument('dataset_type', choices=['gli', 'met', 'goat', 'gli_met'], 
                       help='Dataset type: gli, met, goat, or gli_met')
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