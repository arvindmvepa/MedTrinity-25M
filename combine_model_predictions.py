#!/usr/bin/env python3
"""
Combine predictions from two models into a single JSON and CSV file.
Model 1 format: List with model_answer field
Model 2 format: List with pred field
"""

import json
import csv
import argparse
import re
from pathlib import Path

def clean_question(question):
    """Clean up question by removing extra characters at beginning and end"""
    if not question:
        return ""
    
    # Remove <s><image> from beginning and 4 from end
    cleaned = question.strip()
    cleaned = re.sub(r'^<s><image>', '', cleaned)
    cleaned = re.sub(r'<\|endoftext\|>$', '', cleaned)
    return cleaned.strip()

def extract_volume_basename(volume_file_dir):
    """Extract basename from volume_file_dir path"""
    if not volume_file_dir:
        return ""
    return Path(volume_file_dir).name

def load_user_study_volumes(user_study_file):
    """Load volume names from user study JSON file"""
    if not user_study_file:
        return None
    
    print(f"Loading user study file: {user_study_file}")
    with open(user_study_file, 'r') as f:
        user_study_data = json.load(f)
    
    # Extract unique volume names from user study
    volumes = set()
    for entry in user_study_data:
        if 'volume' in entry and entry['volume']:
            volumes.add(entry['volume'])
    
    print(f"Found {len(volumes)} unique volumes in user study")
    return volumes

def create_volume_to_predictions_map(predictions, source_name):
    """Create a mapping from volume name to predictions for that volume"""
    volume_map = {}
    
    for pred in predictions:
        # Extract volume basename
        volume = ""
        if 'volume_file_dir' in pred and pred['volume_file_dir']:
            volume = extract_volume_basename(pred['volume_file_dir'])
        elif 'seg_file' in pred and pred['seg_file']:
            volume = extract_volume_basename(str(Path(pred['seg_file']).parent))
        
        if volume:
            if volume not in volume_map:
                volume_map[volume] = []
            volume_map[volume].append(pred)
    
    print(f"{source_name}: Found predictions for {len(volume_map)} volumes")
    return volume_map

def combine_predictions(model1_file, model2_file, output_basename, user_study_file=None):
    """Combine predictions from two models into JSON and CSV files"""
    
    # Load user study volumes if provided
    filter_volumes = load_user_study_volumes(user_study_file) if user_study_file else None
    
    # Load model predictions
    model1_predictions = []
    model2_predictions = []
    
    if model1_file:
        print(f"Loading model 1 predictions from: {model1_file}")
        with open(model1_file, 'r') as f:
            model1_predictions = json.load(f)
        print(f"Model 1 predictions: {len(model1_predictions)}")
    else:
        print("Model 1 file not provided - will use null values")
    
    if model2_file:
        print(f"Loading model 2 predictions from: {model2_file}")
        with open(model2_file, 'r') as f:
            model2_predictions = json.load(f)
        print(f"Model 2 predictions: {len(model2_predictions)}")
    else:
        print("Model 2 file not provided - will use null values")
    
    # Create volume-to-predictions mappings
    model1_map = create_volume_to_predictions_map(model1_predictions, "Model 1") if model1_predictions else {}
    model2_map = create_volume_to_predictions_map(model2_predictions, "Model 2") if model2_predictions else {}
    
    # Get all volumes to process
    if filter_volumes:
        # Use only volumes from user study
        volumes_to_process = filter_volumes
        print(f"Processing {len(volumes_to_process)} volumes from user study")
    else:
        # Use all volumes from both models
        volumes_to_process = set(model1_map.keys()) | set(model2_map.keys())
        print(f"Processing {len(volumes_to_process)} volumes from all model predictions")
    
    # Combine predictions
    combined_data = []
    
    for volume in sorted(volumes_to_process):
        model1_preds = model1_map.get(volume, [])
        model2_preds = model2_map.get(volume, [])
        
        # Get the maximum number of predictions for this volume
        max_preds = max(len(model1_preds), len(model2_preds), 1)
        
        for i in range(max_preds):
            # Get predictions for this index, or None if not available
            model1_pred = model1_preds[i] if i < len(model1_preds) else None
            model2_pred = model2_preds[i] if i < len(model2_preds) else None
            
            # Extract question (prefer model1, then model2)
            question = ""
            if model1_pred and 'orig_question' in model1_pred:
                question = clean_question(model1_pred['orig_question'])
            elif model1_pred and 'question' in model1_pred:
                question = clean_question(model1_pred['question'])
            elif model2_pred and 'orig_question' in model2_pred:
                question = clean_question(model2_pred['orig_question'])
            elif model2_pred and 'question' in model2_pred:
                question = clean_question(model2_pred['question'])
            
            # Extract model answers
            model1_answer = model1_pred.get('model_answer', '') if model1_pred else None
            model2_answer = model2_pred.get('pred', '') if model2_pred else None
            
            # Create combined entry
            combined_entry = {
                'volume': volume,
                'question': question,
                'model_1_answer': model1_answer,
                'model_2_answer': model2_answer
            }
            
            combined_data.append(combined_entry)
    
    # Save JSON file
    json_filename = f"{output_basename}.json"
    print(f"Saving combined JSON to: {json_filename}")
    with open(json_filename, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    # Save CSV file
    csv_filename = f"{output_basename}.csv"
    print(f"Saving combined CSV to: {csv_filename}")
    
    if combined_data:
        fieldnames = ['volume', 'question', 'model_1_answer', 'model_2_answer']
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_data)
    
    print(f"✓ Successfully combined {len(combined_data)} predictions")
    
    # Show statistics
    if filter_volumes:
        volumes_with_model1 = sum(1 for entry in combined_data if entry['model_1_answer'] is not None)
        volumes_with_model2 = sum(1 for entry in combined_data if entry['model_2_answer'] is not None)
        print(f"  - Entries with Model 1 predictions: {volumes_with_model1}")
        print(f"  - Entries with Model 2 predictions: {volumes_with_model2}")
    
    # Show sample entries
    if combined_data:
        print("\nSample entries:")
        for i, entry in enumerate(combined_data[:3]):
            print(f"\nEntry {i+1}:")
            print(f"  Volume: {entry['volume']}")
            print(f"  Question: {entry['question'][:100]}..." if len(entry['question']) > 100 else f"  Question: {entry['question']}")
            print(f"  Model 1: {entry['model_1_answer']}")
            print(f"  Model 2: {entry['model_2_answer']}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Combine predictions from two models')
    parser.add_argument('--model1', help='Path to model 1 predictions JSON file (optional)')
    parser.add_argument('--model2', help='Path to model 2 predictions JSON file (optional)')
    parser.add_argument('--user-study', help='Path to user study JSON file to filter volumes (optional)')
    parser.add_argument('--output', help='Output file basename (without .json/.csv extension)')
    
    args = parser.parse_args()
    
    # At least one model must be provided
    if not args.model1 and not args.model2:
        print("Error: At least one model predictions file must be provided")
        return
    
    try:
        combine_predictions(args.model1, args.model2, args.output_basename, args.user_study)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()