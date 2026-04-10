#!/usr/bin/env python3
"""
Combine predictions from two models into a single JSON and CSV file.
Questions are loaded from a single groundtruth file and matched with model predictions by order.
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

def load_user_study_data(user_study_file):
    """Load volume/question combinations from user study JSON file"""
    if not user_study_file:
        return None
    
    print(f"Loading user study file: {user_study_file}")
    with open(user_study_file, 'r') as f:
        user_study_data = json.load(f)
    
    # Extract unique volume/question combinations from user study
    volume_question_pairs = set()
    for entry in user_study_data:
        if 'volume' in entry and entry['volume'] and 'question' in entry and entry['question']:
            volume = entry['volume']
            question = clean_question(entry['question'])
            volume_question_pairs.add((volume, question))
    
    print(f"Found {len(volume_question_pairs)} unique volume/question pairs in user study")
    return volume_question_pairs

def load_groundtruth_data(gt_file):
    """Load groundtruth data with questions and volumes"""
    if not gt_file:
        return []
    
    print(f"Loading groundtruth file: {gt_file}")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    
    print(f"Loaded {len(gt_data)} entries from groundtruth")
    return gt_data

def match_predictions_with_groundtruth(gt_data, model_predictions, model_name):
    """Match model predictions with groundtruth questions by order"""
    matched_data = []
    
    if not model_predictions:
        print(f"{model_name}: No predictions provided")
        return matched_data
    
    print(f"{model_name}: Matching {len(model_predictions)} predictions with {len(gt_data)} groundtruth entries")
    
    for i, gt_entry in enumerate(gt_data):
        # Extract volume from groundtruth
        volume = ""
        if 'volume_file_dir' in gt_entry and gt_entry['volume_file_dir']:
            volume = extract_volume_basename(gt_entry['volume_file_dir'])
        elif 'seg_file' in gt_entry and gt_entry['seg_file']:
            volume = extract_volume_basename(str(Path(gt_entry['seg_file']).parent))
        
        # Extract question from groundtruth
        question = ""
        if 'orig_question' in gt_entry:
            question = clean_question(gt_entry['orig_question'])
        elif 'question' in gt_entry:
            question = clean_question(gt_entry['question'])
        
        # Get corresponding model prediction (same index)
        model_answer = None
        if i < len(model_predictions):
            pred = model_predictions[i]
            # Try different field names for model answer
            for field in ['model_answer', 'pred', 'prediction', 'answer', 'response']:
                if field in pred and pred[field] is not None:
                    model_answer = pred[field]
                    break
            if model_answer is None:
                model_answer = ""
        
        matched_entry = {
            'volume': volume,
            'question': question,
            'model_answer': model_answer,
            'gt_index': i
        }
        
        matched_data.append(matched_entry)
    
    print(f"{model_name}: Successfully matched {len(matched_data)} entries")
    return matched_data

def combine_predictions(model1_file, model2_file, gt_file, output_basename, user_study_file=None):
    """Combine predictions from two models using groundtruth file for questions"""
    
    # Load user study volume/question pairs if provided
    filter_pairs = load_user_study_data(user_study_file) if user_study_file else None
    
    # Load groundtruth data
    gt_data = load_groundtruth_data(gt_file)
    
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
    
    # Match predictions with groundtruth
    model1_matched = match_predictions_with_groundtruth(gt_data, model1_predictions, "Model 1")
    model2_matched = match_predictions_with_groundtruth(gt_data, model2_predictions, "Model 2")
    
    # Create mappings from (volume, question) to prediction
    model1_map = {(entry['volume'], entry['question']): entry for entry in model1_matched}
    model2_map = {(entry['volume'], entry['question']): entry for entry in model2_matched}
    
    # Get all volume/question pairs to process
    if filter_pairs:
        # Use only pairs from user study
        pairs_to_process = filter_pairs
        print(f"Processing {len(pairs_to_process)} volume/question pairs from user study")
    else:
        # Use all pairs from groundtruth
        pairs_to_process = {(entry['volume'], entry['question']) for entry in model1_matched + model2_matched if entry['volume'] and entry['question']}
        print(f"Processing {len(pairs_to_process)} volume/question pairs from groundtruth")
    
    # Combine predictions
    combined_data = []
    
    for volume, question in sorted(pairs_to_process):
        model1_entry = model1_map.get((volume, question))
        model2_entry = model2_map.get((volume, question))
        
        model1_answer = model1_entry['model_answer'] if model1_entry else None
        model2_answer = model2_entry['model_answer'] if model2_entry else None
        
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
    non_null_model1 = sum(1 for entry in combined_data if entry['model_1_answer'] is not None and entry['model_1_answer'] != "")
    non_null_model2 = sum(1 for entry in combined_data if entry['model_2_answer'] is not None and entry['model_2_answer'] != "")
    print(f"  - Entries with Model 1 predictions: {non_null_model1}")
    print(f"  - Entries with Model 2 predictions: {non_null_model2}")
    
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
    parser = argparse.ArgumentParser(description='Combine predictions from two models using groundtruth file for questions')
    parser.add_argument('--model1', help='Path to model 1 predictions JSON file (optional)')
    parser.add_argument('--model2', help='Path to model 2 predictions JSON file (optional)')
    parser.add_argument('--gt', required=True, help='Path to groundtruth JSON file with questions')
    parser.add_argument('--user-study', help='Path to user study JSON file to filter volume/question pairs (optional)')
    parser.add_argument('--output', required=True, help='Output file basename (without .json/.csv extension)')
    
    args = parser.parse_args()
    
    # At least one model must be provided
    if not args.model1 and not args.model2:
        print("Error: At least one model predictions file must be provided")
        return
    
    try:
        combine_predictions(
            model1_file=args.model1, 
            model2_file=args.model2, 
            gt_file=args.gt,
            output_basename=args.output, 
            user_study_file=args.user_study
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()