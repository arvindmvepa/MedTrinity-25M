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
    
    # Remove <s><image> from beginning and <|endoftext|> from end
    cleaned = question.strip()
    cleaned = re.sub(r'^<s><image>', '', cleaned)
    cleaned = re.sub(r'<\|endoftext\|>$', '', cleaned)
    return cleaned.strip()

def extract_volume_basename(volume_file_dir):
    """Extract basename from volume_file_dir path"""
    if not volume_file_dir:
        return ""
    return Path(volume_file_dir).name

def combine_predictions(model1_file, model2_file, output_basename):
    """Combine predictions from two models into JSON and CSV files"""
    
    # Load model predictions
    print(f"Loading model 1 predictions from: {model1_file}")
    with open(model1_file, 'r') as f:
        model1_predictions = json.load(f)
    
    print(f"Loading model 2 predictions from: {model2_file}")
    with open(model2_file, 'r') as f:
        model2_predictions = json.load(f)
    
    print(f"Model 1 predictions: {len(model1_predictions)}")
    print(f"Model 2 predictions: {len(model2_predictions)}")
    
    # Ensure both models have the same number of predictions
    assert len(model1_predictions) == len(model2_predictions), f"Different number of predictions."
    
    # Combine predictions
    combined_data = []
    
    for i in range(min_length):
        model1_pred = model1_predictions[i]
        model2_pred = model2_predictions[i]
        
        # Extract volume basename - try model2 first (has volume_file_dir), then model1
        volume = ""
        if 'volume_file_dir' in model2_pred and model2_pred['volume_file_dir']:
            volume = extract_volume_basename(model2_pred['volume_file_dir'])
        elif 'seg_file' in model1_pred and model1_pred['seg_file']:
            # Extract from seg_file path if volume_file_dir not available
            volume = extract_volume_basename(str(Path(model1_pred['seg_file']).parent))
        
        # Extract and clean question
        question = ""
        if 'orig_question' in model1_pred:
            question = clean_question(model1_pred['orig_question'])
        elif 'question' in model1_pred:
            question = clean_question(model1_pred['question'])
        
        # Extract model answers
        model1_answer = model1_pred.get('model_answer', '')
        model2_answer = model2_pred.get('pred', '')
        
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
    parser.add_argument('model1_predictions', help='Path to model 1 predictions JSON file')
    parser.add_argument('model2_predictions', help='Path to model 2 predictions JSON file')
    parser.add_argument('output_basename', help='Output file basename (without .json/.csv extension)')
    
    args = parser.parse_args()
    
    try:
        combine_predictions(args.model1_predictions, args.model2_predictions, args.output_basename)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()