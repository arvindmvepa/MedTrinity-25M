#!/usr/bin/env python3
"""
Generate comprehensive metrics for clinical annotations evaluation.

This script compares:
- Clinical annotations (ground truth): clinical_annotations_groundtruth_format.json
- Model predictions: brats_gli_3d_vqa_subjTrue_test_aux_updated_v3_seed0.json

Metrics:
- Multi-class accuracy for area, shape, satellite
- Multi-label accuracy for region (localization)
- Per-label per-task results
- Overall task averages
- Overall system average
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

def load_data():
    """Load ground truth and prediction data"""
    
    # Load clinical annotations (ground truth)
    with open('clinical_annotations_groundtruth_format.json', 'r') as f:
        clinical_data = json.load(f)
    
    # Load model predictions
    with open('brats_gli_3d_vqa_subjTrue_test_aux_updated_v3_seed0.json', 'r') as f:
        prediction_data = json.load(f)
    
    return clinical_data, prediction_data

def extract_case_name(seg_file_path):
    """Extract case name from seg_file path"""
    # Extract case name from path like '/local2/.../BraTS-GLI-00063-100/BraTS-GLI-00063-100-seg.nii.gz'
    case_name = Path(seg_file_path).parent.name
    return case_name

def create_case_mapping(prediction_data):
    """Create mapping from case names to prediction data"""
    case_map = {}
    for pred in prediction_data:
        if 'seg_file' in pred:
            case_name = extract_case_name(pred['seg_file'])
            case_map[case_name] = pred
    return case_map

def multi_class_accuracy(true_labels, pred_labels):
    """Calculate multi-class accuracy"""
    if len(true_labels) != len(pred_labels):
        return 0.0
    
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return correct / len(true_labels)

def multi_label_accuracy(true_regions, pred_regions, num_regions=11):
    """
    Calculate multi-label accuracy for region predictions.
    For each region, check if it's correctly predicted as present/absent.
    """
    # Convert region lists to binary vectors
    true_binary = [0] * num_regions
    pred_binary = [0] * num_regions
    
    for region in true_regions:
        if 0 <= region < num_regions:
            true_binary[region] = 1
    
    for region in pred_regions:
        if 0 <= region < num_regions:
            pred_binary[region] = 1
    
    # Calculate accuracy per region (correct present/absent predictions)
    correct = sum(1 for t, p in zip(true_binary, pred_binary) if t == p)
    return correct / num_regions

def evaluate_metrics(clinical_data, prediction_data):
    """Evaluate all metrics"""
    
    # Create case mapping for predictions
    case_map = create_case_mapping(prediction_data)
    
    # Initialize metric storage
    results = {
        'per_label_per_task': defaultdict(lambda: defaultdict(list)),
        'task_averages': {},
        'overall_average': 0.0,
        'coverage': {
            'total_clinical_cases': len(clinical_data),
            'matched_cases': 0,
            'unmatched_cases': []
        }
    }
    
    # Label types to evaluate
    label_types = ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", 
                  "Enhancing Tissue", "Resection Cavity"]
    
    # Tasks to evaluate
    tasks = ['area', 'region', 'shape', 'satellite']
    
    # Process each clinical case
    for clinical_case in clinical_data:
        case_name = clinical_case['mpMRI']
        
        if case_name not in case_map:
            results['coverage']['unmatched_cases'].append(case_name)
            continue
        
        results['coverage']['matched_cases'] += 1
        pred_case = case_map[case_name]
        
        # Evaluate each label type
        for label_type in label_types:
            if label_type not in clinical_case.get('labels', {}):
                continue
            if label_type not in pred_case.get('labels', {}):
                continue
                
            clinical_label = clinical_case['labels'][label_type]
            pred_label = pred_case['labels'][label_type]
            
            # Evaluate each task
            for task in tasks:
                if task not in clinical_label or task not in pred_label:
                    continue
                
                if task == 'region':  # Multi-label task
                    # Convert predictions from 1-indexed to 0-indexed for regions
                    pred_regions_corrected = [r - 1 for r in pred_label[task]]
                    accuracy = multi_label_accuracy(
                        clinical_label[task], 
                        pred_regions_corrected
                    )
                else:  # Multi-class tasks (area, shape, satellite)
                    # Convert predictions from 1-indexed to 0-indexed
                    pred_val_corrected = pred_label[task] - 1
                    accuracy = 1.0 if clinical_label[task] == pred_val_corrected else 0.0
                
                results['per_label_per_task'][label_type][task].append(accuracy)
    
    # Calculate task averages
    all_task_scores = defaultdict(list)
    
    for label_type in label_types:
        for task in tasks:
            if results['per_label_per_task'][label_type][task]:
                task_avg = np.mean(results['per_label_per_task'][label_type][task])
                results['per_label_per_task'][label_type][f'{task}_avg'] = task_avg
                all_task_scores[task].append(task_avg)
    
    # Calculate overall task averages
    for task in tasks:
        if all_task_scores[task]:
            results['task_averages'][task] = np.mean(all_task_scores[task])
    
    # Calculate overall average
    if results['task_averages']:
        results['overall_average'] = np.mean(list(results['task_averages'].values()))
    
    return results

def print_detailed_results(results):
    """Print comprehensive results"""
    
    print("=" * 80)
    print("CLINICAL ANNOTATIONS EVALUATION METRICS")
    print("=" * 80)
    
    # Coverage information
    print(f"\nCOVERAGE:")
    print(f"Total clinical cases: {results['coverage']['total_clinical_cases']}")
    print(f"Matched cases: {results['coverage']['matched_cases']}")
    print(f"Coverage rate: {results['coverage']['matched_cases']/results['coverage']['total_clinical_cases']*100:.1f}%")
    
    if results['coverage']['unmatched_cases']:
        print(f"Unmatched cases: {results['coverage']['unmatched_cases']}")
    
    # Per-label per-task results
    print(f"\nPER-LABEL PER-TASK RESULTS:")
    print("-" * 80)
    
    label_types = ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", 
                  "Enhancing Tissue", "Resection Cavity"]
    tasks = ['area', 'region', 'shape', 'satellite']
    
    # Header
    print(f"{'Label Type':<45} {'Area':<8} {'Region':<8} {'Shape':<8} {'Satellite':<10}")
    print("-" * 80)
    
    for label_type in label_types:
        row = f"{label_type:<45}"
        for task in tasks:
            if f'{task}_avg' in results['per_label_per_task'][label_type]:
                avg = results['per_label_per_task'][label_type][f'{task}_avg']
                row += f"{avg:.3f}".ljust(8)
            else:
                row += "N/A".ljust(8)
        print(row)
    
    # Task averages
    print(f"\nTASK AVERAGES:")
    print("-" * 40)
    for task, avg in results['task_averages'].items():
        task_name = task.capitalize()
        if task == 'region':
            task_name += " (Multi-label)"
        else:
            task_name += " (Multi-class)"
        print(f"{task_name:<25}: {avg:.3f}")
    
    # Overall average
    print(f"\nOVERALL AVERAGE: {results['overall_average']:.3f}")
    
    # Detailed breakdown
    print(f"\nDETAILED BREAKDOWN:")
    print("-" * 80)
    
    for label_type in label_types:
        print(f"\n{label_type}:")
        for task in tasks:
            if results['per_label_per_task'][label_type][task]:
                scores = results['per_label_per_task'][label_type][task]
                print(f"  {task.capitalize():<12}: {np.mean(scores):.3f} "
                      f"(n={len(scores)}, std={np.std(scores):.3f})")

def save_results(results, filename='clinical_evaluation_metrics.json'):
    """Save results to JSON file"""
    
    # Convert defaultdict to regular dict for JSON serialization
    serializable_results = {
        'per_label_per_task': {
            label: dict(tasks) for label, tasks in results['per_label_per_task'].items()
        },
        'task_averages': results['task_averages'],
        'overall_average': results['overall_average'],
        'coverage': results['coverage']
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")

def main():
    """Main evaluation function"""
    
    try:
        # Load data
        print("Loading data...")
        clinical_data, prediction_data = load_data()
        print(f"Loaded {len(clinical_data)} clinical cases and {len(prediction_data)} predictions")
        
        # Evaluate metrics
        print("Evaluating metrics...")
        results = evaluate_metrics(clinical_data, prediction_data)
        
        # Print results
        print_detailed_results(results)
        
        # Save results
        save_results(results)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure both clinical_annotations_groundtruth_format.json and")
        print("brats_gli_3d_vqa_subjTrue_test_aux_updated_v2_seed0.json exist in the current directory.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
