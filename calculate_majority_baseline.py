#!/usr/bin/env python3
"""
Calculate majority baseline metrics for clinical annotations evaluation.

This script calculates what performance you would get by always predicting
the most common class for each task (majority classifier baseline).
"""

import json
import numpy as np
from collections import Counter

def load_data():
    """Load ground truth data"""
    with open('clinical_annotations_groundtruth_format.json', 'r') as f:
        clinical_data = json.load(f)
    return clinical_data

def get_category_mappings():
    """Get category mappings for interpretation"""
    return {
        'area': {
            0: "N/A", 1: "<1%", 2: "1-5%", 3: "5-10%", 
            4: "10-25%", 5: "25-50%", 6: "50-75%"
        },
        'shape': {
            0: "N/A", 1: "focus", 2: "round", 3: "oval", 
            4: "elongated", 5: "irregular"
        },
        'satellite': {
            0: "N/A", 1: "single lesion", 2: "core with satellite lesions", 
            3: "scattered lesions"
        },
        'region': [
            "N/A", "subcortical", "frontal", "temporal", "parietal", 
            "occipital", "limbic", "insula", "cerebellum", "brainstem", "corpus callosum"
        ]
    }

def analyze_ground_truth_distributions(clinical_data):
    """Analyze distribution of labels in ground truth"""
    
    label_types = ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", 
                  "Enhancing Tissue", "Resection Cavity"]
    
    # Track distributions for each label type and task
    distributions = {}
    
    for label_type in label_types:
        distributions[label_type] = {
            'area': Counter(),
            'shape': Counter(), 
            'satellite': Counter(),
            'region_binary': {}  # Changed to track binary presence for each region
        }
    
    # Initialize binary region tracking
    region_names = [
        "N/A", "subcortical", "frontal", "temporal", "parietal", 
        "occipital", "limbic", "insula", "cerebellum", "brainstem", "corpus callosum"
    ]
    
    for label_type in label_types:
        for region_idx in range(len(region_names)):
            distributions[label_type]['region_binary'][region_idx] = {'present': 0, 'absent': 0}
    
    # Process each clinical case
    for clinical_case in clinical_data:
        for label_type in label_types:
            if label_type not in clinical_case.get('labels', {}):
                continue
                
            clinical_label = clinical_case['labels'][label_type]
            
            # Count occurrences for each task
            for task in ['area', 'shape', 'satellite']:
                if task in clinical_label:
                    distributions[label_type][task][clinical_label[task]] += 1
            
            # Special handling for region (multi-label binary classification)
            if 'region' in clinical_label:
                present_regions = set(clinical_label['region'])
                
                # For each possible region, mark as present or absent
                for region_idx in range(len(region_names)):
                    if region_idx in present_regions:
                        distributions[label_type]['region_binary'][region_idx]['present'] += 1
                    else:
                        distributions[label_type]['region_binary'][region_idx]['absent'] += 1
    
    return distributions

def calculate_majority_baselines(distributions):
    """Calculate what performance you'd get with majority classifiers"""
    
    category_maps = get_category_mappings()
    
    print("=" * 80)
    print("MAJORITY BASELINE ANALYSIS")
    print("=" * 80)
    
    results = {}
    
    for label_type, label_distributions in distributions.items():
        print(f"\n{label_type.upper()}:")
        print("-" * 60)
        
        results[label_type] = {}
        
        for task, task_distribution in label_distributions.items():
            if not task_distribution:
                continue
            
            # Skip the old region task, use region_binary instead    
            if task == 'region_binary':
                # For multi-label region task, calculate binary classifier baselines
                print("  Region distribution:")
                
                region_accuracies = []
                binary_results = {}
                
                for region_idx, counts in task_distribution.items():
                    present_count = counts['present']
                    absent_count = counts['absent']
                    total_count = present_count + absent_count
                    
                    if total_count == 0:
                        continue
                        
                    region_name = category_maps['region'][region_idx] if region_idx < len(category_maps['region']) else f"Unknown_{region_idx}"
                    
                    # For binary classification, majority accuracy is max(present_rate, absent_rate)
                    present_rate = present_count / total_count
                    absent_rate = absent_count / total_count
                    majority_accuracy = max(present_rate, absent_rate)
                    majority_class = 'present' if present_rate > absent_rate else 'absent'
                    
                    # Only show regions that appear in at least one case
                    if present_count > 0:
                        print(f"    {region_name}: {present_count}/{total_count} ({present_rate*100:.1f}%) present")
                        region_accuracies.append(majority_accuracy)
                        binary_results[region_idx] = {
                            'majority_class': majority_class,
                            'majority_accuracy': majority_accuracy,
                            'present_count': present_count,
                            'total_count': total_count
                        }
                
                # Average binary classifier accuracy across all regions
                if region_accuracies:
                    avg_region_accuracy = np.mean(region_accuracies)
                    print(f"  → Multi-label region baseline accuracy: {avg_region_accuracy:.3f} (average across binary classifiers)")
                    
                    results[label_type]['region'] = {
                        'binary_results': binary_results,
                        'majority_accuracy': avg_region_accuracy,
                        'num_regions': len(region_accuracies)
                    }
                else:
                    results[label_type]['region'] = {
                        'binary_results': {},
                        'majority_accuracy': 0.0,
                        'num_regions': 0
                    }
                
            else:
                # For single-label tasks
                total_samples = sum(task_distribution.values())
                most_common = task_distribution.most_common()
                majority_class, majority_count = most_common[0]
                majority_accuracy = majority_count / total_samples
                
                print(f"\n  {task.capitalize()} distribution:")
                for class_idx, count in most_common:
                    class_name = category_maps[task].get(class_idx, f"Unknown_{class_idx}")
                    pct = (count / total_samples) * 100
                    marker = " ← MAJORITY" if class_idx == majority_class else ""
                    print(f"    {class_name}: {count}/{total_samples} ({pct:.1f}%){marker}")
                
                majority_name = category_maps[task].get(majority_class, f"Unknown_{majority_class}")
                print(f"  → Majority baseline accuracy: {majority_accuracy:.3f} (always predict '{majority_name}')")
                
                results[label_type][task] = {
                    'majority_class': majority_class,
                    'majority_class_name': majority_name,
                    'majority_count': majority_count,
                    'total_samples': total_samples,
                    'majority_accuracy': majority_accuracy
                }
    
    return results

def calculate_overall_majority_baselines(results):
    """Calculate overall task averages for majority baselines"""
    
    print("\n" + "=" * 80)
    print("MAJORITY BASELINE SUMMARY")
    print("=" * 80)
    
    # Calculate task averages
    task_averages = {}
    
    for task in ['area', 'shape', 'satellite', 'region']:
        accuracies = []
        for label_type, label_results in results.items():
            if task in label_results:
                accuracies.append(label_results[task]['majority_accuracy'])
        
        if accuracies:
            task_averages[task] = np.mean(accuracies)
    
    print("\nMAJORITY BASELINE TASK AVERAGES:")
    print("-" * 40)
    for task, avg_accuracy in task_averages.items():
        print(f"{task.capitalize()} (Multi-{'label' if task == 'region' else 'class'})       : {avg_accuracy:.3f}")
    
    # Overall average
    overall_average = np.mean(list(task_averages.values()))
    print(f"\nOVERALL MAJORITY BASELINE: {overall_average:.3f}")
    
    return task_averages, overall_average

def compare_with_model_performance():
    """Load and compare with actual model performance"""
    
    try:
        # Try to load model results if they exist
        with open('clinical_evaluation_metrics.json', 'r') as f:
            model_results = json.load(f)
        
        print("\n" + "=" * 80)
        print("MODEL VS MAJORITY BASELINE COMPARISON")
        print("=" * 80)
        
        model_overall = model_results.get('overall_average', 0.0)
        
        print(f"\nModel Performance     : {model_overall:.3f}")
        print("Majority Baseline     : (calculated above)")
        
        task_comparison = model_results.get('task_averages', {})
        if task_comparison:
            print("\nTask-by-task comparison:")
            print("-" * 40)
            for task, model_acc in task_comparison.items():
                print(f"{task.capitalize()}: Model={model_acc:.3f}")
        
    except FileNotFoundError:
        print("\nNote: Run 'python evaluate_clinical_metrics.py' first to compare with model performance")

def save_majority_baseline_results(results, task_averages, overall_average):
    """Save results to JSON file"""
    
    output = {
        'majority_baselines': results,
        'task_averages': task_averages,
        'overall_average': overall_average,
        'description': 'Majority baseline performance - always predict most common class for each task'
    }
    
    with open('majority_baseline_metrics.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nMajority baseline results saved to: majority_baseline_metrics.json")

def main():
    """Main majority baseline analysis"""
    
    try:
        print("Loading ground truth data...")
        clinical_data = load_data()
        
        print("Analyzing ground truth distributions...")
        distributions = analyze_ground_truth_distributions(clinical_data)
        
        print("Calculating majority baselines...")
        results = calculate_majority_baselines(distributions)
        
        task_averages, overall_average = calculate_overall_majority_baselines(results)
        
        compare_with_model_performance()
        
        save_majority_baseline_results(results, task_averages, overall_average)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
