#!/usr/bin/env python3
"""
Inter-annotator agreement analysis using Cohen's kappa for comparing two clinical annotation files.
Computes kappa for multi-class tasks and binary labels in multi-label region task.
"""

import json
import numpy as np
import argparse
from pathlib import Path

def cohen_kappa_score(y_true, y_pred):
    """
    Compute Cohen's kappa coefficient manually without sklearn dependency.
    
    Parameters:
    y_true : array-like of shape (n_samples,)
        First annotator labels.
    y_pred : array-like of shape (n_samples,)
        Second annotator labels.
    
    Returns:
    kappa : float
        The kappa statistic, which is a number between -1 and 1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Get unique labels
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    n_samples = len(y_true)
    
    # Create label-to-index mapping
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Create confusion matrix
    confusion_matrix = np.zeros((n_labels, n_labels), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        confusion_matrix[true_idx, pred_idx] += 1
    
    # Calculate observed agreement (accuracy)
    p_o = np.trace(confusion_matrix) / n_samples
    
    # Calculate expected agreement
    marginal_true = np.sum(confusion_matrix, axis=1) / n_samples
    marginal_pred = np.sum(confusion_matrix, axis=0) / n_samples
    p_e = np.sum(marginal_true * marginal_pred)
    
    # Calculate kappa
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa

def parse_file_specs(file_specs):
    """Parse file specifications in format [dataset_type:]file_path"""
    result = []
    for spec in file_specs:
        if ':' in spec:
            dataset_type, file_path = spec.split(':', 1)
        else:
            dataset_type, file_path = 'gli', spec  # Default for backward compatibility
        result.append((dataset_type, file_path))
    return result

def load_annotation_files(annotation_specs1, annotation_specs2):
    """Load annotation files with dataset type specifications"""
    
    annotations1, annotations2 = [], []
    
    # Load annotator 1 files
    for dataset_type, file_path in parse_file_specs(annotation_specs1):
        print(f"Loading annotator 1 {dataset_type} file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for annotation in data:
                annotation['dataset_type'] = dataset_type
            annotations1.extend(data)
    
    # Load annotator 2 files  
    for dataset_type, file_path in parse_file_specs(annotation_specs2):
        print(f"Loading annotator 2 {dataset_type} file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            for annotation in data:
                annotation['dataset_type'] = dataset_type
            annotations2.extend(data)
    
    return annotations1, annotations2

def extract_case_name(mpMRI_name):
    """Extract case name from mpMRI field"""
    # For groundtruth format, mpMRI contains the case name directly
    return mpMRI_name

def create_case_mapping(annotations):
    """Create mapping from case names to annotation data"""
    case_map = {}
    
    for annotation in annotations:
        case_name = extract_case_name(annotation['mpMRI'])
        case_map[case_name] = annotation
    
    return case_map

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
            "occipital", "limbic", "insula", "cerebellum"
        ]
    }

def detect_label_types(annotations):
    """Detect label types from the annotation data across all dataset types"""
    if not annotations:
        return []
    
    # Collect all unique label types across all annotations
    all_label_types = set()
    dataset_types = set()
    
    for annotation in annotations:
        dataset_type = annotation.get('dataset_type', 'gli')
        dataset_types.add(dataset_type)
        if 'labels' in annotation:
            all_label_types.update(annotation['labels'].keys())
    
    label_types = sorted(list(all_label_types))
    print(f"Detected dataset types: {sorted(list(dataset_types))}")
    print(f"Detected label types: {label_types}")
    return label_types

def collect_task_data(annotations1, annotations2):
    """Collect aligned data for all tasks from two annotation files"""
    
    case_map1 = create_case_mapping(annotations1)
    case_map2 = create_case_mapping(annotations2)
    
    # Find common cases
    common_cases = set(case_map1.keys()) & set(case_map2.keys())
    print(f"Found {len(common_cases)} common cases between annotations")
    
    if not common_cases:
        raise ValueError("No common cases found between the two annotation files")
    
    # Detect label types from first annotation file
    label_types = detect_label_types(annotations1)
    
    # Initialize data collectors - overall and per label
    task_data = {
        'area': {'annotator1': [], 'annotator2': []},
        'shape': {'annotator1': [], 'annotator2': []},
        'satellite': {'annotator1': [], 'annotator2': []},
        'region': {}  # Will store binary data for each region
    }
    
    # Initialize per-label data collectors
    per_label_data = {}
    for label_type in label_types:
        per_label_data[label_type] = {
            'area': {'annotator1': [], 'annotator2': []},
            'shape': {'annotator1': [], 'annotator2': []},
            'satellite': {'annotator1': [], 'annotator2': []},
            'region': {}
        }
    
    # Initialize region binary collectors
    region_names = get_category_mappings()['region']
    for region in region_names:
        task_data['region'][region] = {'annotator1': [], 'annotator2': []}
        for label_type in label_types:
            per_label_data[label_type]['region'][region] = {'annotator1': [], 'annotator2': []}

    processed_cases = 0
    
    # Process each common case
    for case_name in common_cases:
        case1 = case_map1[case_name]
        case2 = case_map2[case_name]
        
        processed_cases += 1
        
        # Analyze each label type
        for label_type in label_types:
            if label_type not in case1.get('labels', {}):
                print(f"Warning: Missing label type '{label_type}' in annotator 1 for case {case_name}")
                continue
            if label_type not in case2.get('labels', {}):
                print(f"Warning: Missing label type '{label_type}' in annotator 2 for case {case_name}")
                continue
                
            label1 = case1['labels'][label_type]
            label2 = case2['labels'][label_type]
            
            # Multi-class tasks: area, shape, satellite
            for task in ['area', 'shape', 'satellite']:
                if task in label1 and task in label2:
                    val1 = label1[task]
                    val2 = label2[task]
                    
                    # Overall data
                    task_data[task]['annotator1'].append(val1)
                    task_data[task]['annotator2'].append(val2)
                    
                    # Per-label data
                    per_label_data[label_type][task]['annotator1'].append(val1)
                    per_label_data[label_type][task]['annotator2'].append(val2)
            
            # Multi-label task: region
            if 'region' in label1 and 'region' in label2:
                regions1 = set(label1['region'])
                regions2 = set(label2['region'])
                
                # For each region, create binary labels
                for i, region in enumerate(region_names):
                    binary1 = 1 if i in regions1 else 0
                    binary2 = 1 if i in regions2 else 0
                    
                    # Overall data
                    task_data['region'][region]['annotator1'].append(binary1)
                    task_data['region'][region]['annotator2'].append(binary2)

                    # Per-label data
                    per_label_data[label_type]['region'][region]['annotator1'].append(binary1)
                    per_label_data[label_type]['region'][region]['annotator2'].append(binary2)
    
    print(f"Processed {processed_cases} cases successfully")
    return task_data, per_label_data, label_types

def compute_kappa_metrics(task_data):
    """Compute Cohen's kappa for all tasks"""
    
    results = {}
    
    print("=" * 80)
    print("INTER-ANNOTATOR COHEN'S KAPPA AGREEMENT METRICS")
    print("=" * 80)
    
    # Multi-class tasks
    print("\nMULTI-CLASS TASK KAPPA SCORES:")
    print("-" * 40)
    
    multiclass_kappas = []
    multiclass_accuracies = []
    
    # Collect all multi-class data for overall calculation
    all_multiclass_annotator1 = []
    all_multiclass_annotator2 = []
    
    for task in ['area', 'shape', 'satellite']:
        if len(task_data[task]['annotator1']) > 0:
            annotator1_labels = np.array(task_data[task]['annotator1'])
            annotator2_labels = np.array(task_data[task]['annotator2'])
            
            kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
            accuracy = np.mean(annotator1_labels == annotator2_labels)
            multiclass_kappas.append(kappa)
            multiclass_accuracies.append(accuracy)
            
            # Add to overall pooled data
            all_multiclass_annotator1.extend(annotator1_labels)
            all_multiclass_annotator2.extend(annotator2_labels)
            
            results[task] = {
                'kappa': kappa,
                'accuracy': accuracy,
                'n_samples': len(annotator1_labels),
                'interpretation': interpret_kappa(kappa)
            }
            
            print(f"{task.upper():12} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(annotator1_labels):3}")
        else:
            print(f"{task.upper():12} No data available")
            results[task] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
    
    # Calculate pooled overall multi-class kappa
    if all_multiclass_annotator1:
        overall_multiclass_kappa = cohen_kappa_score(all_multiclass_annotator1, all_multiclass_annotator2)
        overall_multiclass_accuracy = np.mean(np.array(all_multiclass_annotator1) == np.array(all_multiclass_annotator2))
        print(f"{'OVERALL':12} κ = {overall_multiclass_kappa:.4f} ({interpret_kappa(overall_multiclass_kappa):15}) acc = {overall_multiclass_accuracy:.4f} n = {len(all_multiclass_annotator1):3}")
        
        results['multiclass_overall'] = {
            'kappa': overall_multiclass_kappa,
            'accuracy': overall_multiclass_accuracy,
            'n_samples': len(all_multiclass_annotator1),
            'interpretation': interpret_kappa(overall_multiclass_kappa)
        }
    else:
        print(f"{'OVERALL':12} No data available")
        results['multiclass_overall'] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
    
    # Multi-label region task - binary kappa for each region
    print(f"\nMULTI-LABEL REGION BINARY KAPPA SCORES:")
    print("-" * 40)
    
    region_kappas = []
    region_accuracies = []
    region_results = {}
    
    # Get region names from category mappings
    region_names = get_category_mappings()['region']
    
    # Collect all region data for pooled calculation
    all_region_annotator1 = []
    all_region_annotator2 = []
    
    for region in region_names:
        if region in task_data['region'] and len(task_data['region'][region]['annotator1']) > 0:
            binary1 = np.array(task_data['region'][region]['annotator1'])
            binary2 = np.array(task_data['region'][region]['annotator2'])
            
            accuracy = np.mean(binary1 == binary2)
            
            # Check if there's any variation in the data
            if len(np.unique(binary1)) == 1 and len(np.unique(binary2)) == 1:
                # Both are constant - perfect agreement if same, no agreement if different
                if binary1[0] == binary2[0]:
                    kappa = 1.0
                else:
                    kappa = 0.0
            else:
                kappa = cohen_kappa_score(binary1, binary2)
            
            region_kappas.append(kappa)
            region_accuracies.append(accuracy)
            
            # Add to pooled region data
            all_region_annotator1.extend(binary1)
            all_region_annotator2.extend(binary2)
            
            region_results[region] = {
                'kappa': kappa,
                'accuracy': accuracy,
                'n_samples': len(binary1),
                'interpretation': interpret_kappa(kappa),
                'prevalence_annotator1': np.mean(binary1),
                'prevalence_annotator2': np.mean(binary2)
            }
            
            print(f"{region:15} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(binary1):3} prev_a1 = {np.mean(binary1):.3f} prev_a2 = {np.mean(binary2):.3f}")
        else:
            region_results[region] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
    
    # Calculate pooled region kappa (separate from individual results)
    pooled_region_kappa = None
    pooled_region_accuracy = None
    pooled_region_n_samples = 0
    
    if all_region_annotator1:
        pooled_region_kappa = cohen_kappa_score(all_region_annotator1, all_region_annotator2)
        pooled_region_accuracy = np.mean(np.array(all_region_annotator1) == np.array(all_region_annotator2))
        pooled_region_n_samples = len(all_region_annotator1)
        print(f"{'POOLED':15} κ = {pooled_region_kappa:.4f} ({interpret_kappa(pooled_region_kappa):15}) acc = {pooled_region_accuracy:.4f} n = {pooled_region_n_samples:3}")
    else:
        print(f"{'POOLED':15} No data available")
    
    # Average and pooled kappas
    print(f"\nSUMMARY KAPPA SCORES:")
    print("-" * 40)
    
    valid_multiclass_kappas = [k for k in multiclass_kappas if k is not None]
    valid_multiclass_accuracies = [a for a in multiclass_accuracies if a is not None]
    valid_region_kappas = [k for k in region_kappas if k is not None]
    valid_region_accuracies = [a for a in region_accuracies if a is not None]
    
    if valid_multiclass_kappas:
        avg_multiclass_kappa = np.mean(valid_multiclass_kappas)
        avg_multiclass_accuracy = np.mean(valid_multiclass_accuracies)
        print(f"Average Multi-class κ = {avg_multiclass_kappa:.4f} ({interpret_kappa(avg_multiclass_kappa)}) acc = {avg_multiclass_accuracy:.4f}")
    else:
        avg_multiclass_kappa = None
        avg_multiclass_accuracy = None
        print("Average Multi-class κ = No data available")
    
    # Show pooled overall multi-class kappa
    if results['multiclass_overall']['kappa'] is not None:
        pooled_kappa = results['multiclass_overall']['kappa']
        pooled_accuracy = results['multiclass_overall']['accuracy']
        print(f"Pooled Multi-class κ  = {pooled_kappa:.4f} ({interpret_kappa(pooled_kappa)}) acc = {pooled_accuracy:.4f}")
    else:
        pooled_kappa = None
        pooled_accuracy = None
        print("Pooled Multi-class κ  = No data available")
    
    if valid_region_kappas:
        avg_region_kappa = np.mean(valid_region_kappas)
        avg_region_accuracy = np.mean(valid_region_accuracies)
        print(f"Average Region κ     = {avg_region_kappa:.4f} ({interpret_kappa(avg_region_kappa)}) acc = {avg_region_accuracy:.4f}")
    else:
        avg_region_kappa = None
        avg_region_accuracy = None
        print("Average Region κ     = No data available")
    
    # Show pooled region kappa (calculated separately)
    if pooled_region_kappa is not None:
        print(f"Pooled Region κ      = {pooled_region_kappa:.4f} ({interpret_kappa(pooled_region_kappa)}) acc = {pooled_region_accuracy:.4f}")
    else:
        print("Pooled Region κ      = No data available")
    
    # Calculate pooled overall for ALL tasks (multi-class + region)
    all_task_annotator1 = all_multiclass_annotator1 + all_region_annotator1
    all_task_annotator2 = all_multiclass_annotator2 + all_region_annotator2
    
    if all_task_annotator1:
        pooled_all_kappa = cohen_kappa_score(all_task_annotator1, all_task_annotator2)
        pooled_all_accuracy = np.mean(np.array(all_task_annotator1) == np.array(all_task_annotator2))
        print(f"Pooled ALL Tasks κ   = {pooled_all_kappa:.4f} ({interpret_kappa(pooled_all_kappa)}) acc = {pooled_all_accuracy:.4f}")
    else:
        pooled_all_kappa = None
        pooled_all_accuracy = None
        print("Pooled ALL Tasks κ   = No data available")
    
    all_valid_kappas = valid_multiclass_kappas + valid_region_kappas
    all_valid_accuracies = valid_multiclass_accuracies + valid_region_accuracies
    if all_valid_kappas:
        overall_avg_kappa = np.mean(all_valid_kappas)
        overall_avg_accuracy = np.mean(all_valid_accuracies)
        print(f"Overall Average κ    = {overall_avg_kappa:.4f} ({interpret_kappa(overall_avg_kappa)}) acc = {overall_avg_accuracy:.4f}")
    else:
        overall_avg_kappa = None
        overall_avg_accuracy = None
        print("Overall Average κ    = No data available")
    
    # Store summary results
    results['summary'] = {
        'avg_multiclass_kappa': avg_multiclass_kappa,
        'avg_multiclass_accuracy': avg_multiclass_accuracy,
        'pooled_multiclass_kappa': pooled_kappa,
        'pooled_multiclass_accuracy': pooled_accuracy,
        'avg_region_kappa': avg_region_kappa,
        'avg_region_accuracy': avg_region_accuracy,
        'pooled_region_kappa': pooled_region_kappa,
        'pooled_region_accuracy': pooled_region_accuracy,
        'pooled_all_tasks_kappa': pooled_all_kappa,
        'pooled_all_tasks_accuracy': pooled_all_accuracy,
        'overall_avg_kappa': overall_avg_kappa,
        'overall_avg_accuracy': overall_avg_accuracy,
        'n_multiclass_tasks': len(valid_multiclass_kappas),
        'n_region_labels': len(valid_region_kappas)
    }
    
    results['region'] = region_results
    
    return results

def compute_per_label_kappa(per_label_data, label_types):
    """Compute Cohen's kappa for each label type separately"""
    
    label_results = {}
    
    print("\n" + "=" * 80)
    print("PER-LABEL INTER-ANNOTATOR KAPPA ANALYSIS")
    print("=" * 80)
    
    for label_type in label_types:
        print(f"\n{label_type.upper()}:")
        print("-" * 60)
        
        label_results[label_type] = {}
        
        # Multi-class tasks for this label
        print("Multi-class tasks:")
        multiclass_kappas = []
        multiclass_accuracies = []
        
        for task in ['area', 'shape', 'satellite']:
            if len(per_label_data[label_type][task]['annotator1']) > 0:
                annotator1_labels = np.array(per_label_data[label_type][task]['annotator1'])
                annotator2_labels = np.array(per_label_data[label_type][task]['annotator2'])
                
                kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
                accuracy = np.mean(annotator1_labels == annotator2_labels)
                multiclass_kappas.append(kappa)
                multiclass_accuracies.append(accuracy)
                
                label_results[label_type][task] = {
                    'kappa': kappa,
                    'accuracy': accuracy,
                    'n_samples': len(annotator1_labels),
                    'interpretation': interpret_kappa(kappa)
                }
                
                print(f"  {task:10} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(annotator1_labels):2}")
            else:
                label_results[label_type][task] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
                print(f"  {task:10} No data available")
        
        # Region tasks for this label
        print("Region tasks:")
        region_kappas = []
        region_accuracies = []
        label_results[label_type]['region'] = {}
        
        for region in per_label_data[label_type]['region']:
            if len(per_label_data[label_type]['region'][region]['annotator1']) > 0:
                binary1 = np.array(per_label_data[label_type]['region'][region]['annotator1'])
                binary2 = np.array(per_label_data[label_type]['region'][region]['annotator2'])
                
                accuracy = np.mean(binary1 == binary2)
                
                # Check if there's any variation in the data
                if len(np.unique(binary1)) == 1 and len(np.unique(binary2)) == 1:
                    if binary1[0] == binary2[0]:
                        kappa = 1.0
                    else:
                        kappa = 0.0
                else:
                    kappa = cohen_kappa_score(binary1, binary2)
                
                region_kappas.append(kappa)
                region_accuracies.append(accuracy)
                
                label_results[label_type]['region'][region] = {
                    'kappa': kappa,
                    'accuracy': accuracy,
                    'n_samples': len(binary1),
                    'interpretation': interpret_kappa(kappa)
                }
                
                print(f"  {region:13} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(binary1):2}")
            else:
                label_results[label_type]['region'][region] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
        
        # Summary for this label
        valid_multiclass = [k for k in multiclass_kappas if k is not None]
        valid_multiclass_acc = [a for a in multiclass_accuracies if a is not None]
        valid_region = [k for k in region_kappas if k is not None]
        valid_region_acc = [a for a in region_accuracies if a is not None]
        
        if valid_multiclass:
            avg_multiclass = np.mean(valid_multiclass)
            avg_multiclass_acc = np.mean(valid_multiclass_acc)
            print(f"  Average multi-class κ = {avg_multiclass:.4f} ({interpret_kappa(avg_multiclass)}) acc = {avg_multiclass_acc:.4f}")
        else:
            avg_multiclass = None
            avg_multiclass_acc = None
            
        if valid_region:
            avg_region = np.mean(valid_region)
            avg_region_acc = np.mean(valid_region_acc)
            print(f"  Average region κ      = {avg_region:.4f} ({interpret_kappa(avg_region)}) acc = {avg_region_acc:.4f}")
        else:
            avg_region = None
            avg_region_acc = None
        
        all_valid = valid_multiclass + valid_region
        all_valid_acc = valid_multiclass_acc + valid_region_acc
        if all_valid:
            overall_avg = np.mean(all_valid)
            overall_avg_acc = np.mean(all_valid_acc)
            print(f"  Overall average κ     = {overall_avg:.4f} ({interpret_kappa(overall_avg)}) acc = {overall_avg_acc:.4f}")
        else:
            overall_avg = None
            overall_avg_acc = None
            
        label_results[label_type]['summary'] = {
            'avg_multiclass_kappa': avg_multiclass,
            'avg_multiclass_accuracy': avg_multiclass_acc,
            'avg_region_kappa': avg_region,
            'avg_region_accuracy': avg_region_acc,
            'overall_avg_kappa': overall_avg,
            'overall_avg_accuracy': overall_avg_acc
        }
    
    return label_results

def interpret_kappa(kappa):
    """Interpret Cohen's kappa value according to Landis & Koch (1977)"""
    if kappa is None:
        return "No data"
    elif kappa < 0:
        return "Poor"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost perfect"

def create_detailed_kappa_report(results, task_data):
    """Create detailed report with confusion matrix info"""
    
    print(f"\n" + "=" * 80)
    print("DETAILED INTER-ANNOTATOR KAPPA ANALYSIS")
    print("=" * 80)
    
    # Multi-class task details
    for task in ['area', 'shape', 'satellite']:
        if results[task]['kappa'] is not None:
            print(f"\n{task.upper()} TASK ANALYSIS:")
            print("-" * 30)
            
            annotator1_labels = np.array(task_data[task]['annotator1'])
            annotator2_labels = np.array(task_data[task]['annotator2'])
            
            # Class distribution
            unique_a1, counts_a1 = np.unique(annotator1_labels, return_counts=True)
            unique_a2, counts_a2 = np.unique(annotator2_labels, return_counts=True)
            
            print(f"Annotator 1 distribution: {dict(zip(unique_a1, counts_a1))}")
            print(f"Annotator 2 distribution: {dict(zip(unique_a2, counts_a2))}")
            print(f"Cohen's κ = {results[task]['kappa']:.4f} ({results[task]['interpretation']})")
            
            # Simple accuracy
            accuracy = np.mean(annotator1_labels == annotator2_labels)
            print(f"Agreement rate = {accuracy:.4f}")
    
    # Region task summary
    print(f"\nREGION TASK SUMMARY:")
    print("-" * 30)
    
    region_kappas = [r['kappa'] for r in results['region'].values() if r['kappa'] is not None]
    region_accuracies = []

    # Compute accuracies for each region
    for region_name, region_data in results['region'].items():
        if region_data['kappa'] is not None:
            binary1 = np.array(task_data['region'][region_name]['annotator1'])
            binary2 = np.array(task_data['region'][region_name]['annotator2'])
            accuracy = np.mean(binary1 == binary2)
            region_accuracies.append(accuracy)

    if region_kappas:
        print(f"Number of region labels: {len(region_kappas)}")
        print(f"Kappa range: {min(region_kappas):.4f} to {max(region_kappas):.4f}")
        print(f"Mean kappa: {np.mean(region_kappas):.4f}")
        print(f"Std kappa: {np.std(region_kappas):.4f}")
        print(f"Mean agreement: {np.mean(region_accuracies):.4f}")
        print(f"Std agreement: {np.std(region_accuracies):.4f}")
        
        # Identify best and worst performing regions
        best_region = max(results['region'].items(), key=lambda x: x[1]['kappa'] if x[1]['kappa'] is not None else -1)
        worst_region = min(results['region'].items(), key=lambda x: x[1]['kappa'] if x[1]['kappa'] is not None else 2)
        
        # Get accuracies for best/worst regions
        best_binary1 = np.array(task_data['region'][best_region[0]]['annotator1'])
        best_binary2 = np.array(task_data['region'][best_region[0]]['annotator2'])
        best_acc = np.mean(best_binary1 == best_binary2)
        
        worst_binary1 = np.array(task_data['region'][worst_region[0]]['annotator1'])
        worst_binary2 = np.array(task_data['region'][worst_region[0]]['annotator2'])
        worst_acc = np.mean(worst_binary1 == worst_binary2)
        
        print(f"Best agreement: {best_region[0]} (κ = {best_region[1]['kappa']:.4f}, acc = {best_acc:.4f})")
        print(f"Worst agreement: {worst_region[0]} (κ = {worst_region[1]['kappa']:.4f}, acc = {worst_acc:.4f})")

def save_kappa_results(results, label_results, output_file, annotation_files1, annotation_files2):
    """Save kappa analysis results to JSON file"""
    
    # Prepare serializable results
    serializable_results = {
        'metadata': {
            'analysis_type': 'inter_annotator_agreement',
            'annotation_files1': annotation_files1,
            'annotation_files2': annotation_files2,
            'description': 'Inter-annotator agreement analysis using Cohen\'s kappa'
        },
        'overall_analysis': {
            'kappa_scores': {},
            'accuracies': {},
            'summary_statistics': results['summary'],
            'sample_sizes': {},
            'interpretations': {}
        },
        'per_label_analysis': {}
    }
    
    # Multi-class tasks (overall)
    for task in ['area', 'shape', 'satellite']:
        serializable_results['overall_analysis']['kappa_scores'][task] = results[task]['kappa']
        serializable_results['overall_analysis']['accuracies'][task] = results[task]['accuracy']
        serializable_results['overall_analysis']['sample_sizes'][task] = results[task]['n_samples']
        serializable_results['overall_analysis']['interpretations'][task] = results[task]['interpretation']
    
    # Region tasks (overall)
    serializable_results['overall_analysis']['kappa_scores']['region'] = {}
    serializable_results['overall_analysis']['accuracies']['region'] = {}
    serializable_results['overall_analysis']['sample_sizes']['region'] = {}
    serializable_results['overall_analysis']['interpretations']['region'] = {}
    
    for region, data in results['region'].items():
        serializable_results['overall_analysis']['kappa_scores']['region'][region] = data['kappa']
        serializable_results['overall_analysis']['accuracies']['region'][region] = data['accuracy']
        serializable_results['overall_analysis']['sample_sizes']['region'][region] = data['n_samples']
        serializable_results['overall_analysis']['interpretations']['region'][region] = data['interpretation']
    
    # Per-label analysis
    for label_type, label_data in label_results.items():
        serializable_results['per_label_analysis'][label_type] = {
            'kappa_scores': {},
            'accuracies': {},
            'sample_sizes': {},
            'interpretations': {},
            'summary_statistics': label_data['summary']
        }
        
        # Multi-class tasks for this label
        for task in ['area', 'shape', 'satellite']:
            serializable_results['per_label_analysis'][label_type]['kappa_scores'][task] = label_data[task]['kappa']
            serializable_results['per_label_analysis'][label_type]['accuracies'][task] = label_data[task]['accuracy']
            serializable_results['per_label_analysis'][label_type]['sample_sizes'][task] = label_data[task]['n_samples']
            serializable_results['per_label_analysis'][label_type]['interpretations'][task] = label_data[task]['interpretation']
        
        # Region tasks for this label
        serializable_results['per_label_analysis'][label_type]['kappa_scores']['region'] = {}
        serializable_results['per_label_analysis'][label_type]['accuracies']['region'] = {}
        serializable_results['per_label_analysis'][label_type]['sample_sizes']['region'] = {}
        serializable_results['per_label_analysis'][label_type]['interpretations']['region'] = {}
        
        for region, region_data in label_data['region'].items():
            serializable_results['per_label_analysis'][label_type]['kappa_scores']['region'][region] = region_data['kappa']
            serializable_results['per_label_analysis'][label_type]['accuracies']['region'][region] = region_data['accuracy']
            serializable_results['per_label_analysis'][label_type]['sample_sizes']['region'][region] = region_data['n_samples']
            serializable_results['per_label_analysis'][label_type]['interpretations']['region'][region] = region_data['interpretation']
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nInter-annotator agreement analysis saved to: {output_file}")

def main():
    """Main inter-annotator agreement analysis function"""
    
    parser = argparse.ArgumentParser(description='Compute Cohen\'s kappa inter-annotator agreement metrics')
    parser.add_argument('annotation_files1', nargs='+', 
                       help='Annotation files for annotator 1. Single file or format: dataset_type:file_path')
    parser.add_argument('annotation_files2', nargs='+',
                       help='Annotation files for annotator 2. Single file or format: dataset_type:file_path')
    parser.add_argument('--output', '-o', 
                       help='Path to output kappa analysis JSON file (default: inter_annotator_agreement.json)',
                       default='inter_annotator_agreement.json')
    
    args = parser.parse_args()
    
    print(f"Annotator 1 files: {args.annotation_files1}")
    print(f"Annotator 2 files: {args.annotation_files2}")
    print(f"Output file: {args.output}")
    
    try:
        print("Loading annotation files for inter-annotator agreement analysis...")
        annotations1, annotations2 = load_annotation_files(args.annotation_files1, args.annotation_files2)
        
        print("Collecting aligned task data...")
        task_data, per_label_data, label_types = collect_task_data(annotations1, annotations2)
        
        print("Computing Cohen's kappa metrics...")
        results = compute_kappa_metrics(task_data)
        
        # Compute per-label kappa metrics
        label_results = compute_per_label_kappa(per_label_data, label_types)
        
        # Generate detailed report
        create_detailed_kappa_report(results, task_data)
        
        # Save results
        save_kappa_results(results, label_results, args.output, args.annotation_files1, args.annotation_files2)
        
        print(f"\n" + "=" * 80)
        print("INTER-ANNOTATOR AGREEMENT ANALYSIS COMPLETE")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
    except Exception as e:
        print(f"Error during inter-annotator agreement analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()