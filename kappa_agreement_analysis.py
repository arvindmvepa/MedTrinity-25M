#!/usr/bin/env python3
"""
Agreement metrics analysis using Cohen's kappa for clinical annotations vs predictions.
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
        Ground truth (correct) labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    
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

def load_data(dataset_type, prediction_file):
    """Load ground truth and prediction data"""
    # Load clinical data based on dataset type
    clinical_file = f'clinical_annotations_{dataset_type}_groundtruth_format.json'
    
    print(f"Loading clinical data from: {clinical_file}")
    with open(clinical_file, 'r') as f:
        clinical_data = json.load(f)
    
    print(f"Loading prediction data from: {prediction_file}")
    with open(prediction_file, 'r') as f:
        prediction_data = json.load(f)
    
    return clinical_data, prediction_data

def extract_case_name(seg_file_path):
    """Extract case name from seg_file path"""
    case_name = Path(seg_file_path).parent.name
    return case_name

def create_case_mapping(prediction_data):
    """Create mapping from case names to prediction data"""
    case_map = {}
    
    for i, pred in enumerate(prediction_data): 
        case_name = extract_case_name(pred['seg_file'])
        case_map[case_name] = pred
    
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

def get_label_types(dataset_type):
    """Get label types based on dataset"""
    if dataset_type == 'met':
        return ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", "Enhancing Tissue"]
    elif dataset_type == 'goat':
        return ["Necrosis", "Edema/Invaded Tissue", "Enhancing Tissue"]
    else:  # GLI
        return ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", "Enhancing Tissue", "Resection Cavity"]

def collect_task_data(clinical_data, prediction_data, dataset_type):
    """Collect aligned data for all tasks"""
    
    case_map = create_case_mapping(prediction_data)
    label_types = get_label_types(dataset_type)
    
    # Create mapping between clinical and prediction label names for GoAT
    label_mapping = {}
    if dataset_type == 'goat':
        label_mapping = {
            "Non-Enhancing Tumor": "Necrosis",
            "Surrounding Non-enhancing FLAIR hyperintensity": "Edema/Invaded Tissue",
            "Enhancing Tissue": "Enhancing Tumor"
        }
    
    # Initialize data collectors - overall and per label
    task_data = {
        'area': {'true': [], 'pred': []},
        'shape': {'true': [], 'pred': []},
        'satellite': {'true': [], 'pred': []},
        'region': {}  # Will store binary data for each region
    }
    
    # Initialize per-label data collectors using clinical label names
    per_label_data = {}
    clinical_label_types = ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", "Enhancing Tissue"] if dataset_type == 'goat' else label_types
    for label_type in clinical_label_types:
        per_label_data[label_type] = {
            'area': {'true': [], 'pred': []},
            'shape': {'true': [], 'pred': []},
            'satellite': {'true': [], 'pred': []},
            'region': {}
        }
    
    # Initialize region binary collectors
    region_names = get_category_mappings()['region']
    for region in region_names:
        task_data['region'][region] = {'true': [], 'pred': []}
        for label_type in clinical_label_types:
            per_label_data[label_type]['region'][region] = {'true': [], 'pred': []}
    task_data['region']['overall'] = {'true': [], 'pred': []}  # Overall region presence
    for label_type in clinical_label_types:
        per_label_data[label_type]['region']['overall'] = {'true': [], 'pred': []}

    matched_cases = 0
    unmatched_cases = []
    
    # Process each clinical case
    for clinical_case in clinical_data:
        case_name = clinical_case['mpMRI']
        
        if case_name not in case_map:
            unmatched_cases.append(case_name)
            continue
        
        matched_cases += 1
        pred_case = case_map[case_name]
        
        # Analyze each label type
        for clinical_label_type in clinical_label_types:
            # Map clinical label to prediction label for GoAT
            pred_label_type = label_mapping[clinical_label_type]
            assert clinical_label_type in clinical_case.get('labels', {}), f"Missing clinical label: {clinical_label_type}"
            assert pred_label_type in pred_case.get('labels', {}), f"Missing prediction label: {pred_label_type}"

                
            clinical_label = clinical_case['labels'][clinical_label_type]
            pred_label = pred_case['labels'][pred_label_type]
            
            # Multi-class tasks: area, shape, satellite
            for task in ['area', 'shape', 'satellite']:
                if task in clinical_label and task in pred_label:
                    true_val = clinical_label[task]
                    pred_val = pred_label[task] - 1  # Convert predictions from 1-indexed to 0-indexed
                    
                    # Overall data
                    task_data[task]['true'].append(true_val)
                    task_data[task]['pred'].append(pred_val)
                    
                    # Per-label data (use clinical label name for consistency)
                    per_label_data[clinical_label_type][task]['true'].append(true_val)
                    per_label_data[clinical_label_type][task]['pred'].append(pred_val)
            
            # Multi-label task: region
            if 'region' in clinical_label and 'region' in pred_label:
                true_regions = set(clinical_label['region'])
                pred_regions = set([r - 1 for r in pred_label['region']])  # Convert predictions from 1-indexed to 0-indexed
                
                # For each region, create binary labels
                for i, region in enumerate(region_names):
                    true_binary = 1 if i in true_regions else 0
                    pred_binary = 1 if i in pred_regions else 0
                    
                    # Overall data
                    task_data['region'][region]['true'].append(true_binary)
                    task_data['region'][region]['pred'].append(pred_binary)
                    task_data['region']['overall']['true'].append(true_binary)
                    task_data['region']['overall']['pred'].append(pred_binary)

                    # Per-label data (use clinical label name for consistency)
                    per_label_data[clinical_label_type]['region'][region]['true'].append(true_binary)
                    per_label_data[clinical_label_type]['region'][region]['pred'].append(pred_binary)
                    per_label_data[clinical_label_type]['region']['overall']['true'].append(true_binary)
                    per_label_data[clinical_label_type]['region']['overall']['pred'].append(pred_binary)
            
    return task_data, per_label_data

def compute_kappa_metrics(task_data):
    """Compute Cohen's kappa for all tasks"""
    
    results = {}
    
    print("=" * 80)
    print("COHEN'S KAPPA AGREEMENT METRICS")
    print("=" * 80)
    
    # Multi-class tasks
    print("\nMULTI-CLASS TASK KAPPA SCORES:")
    print("-" * 40)
    
    multiclass_kappas = []
    multiclass_accuracies = []
    
    for task in ['area', 'shape', 'satellite']:
        if len(task_data[task]['true']) > 0:
            true_labels = np.array(task_data[task]['true'])
            pred_labels = np.array(task_data[task]['pred'])
            
            kappa = cohen_kappa_score(true_labels, pred_labels)
            accuracy = np.mean(true_labels == pred_labels)
            multiclass_kappas.append(kappa)
            multiclass_accuracies.append(accuracy)
            
            results[task] = {
                'kappa': kappa,
                'accuracy': accuracy,
                'n_samples': len(true_labels),
                'interpretation': interpret_kappa(kappa)
            }
            
            print(f"{task.upper():12} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(true_labels):3}")
        else:
            print(f"{task.upper():12} No data available")
            results[task] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
    
    # Multi-label region task - binary kappa for each region
    print(f"\nMULTI-LABEL REGION BINARY KAPPA SCORES:")
    print("-" * 40)
    
    region_kappas = []
    region_accuracies = []
    region_results = {}
    
    for region in task_data['region']:
        if len(task_data['region'][region]['true']) > 0:
            true_binary = np.array(task_data['region'][region]['true'])
            pred_binary = np.array(task_data['region'][region]['pred'])
            
            accuracy = np.mean(true_binary == pred_binary)
            
            # Check if there's any variation in the data
            if len(np.unique(true_binary)) == 1 and len(np.unique(pred_binary)) == 1:
                # Both are constant - perfect agreement if same, no agreement if different
                if true_binary[0] == pred_binary[0]:
                    kappa = 1.0
                else:
                    kappa = 0.0
            else:
                kappa = cohen_kappa_score(true_binary, pred_binary)
            
            region_kappas.append(kappa)
            region_accuracies.append(accuracy)
            
            region_results[region] = {
                'kappa': kappa,
                'accuracy': accuracy,
                'n_samples': len(true_binary),
                'interpretation': interpret_kappa(kappa),
                'prevalence_true': np.mean(true_binary),
                'prevalence_pred': np.mean(pred_binary)
            }
            
            print(f"{region:15} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(true_binary):3} prev_true = {np.mean(true_binary):.3f} prev_pred = {np.mean(pred_binary):.3f}")
        else:
            print(f"{region:15} No data available")
            region_results[region] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
    
    # Average kappas
    print(f"\nAVERAGE KAPPA SCORES:")
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
    
    if valid_region_kappas:
        avg_region_kappa = np.mean(valid_region_kappas)
        avg_region_accuracy = np.mean(valid_region_accuracies)
        print(f"Average Region κ     = {avg_region_kappa:.4f} ({interpret_kappa(avg_region_kappa)}) acc = {avg_region_accuracy:.4f}")
    else:
        avg_region_kappa = None
        avg_region_accuracy = None
        print("Average Region κ     = No data available")
    
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
        'avg_region_kappa': avg_region_kappa,
        'avg_region_accuracy': avg_region_accuracy,
        'overall_avg_kappa': overall_avg_kappa,
        'overall_avg_accuracy': overall_avg_accuracy,
        'n_multiclass_tasks': len(valid_multiclass_kappas),
        'n_region_labels': len(valid_region_kappas)
    }
    
    results['region'] = region_results
    
    return results

def compute_per_label_kappa(per_label_data, dataset_type):
    """Compute Cohen's kappa for each label type separately"""
    
    label_results = {}
    # Use clinical label names for consistency in output
    clinical_label_types = ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", "Enhancing Tissue"] if dataset_type == 'goat' else get_label_types(dataset_type)
    if dataset_type not in ['met', 'goat']:  # GLI
        clinical_label_types.append("Resection Cavity")
    
    print("\n" + "=" * 80)
    print("PER-LABEL KAPPA ANALYSIS")
    print("=" * 80)
    
    for label_type in clinical_label_types:
        print(f"\n{label_type.upper()}:")
        print("-" * 60)
        
        label_results[label_type] = {}
        
        # Multi-class tasks for this label
        print("Multi-class tasks:")
        multiclass_kappas = []
        multiclass_accuracies = []
        
        for task in ['area', 'shape', 'satellite']:
            if len(per_label_data[label_type][task]['true']) > 0:
                true_labels = np.array(per_label_data[label_type][task]['true'])
                pred_labels = np.array(per_label_data[label_type][task]['pred'])
                
                kappa = cohen_kappa_score(true_labels, pred_labels)
                accuracy = np.mean(true_labels == pred_labels)
                multiclass_kappas.append(kappa)
                multiclass_accuracies.append(accuracy)
                
                label_results[label_type][task] = {
                    'kappa': kappa,
                    'accuracy': accuracy,
                    'n_samples': len(true_labels),
                    'interpretation': interpret_kappa(kappa)
                }
                
                print(f"  {task:10} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(true_labels):2}")
            else:
                label_results[label_type][task] = {'kappa': None, 'accuracy': None, 'n_samples': 0, 'interpretation': 'No data'}
                print(f"  {task:10} No data available")
        
        # Region tasks for this label
        print("Region tasks:")
        region_kappas = []
        region_accuracies = []
        label_results[label_type]['region'] = {}
        
        for region in per_label_data[label_type]['region']:
            if len(per_label_data[label_type]['region'][region]['true']) > 0:
                true_binary = np.array(per_label_data[label_type]['region'][region]['true'])
                pred_binary = np.array(per_label_data[label_type]['region'][region]['pred'])
                
                accuracy = np.mean(true_binary == pred_binary)
                
                # Check if there's any variation in the data
                if len(np.unique(true_binary)) == 1 and len(np.unique(pred_binary)) == 1:
                    if true_binary[0] == pred_binary[0]:
                        kappa = 1.0
                    else:
                        kappa = 0.0
                else:
                    kappa = cohen_kappa_score(true_binary, pred_binary)
                
                region_kappas.append(kappa)
                region_accuracies.append(accuracy)
                
                label_results[label_type]['region'][region] = {
                    'kappa': kappa,
                    'accuracy': accuracy,
                    'n_samples': len(true_binary),
                    'interpretation': interpret_kappa(kappa)
                }
                
                print(f"  {region:13} κ = {kappa:.4f} ({interpret_kappa(kappa):15}) acc = {accuracy:.4f} n = {len(true_binary):2}")
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
    print("DETAILED KAPPA ANALYSIS")
    print("=" * 80)
    
    # Multi-class task details
    for task in ['area', 'shape', 'satellite']:
        if results[task]['kappa'] is not None:
            print(f"\n{task.upper()} TASK ANALYSIS:")
            print("-" * 30)
            
            true_labels = np.array(task_data[task]['true'])
            pred_labels = np.array(task_data[task]['pred'])
            
            # Class distribution
            unique_true, counts_true = np.unique(true_labels, return_counts=True)
            unique_pred, counts_pred = np.unique(pred_labels, return_counts=True)
            
            print(f"True label distribution: {dict(zip(unique_true, counts_true))}")
            print(f"Pred label distribution: {dict(zip(unique_pred, counts_pred))}")
            print(f"Cohen's κ = {results[task]['kappa']:.4f} ({results[task]['interpretation']})")
            
            # Simple accuracy
            accuracy = np.mean(true_labels == pred_labels)
            print(f"Simple accuracy = {accuracy:.4f}")
    
    # Region task summary
    print(f"\nREGION TASK SUMMARY:")
    print("-" * 30)
    
    region_kappas = [r['kappa'] for r in results['region'].values() if r['kappa'] is not None]
    region_accuracies = []

    # Compute accuracies for each region
    for region_name, region_data in results['region'].items():
        if region_data['kappa'] is not None:
            true_binary = np.array(task_data['region'][region_name]['true'])
            pred_binary = np.array(task_data['region'][region_name]['pred'])
            accuracy = np.mean(true_binary == pred_binary)
            region_accuracies.append(accuracy)

    if region_kappas:
        print(f"Number of region labels: {len(region_kappas)}")
        print(f"Kappa range: {min(region_kappas):.4f} to {max(region_kappas):.4f}")
        print(f"Mean kappa: {np.mean(region_kappas):.4f}")
        print(f"Std kappa: {np.std(region_kappas):.4f}")
        print(f"Mean accuracy: {np.mean(region_accuracies):.4f}")
        print(f"Std accuracy: {np.std(region_accuracies):.4f}")
        
        # Identify best and worst performing regions
        best_region = max(results['region'].items(), key=lambda x: x[1]['kappa'] if x[1]['kappa'] is not None else -1)
        worst_region = min(results['region'].items(), key=lambda x: x[1]['kappa'] if x[1]['kappa'] is not None else 2)
        
        # Get accuracies for best/worst regions
        best_true = np.array(task_data['region'][best_region[0]]['true'])
        best_pred = np.array(task_data['region'][best_region[0]]['pred'])
        best_acc = np.mean(best_true == best_pred)
        
        worst_true = np.array(task_data['region'][worst_region[0]]['true'])
        worst_pred = np.array(task_data['region'][worst_region[0]]['pred'])
        worst_acc = np.mean(worst_true == worst_pred)
        
        print(f"Best agreement: {best_region[0]} (κ = {best_region[1]['kappa']:.4f}, acc = {best_acc:.4f})")
        print(f"Worst agreement: {worst_region[0]} (κ = {worst_region[1]['kappa']:.4f}, acc = {worst_acc:.4f})")

def save_kappa_results(results, task_data, label_results, output_file):
    """Save kappa analysis results to JSON file"""
    
    # Prepare serializable results
    serializable_results = {
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
    
    print(f"\nKappa analysis saved to: {output_file}")

def main():
    """Main kappa analysis function"""
    
    parser = argparse.ArgumentParser(description='Compute Cohen\'s kappa agreement metrics')
    parser.add_argument('dataset_type', choices=['gli', 'met', 'goat'], 
                       help='Dataset type: gli, met, or goat')
    parser.add_argument('prediction_file', help='Path to prediction JSON file')
    parser.add_argument('output_file', help='Path to output kappa analysis JSON file')
    
    args = parser.parse_args()
    dataset_type = args.dataset_type
    
    print(f"Dataset: {dataset_type}")
    print(f"Prediction file: {args.prediction_file}")
    print(f"Output file: {args.output_file}")
    
    try:
        print("Loading data for kappa analysis...")
        clinical_data, prediction_data = load_data(args.dataset_type, args.prediction_file)
        
        print("Collecting aligned task data...")
        task_data, per_label_data = collect_task_data(clinical_data, prediction_data, args.dataset_type)
        
        print("Computing Cohen's kappa metrics...")
        results = compute_kappa_metrics(task_data)
        
        # Compute per-label kappa metrics
        label_results = compute_per_label_kappa(per_label_data, args.dataset_type)
        
        # Generate detailed report
        create_detailed_kappa_report(results, task_data)
        
        # Save results (include both overall and per-label)
        save_kappa_results(results, task_data, label_results, args.output_file)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
    except Exception as e:
        print(f"Error during kappa analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
