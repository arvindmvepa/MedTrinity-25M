#!/usr/bin/env python3
"""
Detailed error analysis for clinical annotations evaluation.
Provides comprehensive breakdown of error types and patterns to understand
where the model is failing and guide data creation improvements.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

def load_data():
    """Load ground truth and prediction data"""
    with open('clinical_annotations_groundtruth_format.json', 'r') as f:
        clinical_data = json.load(f)
    
    with open('brats_gli_3d_vqa_subjTrue_test_aux_updated_v3_seed0.json', 'r') as f:
        prediction_data = json.load(f)
    
    return clinical_data, prediction_data

def extract_case_name(seg_file_path):
    """Extract case name from seg_file path"""
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

def analyze_task_errors(clinical_data, prediction_data):
    """Detailed error analysis for each task"""
    
    case_map = create_case_mapping(prediction_data)
    category_maps = get_category_mappings()
    
    # Initialize error tracking
    error_analysis = {
        'area': defaultdict(int),
        'shape': defaultdict(int),
        'satellite': defaultdict(int),
        'region': defaultdict(int)
    }
    
    detailed_errors = []
    label_types = ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", 
                  "Enhancing Tissue", "Resection Cavity"]
    
    # Process each clinical case
    for clinical_case in clinical_data:
        case_name = clinical_case['mpMRI']
        
        if case_name not in case_map:
            continue
            
        pred_case = case_map[case_name]
        
        # Analyze each label type
        for label_type in label_types:
            if (label_type not in clinical_case.get('labels', {}) or 
                label_type not in pred_case.get('labels', {})):
                continue
                
            clinical_label = clinical_case['labels'][label_type]
            pred_label = pred_case['labels'][label_type]
            
            # Analyze each task
            for task in ['area', 'shape', 'satellite']:
                if task in clinical_label and task in pred_label:
                    true_val = clinical_label[task]
                    pred_val = pred_label[task] - 1  # Convert predictions from 1-indexed to 0-indexed
                    
                    if true_val != pred_val:
                        # Record error pattern
                        true_name = category_maps[task].get(true_val, f"Unknown_{true_val}")
                        pred_name = category_maps[task].get(pred_val, f"Unknown_{pred_val}")
                        
                        error_analysis[task][f"{true_name} → {pred_name}"] += 1
                        
                        detailed_errors.append({
                            'case': case_name,
                            'label': label_type,
                            'task': task,
                            'true_value': true_val,
                            'pred_value': pred_val,  # Store the corrected 0-indexed value
                            'true_name': true_name,
                            'pred_name': pred_name,
                            'error_type': f"{true_name} → {pred_name}"
                        })
            
            # Special handling for region (multi-label)
            if 'region' in clinical_label and 'region' in pred_label:
                true_regions = set(clinical_label['region'])
                pred_regions = set([r - 1 for r in pred_label['region']])  # Convert predictions from 1-indexed to 0-indexed
                
                # False positives
                false_positives = pred_regions - true_regions
                for fp in false_positives:
                    region_name = category_maps['region'][fp] if fp < len(category_maps['region']) else f"Unknown_{fp}"
                    error_analysis['region'][f"False Positive: {region_name}"] += 1
                
                # False negatives  
                false_negatives = true_regions - pred_regions
                for fn in false_negatives:
                    region_name = category_maps['region'][fn] if fn < len(category_maps['region']) else f"Unknown_{fn}"
                    error_analysis['region'][f"False Negative: {region_name}"] += 1
                
                # Record detailed region errors
                if false_positives or false_negatives:
                    detailed_errors.append({
                        'case': case_name,
                        'label': label_type,
                        'task': 'region',
                        'true_regions': list(true_regions),
                        'pred_regions': list(pred_regions),  # Store the corrected 0-indexed values
                        'false_positives': list(false_positives),
                        'false_negatives': list(false_negatives)
                    })
    
    return error_analysis, detailed_errors

def analyze_confusion_patterns(error_analysis):
    """Analyze confusion patterns for each task"""
    
    print("=" * 80)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 80)
    
    # Area errors
    print(f"\nAREA PREDICTION ERRORS:")
    print(f"Most common area confusion patterns:")
    area_errors = dict(error_analysis['area'])
    if area_errors:
        sorted_area = sorted(area_errors.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_area[:10]:
            print(f"  {pattern}: {count} cases")
    else:
        print("  No area errors found")
    
    # Shape errors  
    print(f"\nSHAPE PREDICTION ERRORS:")
    print(f"Most common shape confusion patterns:")
    shape_errors = dict(error_analysis['shape'])
    if shape_errors:
        sorted_shape = sorted(shape_errors.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_shape[:10]:
            print(f"  {pattern}: {count} cases")
    else:
        print("  No shape errors found")
    
    # Satellite errors
    print(f"\nSATELLITE PREDICTION ERRORS:")
    print(f"Most common satellite confusion patterns:")
    satellite_errors = dict(error_analysis['satellite'])
    if satellite_errors:
        sorted_satellite = sorted(satellite_errors.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_satellite[:10]:
            print(f"  {pattern}: {count} cases")
    else:
        print("  No satellite errors found")
    
    # Region errors
    print(f"\nREGION PREDICTION ERRORS:")
    print(f"Most common region errors:")
    region_errors = dict(error_analysis['region'])
    if region_errors:
        sorted_region = sorted(region_errors.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_region[:10]:
            print(f"  {pattern}: {count} cases")
    else:
        print("  No region errors found")

def analyze_error_correlations(detailed_errors):
    """Analyze correlations between different types of errors"""
    
    print(f"\nERROR CORRELATION ANALYSIS:")
    print("-" * 40)
    
    # Group errors by case and label
    case_label_errors = defaultdict(lambda: defaultdict(list))
    for error in detailed_errors:
        key = f"{error['case']}_{error['label']}"
        case_label_errors[key][error['task']].append(error)
    
    # Find cases with multiple task errors
    multi_error_cases = 0
    error_combinations = defaultdict(int)
    
    for case_label, task_errors in case_label_errors.items():
        if len(task_errors) > 1:
            multi_error_cases += 1
            tasks = sorted(task_errors.keys())
            combo = " + ".join(tasks)
            error_combinations[combo] += 1
    
    print(f"Cases with errors in multiple tasks: {multi_error_cases}")
    print(f"Most common error combinations:")
    for combo, count in sorted(error_combinations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {combo}: {count} cases")

def analyze_label_specific_patterns(detailed_errors):
    """Analyze error patterns specific to each label type"""
    
    print(f"\nLABEL-SPECIFIC ERROR PATTERNS:")
    print("-" * 40)
    
    label_errors = defaultdict(lambda: defaultdict(int))
    
    for error in detailed_errors:
        if error['task'] != 'region':  # Skip region for this analysis
            label_errors[error['label']][error['error_type']] += 1
    
    for label_type in ["Non-Enhancing Tumor", "Surrounding Non-enhancing FLAIR hyperintensity", 
                      "Enhancing Tissue", "Resection Cavity"]:
        print(f"\n{label_type}:")
        if label_type in label_errors:
            sorted_errors = sorted(label_errors[label_type].items(), 
                                 key=lambda x: x[1], reverse=True)
            for error_type, count in sorted_errors[:5]:
                print(f"  {error_type}: {count}")
        else:
            print("  No errors recorded")

def generate_improvement_recommendations(error_analysis, detailed_errors):
    """Generate specific recommendations for data creation improvements"""
    
    print(f"\nDATA CREATION IMPROVEMENT RECOMMENDATIONS:")
    print("=" * 60)
    
    # Shape analysis
    shape_errors = dict(error_analysis['shape'])
    if shape_errors:
        print(f"\n1. SHAPE ANNOTATION IMPROVEMENTS:")
        common_shape_confusions = sorted(shape_errors.items(), key=lambda x: x[1], reverse=True)[:3]
        for confusion, count in common_shape_confusions:
            print(f"   - {confusion} ({count} cases)")
            
        # Specific recommendations
        if any("irregular" in conf for conf, _ in common_shape_confusions):
            print(f"   → Consider more detailed shape guidelines for 'irregular' vs other categories")
        if any("infiltrative" in conf for conf, _ in common_shape_confusions):
            print(f"   → Add clearer criteria for distinguishing 'infiltrative' patterns")
    
    # Satellite analysis  
    satellite_errors = dict(error_analysis['satellite'])
    if satellite_errors:
        print(f"\n2. SATELLITE LESION IMPROVEMENTS:")
        common_satellite_confusions = sorted(satellite_errors.items(), key=lambda x: x[1], reverse=True)[:3]
        for confusion, count in common_satellite_confusions:
            print(f"   - {confusion} ({count} cases)")
            
        # Specific recommendations
        if any("single lesion" in conf for conf, _ in common_satellite_confusions):
            print(f"   → Improve criteria for distinguishing single vs multi-focal lesions")
        if any("scattered" in conf for conf, _ in common_satellite_confusions):
            print(f"   → Add clearer definitions for 'scattered' vs 'multifocal' patterns")
    
    # Area analysis
    area_errors = dict(error_analysis['area'])
    if area_errors:
        print(f"\n3. AREA/VOLUME IMPROVEMENTS:")
        common_area_confusions = sorted(area_errors.items(), key=lambda x: x[1], reverse=True)[:3]
        for confusion, count in common_area_confusions:
            print(f"   - {confusion} ({count} cases)")
            
        # Check for systematic over/under-estimation
        overestimation = sum(count for conf, count in area_errors.items() 
                           if "→" in conf and extract_volume_direction(conf) > 0)
        underestimation = sum(count for conf, count in area_errors.items() 
                            if "→" in conf and extract_volume_direction(conf) < 0)
        
        if overestimation > underestimation * 1.5:
            print(f"   → Model tends to overestimate volumes - consider recalibration")
        elif underestimation > overestimation * 1.5:
            print(f"   → Model tends to underestimate volumes - consider recalibration")
    
    # Region analysis
    region_errors = dict(error_analysis['region'])
    if region_errors:
        print(f"\n4. REGION LOCALIZATION IMPROVEMENTS:")
        false_positives = {k: v for k, v in region_errors.items() if "False Positive" in k}
        false_negatives = {k: v for k, v in region_errors.items() if "False Negative" in k}
        
        if false_positives:
            print(f"   Most over-predicted regions:")
            for region_error, count in sorted(false_positives.items(), key=lambda x: x[1], reverse=True)[:3]:
                region = region_error.replace("False Positive: ", "")
                print(f"   - {region}: {count} false positives")
        
        if false_negatives:
            print(f"   Most under-predicted regions:")
            for region_error, count in sorted(false_negatives.items(), key=lambda x: x[1], reverse=True)[:3]:
                region = region_error.replace("False Negative: ", "")
                print(f"   - {region}: {count} missed detections")

def extract_volume_direction(confusion_pattern):
    """Extract direction of volume error (positive = overestimation)"""
    volume_order = ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"]
    
    try:
        parts = confusion_pattern.split(" → ")
        true_vol = parts[0].strip()
        pred_vol = parts[1].strip()
        
        true_idx = volume_order.index(true_vol) if true_vol in volume_order else -1
        pred_idx = volume_order.index(pred_vol) if pred_vol in volume_order else -1
        
        if true_idx >= 0 and pred_idx >= 0:
            return pred_idx - true_idx
    except:
        pass
    
    return 0

def save_detailed_analysis(error_analysis, detailed_errors):
    """Save detailed analysis to JSON file"""
    
    # Convert defaultdict to regular dict for JSON serialization
    serializable_analysis = {
        'error_patterns': {
            task: dict(patterns) for task, patterns in error_analysis.items()
        },
        'detailed_errors': detailed_errors,
        'summary_stats': {
            'total_errors': len(detailed_errors),
            'errors_by_task': {
                task: len([e for e in detailed_errors if e['task'] == task])
                for task in ['area', 'shape', 'satellite', 'region']
            }
        }
    }
    
    with open('detailed_error_analysis.json', 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: detailed_error_analysis.json")

def main():
    """Main error analysis function"""
    
    try:
        print("Loading data for error analysis...")
        clinical_data, prediction_data = load_data()
        
        print("Analyzing error patterns...")
        error_analysis, detailed_errors = analyze_task_errors(clinical_data, prediction_data)
        
        # Generate comprehensive analysis
        analyze_confusion_patterns(error_analysis)
        analyze_error_correlations(detailed_errors)
        analyze_label_specific_patterns(detailed_errors)
        generate_improvement_recommendations(error_analysis, detailed_errors)
        
        # Save results
        save_detailed_analysis(error_analysis, detailed_errors)
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
