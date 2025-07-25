#!/usr/bin/env python3
"""
Script to evaluate clinical annotations quality against the JSON auxiliary data
"""
import pandas as pd
import json
import re
from pathlib import Path

def load_json_data():
    """Load the JSON auxiliary data"""
    with open('brats_gli_3d_vqa_subjTrue_test_aux_v6_seed0.json', 'r') as f:
        json_data = json.load(f)
    return json_data

def load_clinical_data():
    """Load the converted clinical annotations"""
    with open('clinical_annotations_numerical.json', 'r') as f:
        clinical_data = json.load(f)
    return clinical_data

def extract_case_id_from_path(seg_file_path):
    """Extract case ID from segmentation file path"""
    # Extract pattern like BraTS-GLI-00063-101 from path
    pattern = r'BraTS-GLI-\d+-\d+'
    match = re.search(pattern, seg_file_path)
    return match.group(0) if match else None

def match_cases(json_data, clinical_data):
    """Match cases between JSON and clinical data"""
    
    # Create mapping from JSON data
    json_cases = {}
    for entry in json_data:
        case_id = extract_case_id_from_path(entry['seg_file'])
        if case_id:
            json_cases[case_id] = entry
    
    # Create mapping from clinical data
    clinical_cases = {}
    for entry in clinical_data['clinical_annotations']:
        case_id = entry['case_id']
        clinical_cases[case_id] = entry
    
    # Find matches
    matched_cases = []
    for case_id in clinical_cases:
        if case_id in json_cases:
            matched_cases.append({
                'case_id': case_id,
                'json_data': json_cases[case_id],
                'clinical_data': clinical_cases[case_id]
            })
    
    print(f"Found {len(matched_cases)} matching cases out of {len(clinical_cases)} clinical annotations")
    print(f"JSON data contains {len(json_cases)} cases total")
    
    return matched_cases

def evaluate_volume_correlation(matched_cases):
    """Evaluate correlation between clinical volume annotations and JSON area data"""
    
    correlations = []
    volume_mapping = {0: "<1%", 1: "1-5%", 2: "5-10%", 3: "10-25%", -1: "missing"}
    
    for case in matched_cases:
        case_id = case['case_id']
        json_labels = case['json_data']['labels']
        clinical_labels = case['clinical_data']['clinical_annotations']
        
        print(f"\n--- Case: {case_id} ---")
        
        # Map label names
        label_mapping = {
            "Non-Enhancing Tumor": "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity": "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue": "Enhancing Tissue", 
            "Resection Cavity": "Resection Cavity"
        }
        
        for clinical_label, json_label in label_mapping.items():
            if clinical_label in clinical_labels and json_label in json_labels:
                clinical_vol = clinical_labels[clinical_label]['volume']
                json_area = json_labels[json_label]['area']
                
                print(f"  {clinical_label}:")
                print(f"    Clinical volume: {volume_mapping.get(clinical_vol, 'unknown')} (code: {clinical_vol})")
                print(f"    JSON area: {json_area}")
                
                # Store for analysis
                if clinical_vol != -1:  # Not missing
                    correlations.append({
                        'case_id': case_id,
                        'label': clinical_label,
                        'clinical_volume_code': clinical_vol,
                        'json_area': json_area
                    })
    
    return correlations

def analyze_location_data(matched_cases):
    """Analyze location annotations"""
    
    brain_regions = ["frontal", "temporal", "parietal", "occipital", "subcortical", "limbic", "insula"]
    
    for case in matched_cases[:3]:  # Show first 3 cases
        case_id = case['case_id']
        clinical_labels = case['clinical_data']['clinical_annotations']
        
        print(f"\n--- Location Analysis for {case_id} ---")
        
        for label_name, label_data in clinical_labels.items():
            location_vec = label_data['location']
            if any(loc != -1 for loc in location_vec):  # Has location data
                active_regions = [brain_regions[i] for i, val in enumerate(location_vec) if val == 1]
                print(f"  {label_name}: {', '.join(active_regions) if active_regions else 'No regions'}")

def generate_evaluation_report(matched_cases, correlations):
    """Generate a comprehensive evaluation report"""
    
    print("\n" + "="*60)
    print("CLINICAL ANNOTATIONS QUALITY EVALUATION REPORT")
    print("="*60)
    
    print(f"\n1. DATA COVERAGE:")
    print(f"   - Total clinical annotations: {len(matched_cases)}")
    print(f"   - Volume correlations available: {len(correlations)}")
    
    print(f"\n2. VOLUME ANNOTATION ANALYSIS:")
    if correlations:
        # Group by volume categories
        vol_groups = {}
        for corr in correlations:
            vol_code = corr['clinical_volume_code']
            if vol_code not in vol_groups:
                vol_groups[vol_code] = []
            vol_groups[vol_code].append(corr['json_area'])
        
        volume_mapping = {0: "<1%", 1: "1-5%", 2: "5-10%", 3: "10-25%"}
        
        for vol_code, areas in vol_groups.items():
            vol_name = volume_mapping.get(vol_code, f"Code {vol_code}")
            avg_area = sum(areas) / len(areas)
            print(f"   - {vol_name}: {len(areas)} annotations, avg JSON area = {avg_area:.2f}")
    
    print(f"\n3. MATCHED CASES:")
    for i, case in enumerate(matched_cases[:5]):  # Show first 5
        print(f"   {i+1}. {case['case_id']}")
    
    print(f"\n4. RECOMMENDATIONS:")
    print(f"   - Use this data to train/evaluate VQA models on clinical annotations")
    print(f"   - Consider the volume-area correlations for validation")
    print(f"   - Location multilabel data can be used for spatial reasoning questions")

def main():
    """Main evaluation function"""
    
    print("Loading data...")
    json_data = load_json_data()
    clinical_data = load_clinical_data()
    
    print("Matching cases...")
    matched_cases = match_cases(json_data, clinical_data)
    
    if not matched_cases:
        print("No matching cases found! Check case ID formats.")
        return
    
    print("Evaluating volume correlations...")
    correlations = evaluate_volume_correlation(matched_cases)
    
    print("Analyzing location data...")
    analyze_location_data(matched_cases)
    
    print("Generating evaluation report...")
    generate_evaluation_report(matched_cases, correlations)
    
    # Save matched data for further analysis
    output_data = {
        'matched_cases': len(matched_cases),
        'correlations': correlations,
        'case_ids': [case['case_id'] for case in matched_cases]
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEvaluation results saved to 'evaluation_results.json'")

if __name__ == "__main__":
    main()
