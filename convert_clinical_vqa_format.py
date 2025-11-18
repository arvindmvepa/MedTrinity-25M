#!/usr/bin/env python3
"""
Updated conversion script using the exact VQA category mappings with 0-based indexing
"""
import pandas as pd
import json

def debug_excel_structure():
    """Debug the Excel file structure thoroughly"""
    print("=== DEBUGGING EXCEL STRUCTURE ===")
    
    df = pd.read_excel('clinical-annotation.xlsx')
    print(f"Excel shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n=== FIRST 30 ROWS (RAW DATA) ===")
    for i in range(min(30, len(df))):
        row_values = []
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if pd.isna(val):
                row_values.append("NaN")
            else:
                row_values.append(f"'{val}'")
        print(f"Row {i:2d}: {' | '.join(row_values)}")
    
    return df

def extract_first_case(df, num_labels=4):
    """Extract the first case (from column name and rows 1-4)"""
    # The first case ID is in the column name
    first_case_id = df.columns[0]  # 'BraTS-GLI-00063-101' or 'BraTS-MET-...'
    
    print(f"\n=== EXTRACTING FIRST CASE: {first_case_id} ===")
    
    # Data is in rows 1-4
    question_types = ['volume', 'location', 'shape', 'spread out']
    case_data = {}
    
    for q_idx, question in enumerate(question_types):
        row_idx = q_idx + 1  # Start from row 1
        
        # Check question name
        question_cell = df.iloc[row_idx, 0]
        print(f"  Row {row_idx}, Col 0: '{question_cell}' (expected: '{question}')")
        
        # Extract answers for num_labels (columns 1 to num_labels+1)
        answers = []
        for col_idx in range(1, num_labels + 1):
            if col_idx < len(df.columns):
                answer = df.iloc[row_idx, col_idx]
                if pd.isna(answer):
                    answers.append(None)
                else:
                    answers.append(str(answer).strip())
            else:
                answers.append(None)
        
        case_data[question] = answers
        print(f"  {question}: {answers}")
    
    return first_case_id, case_data

def find_other_cases(df, num_labels=4):
    """Find all other cases that start with explicit case IDs"""
    print("\n=== FINDING OTHER CASES ===")
    
    other_cases = []
    
    for i in range(len(df)):
        val = df.iloc[i, 0]
        if isinstance(val, str) and (val.startswith('BraTS-GLI-') or val.startswith('BraTS-MET-')):
            # This is a case ID row
            case_id = val
            
            # Skip the header row (should be row i+1)
            header_row = i + 1
            if header_row < len(df):
                header_check = df.iloc[header_row, 1:num_labels+1].tolist()
                print(f"Found case at row {i}: {case_id}")
                print(f"  Header row {header_row}: {header_check}")
                
                # Extract data from rows i+2 to i+5
                case_data = {}
                question_types = ['volume', 'location', 'shape', 'spread out']
                
                for q_idx, question in enumerate(question_types):
                    data_row = i + 2 + q_idx
                    if data_row < len(df):
                        # Check question name
                        question_cell = df.iloc[data_row, 0]
                        print(f"    Row {data_row}, Question: '{question_cell}' (expected: '{question}')")
                        
                        # Extract answers
                        answers = []
                        for col_idx in range(1, num_labels + 1):
                            if col_idx < len(df.columns):
                                answer = df.iloc[data_row, col_idx]
                                if pd.isna(answer):
                                    answers.append(None)
                                else:
                                    answers.append(str(answer).strip())
                            else:
                                answers.append(None)
                        
                        case_data[question] = answers
                        print(f"    {question}: {answers}")
                
                other_cases.append((case_id, case_data))
    
    return other_cases

def convert_to_numerical_vqa_format(case_id, case_data, label_names=None):
    """Convert case data to numerical format using VQA system mappings (0-based indexing)"""
    print(f"\n=== CONVERTING {case_id} TO VQA NUMERICAL FORMAT ===")
    
    # VQA system mappings (0-based indexing)
    volume_categories = ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"]
    volume_mapping = {cat.lower(): i for i, cat in enumerate(volume_categories)}
    
    shape_categories = ["N/A", "focus", "round", "oval", "elongated", "irregular"]
    shape_mapping = {cat.lower(): i for i, cat in enumerate(shape_categories)}
    
    satellite_categories = ["N/A", "single lesion", "core with satellite lesions", "scattered lesions"]
    satellite_mapping = {cat.lower(): i for i, cat in enumerate(satellite_categories)}
    # Add aliases for satellite mapping
    satellite_mapping["scattered"] = satellite_mapping["scattered lesions"]  # Handle "scattered" as "scattered lesions"
    
    brain_regions = ["n/a", "frontal", "parietal", "occipital", "temporal", "limbic", "insula", "subcortical", "cerebellum"]
    lobe_mapping = {region.lower(): i for i, region in enumerate(brain_regions)}
    
    # Use provided label names or default to GLI labels
    if label_names is None:
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue", 
            "Resection Cavity"
        ]
    
    numerical_case = {
        "case_id": case_id,
        "clinical_annotations": {}
    }
    
    # Process each label
    for label_idx, label_name in enumerate(label_names):
        print(f"\n  Processing {label_name} (index {label_idx}):")
        
        label_data = {}
        
        # Volume
        if 'volume' in case_data and label_idx < len(case_data['volume']):
            volume_answer = case_data['volume'][label_idx]
            if volume_answer is None:
                label_data['volume'] = 0  # N/A
                print("    Volume: Missing -> 0 (N/A)")
            else:
                volume_clean = volume_answer.lower().strip()
                if volume_clean in volume_mapping:
                    label_data['volume'] = volume_mapping[volume_clean]
                    print(f"    Volume: '{volume_answer}' -> {label_data['volume']} ({volume_categories[label_data['volume']]})")
                else:
                    print(f"    ERROR: Unknown volume value '{volume_answer}'")
                    print(f"    Valid values: {volume_categories}")
                    raise ValueError(f"Unknown volume value: {volume_answer}")
        else:
            label_data['volume'] = 0  # N/A
            print("    Volume: Missing -> 0 (N/A)")
        
        # Location (multilabel using VQA lobe indices)
        if 'location' in case_data and label_idx < len(case_data['location']):
            location_answer = case_data['location'][label_idx]
            location_indices = encode_location_vqa_format(location_answer, brain_regions, lobe_mapping)
            label_data['location'] = location_indices
            print(f"    Location: '{location_answer}' -> {location_indices}")
        else:
            label_data['location'] = [0]  # N/A
            print(f"    Location: Missing -> [0] (N/A)")
        
        # Shape
        if 'shape' in case_data and label_idx < len(case_data['shape']):
            shape_answer = case_data['shape'][label_idx]
            if shape_answer is None:
                label_data['shape'] = 0  # N/A
                print("    Shape: Missing -> 0 (N/A)")
            else:
                shape_clean = shape_answer.lower().strip()
                if shape_clean in shape_mapping:
                    label_data['shape'] = shape_mapping[shape_clean]
                    print(f"    Shape: '{shape_answer}' -> {label_data['shape']} ({shape_categories[label_data['shape']]})")
                else:
                    print(f"    ERROR: Unknown shape value '{shape_answer}'")
                    print(f"    Valid values: {shape_categories}")
                    raise ValueError(f"Unknown shape value: {shape_answer}")
        else:
            label_data['shape'] = 0  # N/A
            print("    Shape: Missing -> 0 (N/A)")
        
        # Spread pattern (satellite)
        if 'spread out' in case_data and label_idx < len(case_data['spread out']):
            spread_answer = case_data['spread out'][label_idx]
            if spread_answer is None:
                label_data['satellite'] = 0  # N/A
                print("    Satellite: Missing -> 0 (N/A)")
            else:
                spread_clean = spread_answer.lower().strip()
                if spread_clean in satellite_mapping:
                    label_data['satellite'] = satellite_mapping[spread_clean]
                    print(f"    Satellite: '{spread_answer}' -> {label_data['satellite']} ({satellite_categories[label_data['satellite']] if label_data['satellite'] < len(satellite_categories) else 'scattered lesions'})")
                else:
                    print(f"    ERROR: Unknown satellite pattern '{spread_answer}'")
                    print(f"    Valid values: {satellite_categories}")
                    raise ValueError(f"Unknown satellite pattern: {spread_answer}")
        else:
            label_data['satellite'] = 0  # N/A
            print("    Satellite: Missing -> 0 (N/A)")
        
        numerical_case['clinical_annotations'][label_name] = label_data
    
    return numerical_case

def encode_location_vqa_format(location_str, brain_regions, lobe_mapping):
    """Encode location using VQA format (return sorted list of indices)"""
    if location_str is None:
        return [0]  # N/A
    
    # Clean the location string and split by commas
    location_clean = location_str.lower().replace(' ', '').replace(',', ',')
    present_regions = [region.strip() for region in location_clean.split(',') if region.strip()]
    
    print(f"      Location parsing: '{location_str}' -> regions: {present_regions}")
    
    # Find matching lobe indices
    found_indices = []
    for region_str in present_regions:
        # Check for exact matches first
        if region_str in lobe_mapping:
            found_indices.append(lobe_mapping[region_str])
        else:
            # Check for partial matches (e.g., "frontal" in "frontal,subcortical")
            for lobe, idx in lobe_mapping.items():
                if lobe != "n/a" and lobe in region_str:
                    found_indices.append(idx)
    
    # Remove duplicates and sort
    found_indices = sorted(list(set(found_indices)))
    
    if not found_indices:
        return [0]  # N/A if no matches found
    
    # Show which regions were matched
    matched_regions = [brain_regions[idx] for idx in found_indices]
    print(f"      Matched regions: {matched_regions} -> indices: {found_indices}")
    
    return found_indices

def main():
    """Main conversion function with VQA format mappings"""
    
    # Detect dataset type from Excel file name or first case ID
    import sys
    
    excel_file = 'clinical-annotation.xlsx'
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    
    # Step 1: Debug Excel structure
    df = debug_excel_structure()
    
    # Detect dataset from first case ID
    first_case_id_col = df.columns[0]
    
    # Detect dataset type
    if 'MET' in first_case_id_col:
        dataset_type = 'MET'
        num_labels = 3
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue"
        ]
    elif 'GoAT' in first_case_id_col or 'GOAT' in first_case_id_col:
        dataset_type = 'GoAT'
        num_labels = 3
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue"
        ]
    else:  # GLI dataset
        dataset_type = 'GLI'
        num_labels = 4
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue", 
            "Resection Cavity"
        ]
    
    print(f"\n*** Detected {dataset_type} dataset - using {num_labels} labels ***\n")
    
    # Step 2: Extract first case (from column and rows 1-4)
    first_case_id, first_case_data = extract_first_case(df, num_labels)
    
    # Step 3: Find other cases
    other_cases = find_other_cases(df, num_labels)
    
    # Step 4: Convert all cases
    all_numerical_data = []
    
    # Convert first case
    print(f"\n{'='*60}")
    print(f"CONVERTING FIRST CASE: {first_case_id}")
    print(f"{'='*60}")
    
    try:
        numerical_case = convert_to_numerical_vqa_format(first_case_id, first_case_data, label_names)
        all_numerical_data.append(numerical_case)
        print(f"✓ Successfully processed {first_case_id}")
    except Exception as e:
        print(f"✗ ERROR processing {first_case_id}: {e}")
        raise e
    
    # Convert other cases
    for i, (case_id, case_data) in enumerate(other_cases):
        print(f"\n{'='*60}")
        print(f"CONVERTING CASE {i+2}/{len(other_cases)+1}: {case_id}")
        print(f"{'='*60}")
        
        try:
            numerical_case = convert_to_numerical_vqa_format(case_id, case_data, label_names)
            all_numerical_data.append(numerical_case)
            print(f"✓ Successfully processed {case_id}")
        except Exception as e:
            print(f"✗ ERROR processing {case_id}: {e}")
            raise e
    
    # Step 5: Save results
    volume_categories = ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"]
    shape_categories = ["N/A", "focus", "round", "oval", "elongated", "irregular"]
    satellite_categories = ["N/A", "single lesion", "core with satellite lesions", "scattered lesions"]
    brain_regions = ["n/a", "frontal", "parietal", "occipital", "temporal", "limbic", "insula", "subcortical", "cerebellum", "brainstem"]
    
    output = {
        "metadata": {
            "description": "Clinical annotations converted to VQA numerical format (0-based indexing)",
            "dataset_type": dataset_type,
            "num_labels": num_labels,
            "label_names": label_names,
            "volume_categories": volume_categories,
            "volume_mapping": {str(i): cat for i, cat in enumerate(volume_categories)},
            "shape_categories": shape_categories,
            "shape_mapping": {str(i): cat for i, cat in enumerate(shape_categories)},
            "satellite_categories": satellite_categories,
            "satellite_mapping": {str(i): cat for i, cat in enumerate(satellite_categories)},
            "brain_regions": brain_regions,
            "location_encoding": "sorted list of region indices (0=N/A, 1=frontal, etc.)"
        },
        "clinical_annotations": all_numerical_data
    }
    
    output_file = f'clinical_annotations_{dataset_type.lower()}_vqa_format.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"Processed {len(all_numerical_data)} cases")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")
    
    # Show summary of first case
    if all_numerical_data:
        print(f"\nFirst case summary ({all_numerical_data[0]['case_id']}):")
        print(json.dumps(all_numerical_data[0]['clinical_annotations'], indent=2))

if __name__ == "__main__":
    main()
