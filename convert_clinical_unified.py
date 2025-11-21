#!/usr/bin/env python3
"""
Unified clinical annotation converter with multiple processing modes:
1. Clinical VQA format (original) - for multi-case Excel files
2. Clinical VQA format (v2) - for single-case Excel files with TRUE/FALSE location format
3. Convert to groundtruth format - convert VQA JSON to groundtruth JSON format

The script automatically suggests the best processing mode based on Excel file structure.
"""
import pandas as pd
import json
import argparse
import os

def analyze_excel_structure(excel_file):
    """Analyze Excel file structure and suggest processing mode"""
    print("=== ANALYZING EXCEL STRUCTURE ===")
    
    df = pd.read_excel(excel_file)
    print(f"Excel shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for indicators of v1 vs v2 format
    has_case_in_column = any('BraTS' in str(col) for col in df.columns)
    has_case_in_cells = False
    has_true_false_locations = False
    
    # Look for case IDs in cells and TRUE/FALSE patterns
    for i in range(min(20, len(df))):
        for j in range(min(5, len(df.columns))):
            val = df.iloc[i, j]
            if isinstance(val, str) and 'BraTS' in val:
                has_case_in_cells = True
            if val is True or val is False:
                has_true_false_locations = True
    
    print(f"Has case ID in column names: {has_case_in_column}")
    print(f"Has case ID in cells: {has_case_in_cells}")
    print(f"Has TRUE/FALSE values: {has_true_false_locations}")
    
    # Suggest processing mode
    if has_case_in_column and not has_true_false_locations:
        suggested_mode = "v1"
        reason = "Multi-case format with case IDs in column headers"
    elif has_true_false_locations or (not has_case_in_column and has_case_in_cells):
        suggested_mode = "v2"
        reason = "Single-case format with TRUE/FALSE location data"
    else:
        suggested_mode = "v1"
        reason = "Default to multi-case format"
    
    print(f"\nSUGGESTED MODE: {suggested_mode} ({reason})")
    return df, suggested_mode

# ===== V1 FUNCTIONS (Multi-case format) =====

def extract_first_case_v1(df, num_labels=4):
    """Extract the first case (from column name and rows 1-4) - V1 format"""
    # The first case ID is in the column name
    first_case_id = df.columns[0]  # 'BraTS-GLI-00063-101' or 'BraTS-MET-...'
    
    print(f"\n=== EXTRACTING FIRST CASE (V1): {first_case_id} ===")
    
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

def find_other_cases_v1(df, num_labels=4):
    """Find all other cases that start with explicit case IDs - V1 format"""
    print("\n=== FINDING OTHER CASES (V1) ===")
    
    other_cases = []
    
    for i in range(len(df)):
        val = df.iloc[i, 0]
        if isinstance(val, str) and (val.startswith('BraTS-GLI-') or val.startswith('BraTS-MET-') or val.startswith('BraTS-GoAT-')):
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

# ===== V2 FUNCTIONS (Single-case format) =====

def extract_case_data_v2(df, num_labels=4):
    """Extract case data from the Excel structure - V2 format"""
    # The case ID is in cell A1, but might be NaN, so check multiple locations
    case_id = df.iloc[0, 0]
    if pd.isna(case_id):
        # Try to find case ID in other locations
        for row_idx in range(min(5, len(df))):
            for col_idx in range(min(5, len(df.columns))):
                val = df.iloc[row_idx, col_idx]
                if isinstance(val, str) and (val.startswith('BraTS-') or 'BraTS' in val):
                    case_id = val
                    break
            if not pd.isna(case_id) and case_id != df.iloc[0, 0]:
                break
    
    # If still NaN, use a placeholder
    if pd.isna(case_id):
        case_id = "Unknown_Case"
    
    print(f"\n=== EXTRACTING CASE (V2): {case_id} ===")
    
    # Label abbreviations are in row 2 (index 1)
    label_abbrevs = []
    for col_idx in range(1, num_labels + 1):
        if col_idx < len(df.columns):
            abbrev = df.iloc[1, col_idx]
            label_abbrevs.append(str(abbrev) if not pd.isna(abbrev) else None)
        else:
            label_abbrevs.append(None)
    
    print(f"Label abbreviations: {label_abbrevs}")
    
    # Find the question rows
    case_data = {}
    
    # Find volume row by searching for 'volume' in first column
    volume_row_idx = None
    for row_idx in range(len(df)):
        row_label = df.iloc[row_idx, 0]
        if isinstance(row_label, str) and row_label.lower().strip() == 'volume':
            volume_row_idx = row_idx
            break
    
    volume_answers = []
    if volume_row_idx is not None:
        print(f"Found volume row at index {volume_row_idx}")
        for col_idx in range(1, num_labels + 1):
            if col_idx < len(df.columns):
                answer = df.iloc[volume_row_idx, col_idx]
                if pd.isna(answer):
                    volume_answers.append(None)
                elif isinstance(answer, bool):
                    # Handle True/False as N/A
                    volume_answers.append("N/A")
                else:
                    volume_answers.append(str(answer).strip().strip('"'))
            else:
                volume_answers.append(None)
    else:
        print("Volume row not found, using N/A for all labels")
        volume_answers = ["N/A"] * num_labels
    
    case_data['volume'] = volume_answers
    print(f"Volume answers: {volume_answers}")
    
    # Location data - find brain region rows (rows with brain region names)
    brain_regions = ["frontal", "parietal", "occipital", "temporal", "limbic", "insula", "subcortical", "cerebellum", "brainstem"]
    location_data = {region: [] for region in brain_regions}
    
    # Scan through rows to find brain regions
    for row_idx in range(len(df)):
        row_label = df.iloc[row_idx, 0]
        if isinstance(row_label, str):
            row_label_clean = row_label.strip().strip('"').lower()
            if row_label_clean in brain_regions:
                print(f"Found brain region '{row_label_clean}' at row {row_idx}")
                
                # Extract TRUE/FALSE values for each label
                region_values = []
                for col_idx in range(1, num_labels + 1):
                    if col_idx < len(df.columns):
                        val = df.iloc[row_idx, col_idx]
                        # Check if it's TRUE or has any truthy value
                        is_present = (val is True or 
                                    (isinstance(val, str) and val.upper() == 'TRUE') or
                                    val == 1)
                        region_values.append(is_present)
                    else:
                        region_values.append(False)
                
                location_data[row_label_clean] = region_values
                print(f"  {row_label_clean}: {region_values}")
    
    # Convert location data to per-label format
    location_answers = []
    for label_idx in range(num_labels):
        present_regions = []
        for region, values in location_data.items():
            if label_idx < len(values) and values[label_idx]:
                present_regions.append(region)
        
        if present_regions:
            location_answers.append(", ".join(present_regions))
        else:
            location_answers.append("N/A")
    
    case_data['location'] = location_answers
    print(f"Location answers: {location_answers}")
    
    # Find shape row
    shape_answers = []
    shape_row_idx = None
    for row_idx in range(len(df)):
        row_label = df.iloc[row_idx, 0]
        if isinstance(row_label, str) and row_label.lower().strip() == 'shape':
            shape_row_idx = row_idx
            break
    
    if shape_row_idx is not None:
        print(f"Found shape row at index {shape_row_idx}")
        for col_idx in range(1, num_labels + 1):
            if col_idx < len(df.columns):
                answer = df.iloc[shape_row_idx, col_idx]
                shape_answers.append(str(answer).strip().strip('"') if not pd.isna(answer) else None)
            else:
                shape_answers.append(None)
    else:
        shape_answers = [None] * num_labels
    
    case_data['shape'] = shape_answers
    print(f"Shape answers: {shape_answers}")
    
    # Find spread out row
    spread_answers = []
    spread_row_idx = None
    for row_idx in range(len(df)):
        row_label = df.iloc[row_idx, 0]
        if isinstance(row_label, str) and 'spread' in row_label.lower():
            spread_row_idx = row_idx
            break
    
    if spread_row_idx is not None:
        print(f"Found spread out row at index {spread_row_idx}")
        for col_idx in range(1, num_labels + 1):
            if col_idx < len(df.columns):
                answer = df.iloc[spread_row_idx, col_idx]
                spread_answers.append(str(answer).strip().strip('"') if not pd.isna(answer) else None)
            else:
                spread_answers.append(None)
    else:
        spread_answers = [None] * num_labels
    
    case_data['spread out'] = spread_answers
    print(f"Spread out answers: {spread_answers}")
    
    return case_id, case_data

# ===== SHARED FUNCTIONS =====

def convert_to_groundtruth_format(case_id, case_data, label_names):
    """Convert case data directly to groundtruth format"""
    print(f"\n=== CONVERTING {case_id} TO GROUNDTRUTH FORMAT ===")
    
    # VQA system mappings (0-based indexing)
    volume_categories = ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"]
    volume_mapping = {cat.lower(): i for i, cat in enumerate(volume_categories)}
    
    shape_categories = ["N/A", "focus", "round", "oval", "elongated", "irregular"]
    shape_mapping = {cat.lower(): i for i, cat in enumerate(shape_categories)}
    
    satellite_categories = ["N/A", "single lesion", "core with satellite lesions", "scattered lesions"]
    satellite_mapping = {cat.lower(): i for i, cat in enumerate(satellite_categories)}
    # Add aliases for satellite mapping
    satellite_mapping["scattered"] = satellite_mapping["scattered lesions"]  # Handle "scattered" as "scattered lesions"
    
    brain_regions = ["n/a", "frontal", "parietal", "occipital", "temporal", "limbic", "insula", "subcortical", "cerebellum", "brainstem"]
    lobe_mapping = {region.lower(): i for i, region in enumerate(brain_regions)}
    
    # Create the groundtruth entry
    gt_entry = {
        "mpMRI": case_id,  # Use case_id as mpMRI name
        "labels": {}
    }
    
    # Process each label
    for label_idx, label_name in enumerate(label_names):
        print(f"\n  Processing {label_name} (index {label_idx}):")
        
        label_data = {}
        
        # Volume -> Area
        if 'volume' in case_data and label_idx < len(case_data['volume']):
            volume_answer = case_data['volume'][label_idx]
            if (volume_answer is None or 
                str(volume_answer).lower().strip() == 'n/a' or 
                str(volume_answer).lower().strip() == 'nan' or
                str(volume_answer).lower().strip() == 'true' or
                str(volume_answer).lower().strip() == 'false'):
                label_data['area'] = 0  # N/A
                print(f"    Area: '{volume_answer}' -> 0 (N/A)")
            else:
                volume_clean = str(volume_answer).lower().strip()
                if volume_clean in volume_mapping:
                    label_data['area'] = volume_mapping[volume_clean]
                    print(f"    Area: '{volume_answer}' -> {label_data['area']} ({volume_categories[label_data['area']]})")
                else:
                    # If it's not a recognized volume category, treat as N/A
                    print(f"    WARNING: Unknown volume value '{volume_answer}', treating as N/A")
                    label_data['area'] = 0  # N/A
        else:
            label_data['area'] = 0  # N/A
            print("    Area: Missing -> 0 (N/A)")
        
        # Location -> Region (multilabel using VQA lobe indices)
        if 'location' in case_data and label_idx < len(case_data['location']):
            location_answer = case_data['location'][label_idx]
            location_indices = encode_location_vqa_format(location_answer, brain_regions, lobe_mapping)
            label_data['region'] = location_indices
            print(f"    Region: '{location_answer}' -> {location_indices}")
        else:
            label_data['region'] = [0]  # N/A
            print(f"    Region: Missing -> [0] (N/A)")
        
        # Shape
        if 'shape' in case_data and label_idx < len(case_data['shape']):
            shape_answer = case_data['shape'][label_idx]
            if (shape_answer is None or 
                str(shape_answer).lower().strip() == 'n/a' or 
                str(shape_answer).lower().strip() == 'nan' or
                str(shape_answer).lower().strip() == 'true' or
                str(shape_answer).lower().strip() == 'false'):
                label_data['shape'] = 0  # N/A
                print(f"    Shape: '{shape_answer}' -> 0 (N/A)")
            else:
                shape_clean = str(shape_answer).lower().strip()
                if shape_clean in shape_mapping:
                    label_data['shape'] = shape_mapping[shape_clean]
                    print(f"    Shape: '{shape_answer}' -> {label_data['shape']} ({shape_categories[label_data['shape']]})")
                else:
                    # If it's not a recognized shape category, treat as N/A
                    print(f"    WARNING: Unknown shape value '{shape_answer}', treating as N/A")
                    label_data['shape'] = 0  # N/A
        else:
            label_data['shape'] = 0  # N/A
            print("    Shape: Missing -> 0 (N/A)")
        
        # Spread pattern (satellite)
        if 'spread out' in case_data and label_idx < len(case_data['spread out']):
            spread_answer = case_data['spread out'][label_idx]
            if (spread_answer is None or 
                str(spread_answer).lower().strip() == 'n/a' or 
                str(spread_answer).lower().strip() == 'nan' or
                str(spread_answer).lower().strip() == 'true' or
                str(spread_answer).lower().strip() == 'false'):
                label_data['satellite'] = 0  # N/A
                print(f"    Satellite: '{spread_answer}' -> 0 (N/A)")
            else:
                spread_clean = str(spread_answer).lower().strip()
                if spread_clean in satellite_mapping:
                    label_data['satellite'] = satellite_mapping[spread_clean]
                    print(f"    Satellite: '{spread_answer}' -> {label_data['satellite']} ({satellite_categories[label_data['satellite']] if label_data['satellite'] < len(satellite_categories) else 'scattered lesions'})")
                else:
                    # If it's not a recognized satellite category, treat as N/A
                    print(f"    WARNING: Unknown satellite pattern '{spread_answer}', treating as N/A")
                    label_data['satellite'] = 0  # N/A
        else:
            label_data['satellite'] = 0  # N/A
            print("    Satellite: Missing -> 0 (N/A)")
        
        gt_entry["labels"][label_name] = label_data
    
    return gt_entry

def encode_location_vqa_format(location_str, brain_regions, lobe_mapping):
    """Encode location using VQA format (return sorted list of indices)"""
    if location_str is None or location_str.lower().strip() == 'n/a':
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

def convert_to_groundtruth_format(vqa_data, output_file):
    """Convert VQA clinical annotations to groundtruth format"""
    print(f"\n=== CONVERTING TO GROUNDTRUTH FORMAT ===")
    
    # Extract the clinical annotations (skip metadata)
    if isinstance(vqa_data, dict) and 'clinical_annotations' in vqa_data:
        clinical_annotations = vqa_data['clinical_annotations']
    else:
        clinical_annotations = vqa_data
    
    # Convert to groundtruth format
    groundtruth_format = []
    
    for annotation in clinical_annotations:
        case_id = annotation['case_id']
        
        # Create the groundtruth entry
        gt_entry = {
            "mpMRI": case_id,  # Use case_id as mpMRI name
            "labels": {}
        }
        
        # Convert each label type
        for label_type, label_data in annotation['clinical_annotations'].items():
            gt_entry["labels"][label_type] = {
                "area": label_data["volume"],  # volume -> area
                "region": label_data["location"],  # location -> region
                "shape": label_data["shape"],
                "satellite": label_data["satellite"]
            }
        
        groundtruth_format.append(gt_entry)
    
    # Save the converted data
    with open(output_file, 'w') as f:
        json.dump(groundtruth_format, f, indent=4)
    
    print(f"Converted {len(groundtruth_format)} clinical annotations to groundtruth format")
    print(f"Saved to: {output_file}")
    
    return groundtruth_format

def process_clinical_annotations(input_file, dataset_type, processing_mode='auto'):
    """Process clinical annotations and convert directly to groundtruth format"""
    print(f"Loading Excel file: {input_file}")
    
    # Analyze Excel structure and suggest processing mode
    df, suggested_mode = analyze_excel_structure(input_file)
    
    # Use suggested mode if auto is selected
    if processing_mode == 'auto':
        processing_mode = suggested_mode
        print(f"Using suggested processing mode: {processing_mode}")
    
    # Set dataset configuration
    dataset_type = dataset_type.upper()
    if dataset_type == 'MET':
        num_labels = 3
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue"
        ]
    elif dataset_type == 'GOAT':
        num_labels = 3
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue"
        ]
    else:  # GLI dataset
        num_labels = 4
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue", 
            "Resection Cavity"
        ]
    
    print(f"\n*** Processing {dataset_type} dataset - using {num_labels} labels ***")
    print(f"*** Using processing mode: {processing_mode} ***\n")
    
    groundtruth_data = []
    
    if processing_mode == 'v1':
        # V1 processing: Multi-case format
        
        # Extract first case (from column and rows 1-4)
        first_case_id, first_case_data = extract_first_case_v1(df, num_labels)
        
        # Find other cases
        other_cases = find_other_cases_v1(df, num_labels)
        
        # Convert first case
        print(f"\n{'='*60}")
        print(f"CONVERTING FIRST CASE: {first_case_id}")
        print(f"{'='*60}")
        
        gt_entry = convert_to_groundtruth_format(first_case_id, first_case_data, label_names)
        groundtruth_data.append(gt_entry)
        print(f"✓ Successfully processed {first_case_id}")
        
        # Convert other cases
        for i, (case_id, case_data) in enumerate(other_cases):
            print(f"\n{'='*60}")
            print(f"CONVERTING CASE {i+2}/{len(other_cases)+1}: {case_id}")
            print(f"{'='*60}")
            
            gt_entry = convert_to_groundtruth_format(case_id, case_data, label_names)
            groundtruth_data.append(gt_entry)
            print(f"✓ Successfully processed {case_id}")
    
    elif processing_mode == 'v2':
        # V2 processing: Single-case format
        
        # Extract single case data
        case_id, case_data = extract_case_data_v2(df, num_labels)
        
        # Convert case
        print(f"\n{'='*60}")
        print(f"CONVERTING CASE: {case_id}")
        print(f"{'='*60}")
        
        gt_entry = convert_to_groundtruth_format(case_id, case_data, label_names)
        groundtruth_data.append(gt_entry)
        print(f"✓ Successfully processed {case_id}")
    
    return groundtruth_data

def main():
    """Main function for direct Excel to groundtruth conversion"""
    parser = argparse.ArgumentParser(description='Convert Clinical Annotations from Excel to Groundtruth JSON Format')
    parser.add_argument('input_file', help='Input Excel file path')
    parser.add_argument('--dataset-type', choices=['gli', 'met', 'goat'], default='gli',
                       help='Dataset type (default: gli)')
    parser.add_argument('--processing-mode', choices=['auto', 'v1', 'v2'], default='auto',
                       help='Processing mode: auto (suggested), v1 (multi-case), v2 (single-case)')
    parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        # Remove .xlsx extension and add _groundtruth_format.json
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_groundtruth_format.json"
    
    print(f"=== CLINICAL ANNOTATION TO GROUNDTRUTH CONVERTER ===")
    
    # Process Excel file directly to groundtruth format
    groundtruth_data = process_clinical_annotations(args.input_file, args.dataset_type, args.processing_mode)
    
    # Save groundtruth output
    with open(output_file, 'w') as f:
        json.dump(groundtruth_data, f, indent=4)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"Processed {len(groundtruth_data)} cases")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")
    
    # Show sample output
    if groundtruth_data:
        print(f"\nSample output (first case):")
        print(json.dumps(groundtruth_data[0], indent=2))

if __name__ == "__main__":
    main()