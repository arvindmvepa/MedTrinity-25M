#!/usr/bin/env python3
"""
Conversion script for clinical annotations with TRUE/FALSE location format
"""
import pandas as pd
import json
import argparse

def debug_excel_structure(excel_file):
    """Debug the Excel file structure thoroughly"""
    print("=== DEBUGGING EXCEL STRUCTURE ===")
    
    df = pd.read_excel(excel_file)
    print(f"Excel shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n=== FIRST 20 ROWS (RAW DATA) ===")
    for i in range(min(20, len(df))):
        row_values = []
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if pd.isna(val):
                row_values.append("NaN")
            else:
                row_values.append(f"'{val}'")
        print(f"Row {i:2d}: {' | '.join(row_values)}")
    
    return df

def extract_case_data(df, num_labels=4):
    """Extract case data from the Excel structure"""
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
    
    print(f"\n=== EXTRACTING CASE: {case_id} ====")
    
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
    
    brain_regions = ["n/a", "frontal", "parietal", "occipital", "temporal", "limbic", "insula", "subcortical", "cerebellum", "brainstem"]
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
            if (volume_answer is None or 
                str(volume_answer).lower().strip() == 'n/a' or 
                str(volume_answer).lower().strip() == 'nan' or
                str(volume_answer).lower().strip() == 'true' or
                str(volume_answer).lower().strip() == 'false'):
                label_data['volume'] = 0  # N/A
                print(f"    Volume: '{volume_answer}' -> 0 (N/A)")
            else:
                volume_clean = str(volume_answer).lower().strip()
                if volume_clean in volume_mapping:
                    label_data['volume'] = volume_mapping[volume_clean]
                    print(f"    Volume: '{volume_answer}' -> {label_data['volume']} ({volume_categories[label_data['volume']]})") 
                else:
                    # If it's not a recognized volume category, treat as N/A
                    print(f"    WARNING: Unknown volume value '{volume_answer}', treating as N/A")
                    label_data['volume'] = 0  # N/A
        else:
            label_data['volume'] = 0  # N/A
            print("    Volume: Missing -> 0 (N/A)")        # Location (multilabel using VQA lobe indices)
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
        
        numerical_case['clinical_annotations'][label_name] = label_data
    
    return numerical_case

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

def main():
    """Main conversion function with VQA format mappings"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert clinical annotations to VQA format (v2)')
    parser.add_argument('dataset_type', choices=['gli', 'met', 'goat'], 
                       help='Dataset type: gli, met, or goat')
    
    args = parser.parse_args()
    dataset_type = args.dataset_type.upper()
    
    # Construct Excel filename
    excel_file = f'clinical-annotation_{args.dataset_type}_mike.xlsx'
    
    print(f"Loading Excel file: {excel_file}")
    
    # Step 1: Debug Excel structure
    df = debug_excel_structure(excel_file)
    
    # Set dataset configuration based on argument
    if dataset_type == 'met':
        num_labels = 3
        label_names = [
            "Non-Enhancing Tumor",
            "Surrounding Non-enhancing FLAIR hyperintensity",
            "Enhancing Tissue"
        ]
    elif dataset_type == 'goat':
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
    
    print(f"\n*** Processing {dataset_type} dataset - using {num_labels} labels ***\n")
    
    # Step 2: Extract case data
    case_id, case_data = extract_case_data(df, num_labels)
    
    # Step 3: Convert case
    print(f"\n{'='*60}")
    print(f"CONVERTING CASE: {case_id}")
    print(f"{'='*60}")
    
    try:
        numerical_case = convert_to_numerical_vqa_format(case_id, case_data, label_names)
        print(f"✓ Successfully processed {case_id}")
    except Exception as e:
        print(f"✗ ERROR processing {case_id}: {e}")
        raise e
    
    # Step 4: Save results
    volume_categories = ["N/A", "<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%"]
    shape_categories = ["N/A", "focus", "round", "oval", "elongated", "irregular"]
    satellite_categories = ["N/A", "single lesion", "core with satellite lesions", "scattered lesions"]
    brain_regions = ["n/a", "frontal", "parietal", "occipital", "temporal", "limbic", "insula", "subcortical", "cerebellum", "brainstem"]
    
    output = {
        "metadata": {
            "description": "Clinical annotations converted to VQA numerical format (0-based indexing) - v2",
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
        "clinical_annotations": [numerical_case]
    }
    
    output_file = f'clinical_annotations_{dataset_type.lower()}_vqa_format_mike.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"Processed 1 case: {case_id}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")
    
    # Show case summary
    print(f"\nCase summary ({numerical_case['case_id']}):")
    print(json.dumps(numerical_case['clinical_annotations'], indent=2))

if __name__ == "__main__":
    main()