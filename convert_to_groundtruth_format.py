#!/usr/bin/env python3
"""
Convert clinical annotations to match the groundtruth JSON format.
Changes:
- Replace 'volume' with 'area'
- Replace 'location' with 'region' 
- Add 'mpMRI' field instead of 'id' and 'seg_file'
- Maintain the same structure as the groundtruth file
"""

import json

def convert_to_groundtruth_format(input_file, output_file):
    """Convert clinical annotations to groundtruth format"""
    
    # Load the current clinical annotations
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract the clinical annotations (skip metadata)
    clinical_annotations = data['clinical_annotations']
    
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
    print(f"Output saved to: {output_file}")
    
    # Show a sample entry
    if groundtruth_format:
        print("\nSample converted entry:")
        print(json.dumps(groundtruth_format[0], indent=2))

if __name__ == "__main__":
    input_file = "clinical_annotations_vqa_format.json"
    output_file = "clinical_annotations_groundtruth_format.json"
    
    convert_to_groundtruth_format(input_file, output_file)
