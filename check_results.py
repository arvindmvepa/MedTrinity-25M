import json

with open('clinical_annotations_numerical_corrected.json', 'r') as f:
    data = json.load(f)

print(f"Total cases: {len(data['clinical_annotations'])}")
print("\nCase IDs:")
for i, case in enumerate(data['clinical_annotations']):
    print(f"  {i+1}. {case['case_id']}")

# Show example of location data to verify it's working
print(f"\nExample location data for {data['clinical_annotations'][0]['case_id']}:")
for label_name, label_data in data['clinical_annotations'][0]['clinical_annotations'].items():
    location = label_data['location']
    if any(x == 1 for x in location):  # Has some location data
        regions = data['metadata']['location_regions']
        active_regions = [regions[i] for i, val in enumerate(location) if val == 1]
        print(f"  {label_name}: {active_regions}")
