import collections
import json


evaluation_file = "brats_gli_3d_vqa_subjTrue_test_aux_updated_v2_seed0.json"
with open(evaluation_file, 'r') as f:
    data = json.load(f)

# Dictionary to store counts for each label type and attribute
label_counts = collections.defaultdict(lambda: {
    'area': collections.Counter(),
    'shape': collections.Counter(),
    'satellite': collections.Counter()
})

# Iterate through each entry in the data list
for entry in data:
    # Iterate through each label type (e.g., "Non-Enhancing Tumor") in the entry
    for label_type, attributes in entry.get("labels", {}).items():
        # Increment the count for the specific value of area, shape, and satellite
        if 'area' in attributes:
            label_counts[label_type]['area'][attributes['area']] += 1
        if 'shape' in attributes:
            label_counts[label_type]['shape'][attributes['shape']] += 1
        if 'satellite' in attributes:
            label_counts[label_type]['satellite'][attributes['satellite']] += 1

# Dictionary to store the majority class for each attribute per label type
majority_classifiers = {}

# Iterate through the collected counts
for label_type, counts in label_counts.items():
    majority_classifiers[label_type] = {}
    # Find the most common value (majority class) for each attribute
    # The `most_common(1)` method returns a list of tuples [(value, count)],
    # so we take the first element [0] and then its first item [0] which is the value.
    # We add a check in case a specific attribute had no counts (empty Counter)
    if counts['area']:
        majority_classifiers[label_type]['area'] = counts['area'].most_common(1)[0][0]
    else:
         majority_classifiers[label_type]['area'] = None # Or some other default

    if counts['shape']:
        majority_classifiers[label_type]['shape'] = counts['shape'].most_common(1)[0][0]
    else:
        majority_classifiers[label_type]['shape'] = None

    if counts['satellite']:
        majority_classifiers[label_type]['satellite'] = counts['satellite'].most_common(1)[0][0]
    else:
         majority_classifiers[label_type]['satellite'] = None

print(json.dumps(majority_classifiers, indent=4))

# Optional: Print the counts for verification
# print("\n--- Detailed Counts ---")
# print(json.dumps(label_counts, indent=4))
