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

# Evaluate the Majority Classifier ---

# Structure to hold correct counts and total counts for metric calculation
evaluation_counts = collections.defaultdict(lambda: {
    'area': {'correct': 0, 'total': 0},
    'shape': {'correct': 0, 'total': 0},
    'satellite': {'correct': 0, 'total': 0}
})

# Iterate through data again for evaluation
for entry in data:
    for label_type, attributes in entry.get("labels", {}).items():
        # Ensure this label type was seen during training (i.e., in majority_classifiers)
        if label_type in majority_classifiers:
            # Get the predictions from the majority classifier
            predictions = majority_classifiers[label_type]

            # Evaluate 'area'
            if 'area' in attributes:
                actual_area = attributes['area']
                predicted_area = predictions.get('area') # Get predicted majority value
                evaluation_counts[label_type]['area']['total'] += 1
                if actual_area == predicted_area:
                    evaluation_counts[label_type]['area']['correct'] += 1

            # Evaluate 'shape'
            if 'shape' in attributes:
                actual_shape = attributes['shape']
                predicted_shape = predictions.get('shape') # Get predicted majority value
                evaluation_counts[label_type]['shape']['total'] += 1
                if actual_shape == predicted_shape:
                    evaluation_counts[label_type]['shape']['correct'] += 1

            # Evaluate 'satellite'
            if 'satellite' in attributes:
                actual_satellite = attributes['satellite']
                predicted_satellite = predictions.get('satellite') # Get predicted majority value
                evaluation_counts[label_type]['satellite']['total'] += 1
                if actual_satellite == predicted_satellite:
                    evaluation_counts[label_type]['satellite']['correct'] += 1

# Calculate Metrics (Accuracy) ---

metrics = {}

for label_type, counts in evaluation_counts.items():
    metrics[label_type] = {}
    for attribute in ['area', 'shape', 'satellite']:
        correct = counts[attribute]['correct']
        total = counts[attribute]['total']
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = None # Or 0, or 'N/A' depending on preference

        metrics[label_type][attribute] = {'accuracy': accuracy}
        # Store counts for context if desired
        # metrics[label_type][attribute]['correct'] = correct
        # metrics[label_type][attribute]['total'] = total

overall_metrics = {}
for attribute in ['area', 'shape', 'satellite']:
    for label_type in metrics.keys():
        if attribute not in overall_metrics:
            overall_metrics[attribute] = []
        overall_metrics[attribute] += [metrics[label_type][attribute]['accuracy']]
    overall_metrics[attribute] = sum(overall_metrics[attribute]) / len(overall_metrics[attribute])


print("\n--- Majority Classifier Metrics (Accuracy) ---")
print(json.dumps(metrics, indent=4))
print("\n--- Overall Metrics ---")
print(json.dumps(overall_metrics, indent=4))

