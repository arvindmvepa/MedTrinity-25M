import collections
import json
import itertools
import statistics # Import the statistics module for median functions

# --- Configuration ---
evaluation_file = "brats_gli_3d_vqa_subjTrue_test_aux_updated_v2_seed0.json"

# --- Load Data ---
try:
    with open(evaluation_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Evaluation file not found at {evaluation_file}")
    # Create dummy data for demonstration if file is missing
    data = [
        {"id": 0, "labels": {"Non-Enhancing Tumor": {"area": 2, "shape": 4, "satellite": 2, "region": [1, 2]}, "Enhancing Tissue": {"area": 2, "shape": 6, "satellite": 3, "region": [1]}}},
        {"id": 1, "labels": {"Non-Enhancing Tumor": {"area": 3, "shape": 4, "satellite": 2, "region": [1]}, "Enhancing Tissue": {"area": 1, "shape": 6, "satellite": 3, "region": [1, 3]}}},
        {"id": 2, "labels": {"Non-Enhancing Tumor": {"area": 2, "shape": 5, "satellite": 2, "region": [1, 2]}, "Enhancing Tissue": {"area": 2, "shape": 5, "satellite": 4, "region": []}}},
        {"id": 3, "labels": {"Non-Enhancing Tumor": {"area": 1, "shape": 4, "satellite": 1, "region": []}}},
        # Add an entry to test even number of areas for median calculation
        {"id": 4, "labels": {"Non-Enhancing Tumor": {"area": 4, "shape": 4, "satellite": 1, "region": [2]}}}
    ]
    print("Warning: Using dummy data as file was not found.")


# --- Helper Function: Calculate IoU ---
def calculate_iou(list1, list2):
    """Calculates Intersection over Union (IoU) for two lists representing regions."""
    set1 = set(list1) if list1 is not None else set()
    set2 = set(list2) if list2 is not None else set()
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if not set1 and not set2: return 1.0
    if not union: return 1.0
    if not set1 or not set2: return 0.0
    return intersection / union

# --- Step 1: Collect Data and Determine Best Predictors ---

attribute_collections = collections.defaultdict(lambda: {
    'area': [],
    'shape': collections.Counter(),
    'satellite': collections.Counter(),
    'region': []
})

for entry in data:
    for label_type, attributes in entry.get("labels", {}).items():
        if 'area' in attributes:
            attribute_collections[label_type]['area'].append(attributes['area'])
        if 'shape' in attributes:
            attribute_collections[label_type]['shape'][attributes['shape']] += 1
        if 'satellite' in attributes:
            attribute_collections[label_type]['satellite'][attributes['satellite']] += 1
        if 'region' in attributes:
             region_list = attributes['region'] if isinstance(attributes['region'], list) else []
             attribute_collections[label_type]['region'].append(region_list)
        elif 'region' not in attributes and label_type in attribute_collections:
             attribute_collections[label_type]['region'].append([])


# --- Determine Best Predictors ---
baseline_predictors = {}

for label_type, collections_dict in attribute_collections.items():
    baseline_predictors[label_type] = {}

    # --- Area: Use the Median (specifically median_low) as the baseline predictor for MAE ---
    area_values = collections_dict['area']
    if area_values:
        # median_low returns the lower of the two middle elements for even counts
        baseline_predictors[label_type]['area'] = statistics.median_low(area_values)
    else:
        baseline_predictors[label_type]['area'] = None # Or 0? Defaulting to None if no data

    # Shape: Majority class
    if collections_dict['shape']:
        baseline_predictors[label_type]['shape'] = collections_dict['shape'].most_common(1)[0][0]
    else:
        baseline_predictors[label_type]['shape'] = None

    # Satellite: Majority class
    if collections_dict['satellite']:
        baseline_predictors[label_type]['satellite'] = collections_dict['satellite'].most_common(1)[0][0]
    else:
        baseline_predictors[label_type]['satellite'] = None

    # Region: Find candidate list maximizing average IoU
    ground_truth_regions = collections_dict['region']
    if not ground_truth_regions:
         baseline_predictors[label_type]['region'] = []
         continue

    unique_region_tuples = set(tuple(sorted(lst)) for lst in ground_truth_regions if lst is not None)
    candidate_regions = [list(t) for t in unique_region_tuples]
    if any(not lst for lst in ground_truth_regions):
        if [] not in candidate_regions:
             candidate_regions.append([])

    best_candidate = []
    max_avg_iou = -1.0

    if not candidate_regions:
         baseline_predictors[label_type]['region'] = []
         continue

    for candidate in candidate_regions:
        current_sum_iou = 0.0
        # Ensure ground_truth_regions is not empty before division
        if not ground_truth_regions: continue # Skip if no GT regions for this label type
        for gt_region in ground_truth_regions:
            current_sum_iou += calculate_iou(candidate, gt_region)

        average_iou = current_sum_iou / len(ground_truth_regions)

        if average_iou > max_avg_iou:
            max_avg_iou = average_iou
            best_candidate = candidate

    baseline_predictors[label_type]['region'] = best_candidate


print("--- Baseline Predictors (Area uses Median) ---")
print(json.dumps(baseline_predictors, indent=4))
print("-" * 30)


# --- Step 2: Evaluate the Baseline Predictors ---

evaluation_sums = collections.defaultdict(lambda: {
    'area': {'error_sum': 0.0, 'total': 0},
    'shape': {'correct': 0, 'total': 0},
    'satellite': {'correct': 0, 'total': 0},
    'region': {'iou_sum': 0.0, 'total': 0}
})

# Iterate through data again for evaluation
for entry in data:
    for label_type, attributes in entry.get("labels", {}).items():
        if label_type in baseline_predictors:
            predictions = baseline_predictors[label_type]

            # Evaluate 'area' (MAE)
            if 'area' in attributes and predictions.get('area') is not None:
                actual_area = attributes['area']
                predicted_area = predictions['area'] # Median area
                evaluation_sums[label_type]['area']['total'] += 1
                evaluation_sums[label_type]['area']['error_sum'] += abs(actual_area - predicted_area)

            # Evaluate 'shape' (Accuracy)
            if 'shape' in attributes and predictions.get('shape') is not None:
                actual_shape = attributes['shape']
                predicted_shape = predictions['shape'] # Majority shape
                evaluation_sums[label_type]['shape']['total'] += 1
                if actual_shape == predicted_shape:
                    evaluation_sums[label_type]['shape']['correct'] += 1

            # Evaluate 'satellite' (Accuracy)
            if 'satellite' in attributes and predictions.get('satellite') is not None:
                actual_satellite = attributes['satellite']
                predicted_satellite = predictions['satellite'] # Majority satellite
                evaluation_sums[label_type]['satellite']['total'] += 1
                if actual_satellite == predicted_satellite:
                    evaluation_sums[label_type]['satellite']['correct'] += 1

            # Evaluate 'region' (Average IoU)
            actual_region = attributes.get('region', []) # Default to empty list
            if actual_region is None: actual_region = [] # Ensure list type

            predicted_region = predictions.get('region') # Best region list
            # predicted_region should already be a list (or empty list)

            evaluation_sums[label_type]['region']['total'] += 1
            iou = calculate_iou(actual_region, predicted_region)
            evaluation_sums[label_type]['region']['iou_sum'] += iou


# --- Step 3: Calculate Final Metrics ---

metrics = {}
attributes_to_process = ['area', 'shape', 'satellite', 'region']

for label_type, sums in evaluation_sums.items():
    metrics[label_type] = {}
    for attribute in attributes_to_process:
        total = sums[attribute]['total']
        metric_value = None
        metric_name = None

        if attribute == 'area': metric_name = 'mae'
        elif attribute in ['shape', 'satellite']: metric_name = 'accuracy'
        elif attribute == 'region': metric_name = 'avg_iou'

        if total > 0:
            if attribute == 'area':
                metric_value = sums[attribute]['error_sum'] / total
            elif attribute in ['shape', 'satellite']:
                metric_value = sums[attribute]['correct'] / total
            elif attribute == 'region':
                metric_value = sums[attribute]['iou_sum'] / total

        metrics[label_type][attribute] = {metric_name: metric_value}


# Calculate Overall Metrics
overall_metrics = {}
for attribute in attributes_to_process:
    all_values = []
    metric_name = None

    for label_type in metrics.keys():
        if attribute in metrics[label_type]:
             current_metric_dict = metrics[label_type][attribute]
             if not metric_name:
                 # Make sure dict is not empty before getting keys
                 if current_metric_dict:
                     metric_name = list(current_metric_dict.keys())[0]
                 else:
                     # Handle cases where the dict might be empty, though unlikely here
                     metric_name = 'unknown_metric' # Placeholder

             value = current_metric_dict.get(metric_name)
             if value is not None:
                 all_values.append(value)

    # Ensure metric_name was found
    if not metric_name:
         if attribute == 'area': metric_name = 'mae'
         elif attribute in ['shape', 'satellite']: metric_name = 'accuracy'
         elif attribute == 'region': metric_name = 'avg_iou'
         else: metric_name = 'unknown_metric'

    if all_values:
        overall_metrics[attribute] = {metric_name: sum(all_values) / len(all_values)}
    else:
        overall_metrics[attribute] = {metric_name: None}


print("\n--- Baseline Metrics (Area uses Median) ---")
print(json.dumps(metrics, indent=4))
print("\n--- Overall Metrics ---")
print(json.dumps(overall_metrics, indent=4))
