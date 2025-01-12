import os.path
from glob import glob
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from collections import Counter, defaultdict
import json
import re
from create_brats_imaging_dataset import load_color_seg_png_as_labels
from vqa_utils import label_names, analyze_label_relationship, analyze_label_summary, get_seg_ids_empty_counts, extract_label_intensity_components


def generate_train_val_test_splits(all_vqa_questions, seed=0, train_frac=0.8, val_frac=0.1,
                                   train_file="brats_gli_vqa_train.json", val_file="brats_gli_vqa_val.json",
                                   test_file="brats_gli_vqa_test.json"):
    random_state = np.random.RandomState(seed)
    study_names = sorted(list({q["study_name"] for q in all_vqa_questions}))
    random_state.shuffle(study_names)
    total_studies = len(study_names)
    train_end = int(total_studies * train_frac)
    val_end = int(total_studies * (train_frac + val_frac))
    train_studies = study_names[:train_end]
    val_studies = study_names[train_end:val_end]
    test_studies = study_names[val_end:]
    train_questions = [q for q in all_vqa_questions if q["study_name"] in train_studies]
    val_questions = [q for q in all_vqa_questions if q["study_name"] in val_studies]
    test_questions = [q for q in all_vqa_questions if q["study_name"] in test_studies]
    print(f"Train studies: {len(train_studies)}, Val studies: {len(val_studies)}, Test studies: {len(test_studies)}")
    print(f"Train questions: {len(train_questions)}, Val questions: {len(val_questions)}, Test questions: {len(test_questions)}")

    with open(train_file, 'w') as f:
        json.dump(train_questions, f, indent=2)
    with open(val_file, 'w') as f:
        json.dump(val_questions, f, indent=2)
    with open(test_file, 'w') as f:
        json.dump(test_questions, f, indent=2)

    return train_questions, val_questions, test_questions

def postprocess_vqa_data(all_vqa_questions, max_num_of_seg_ids_per_empty_count=100,
                         default_modality="t1c", save_vqa_file="brats_gli_vqa_clean_data.json", seed=0):
    filtered_vqa_questions = filter_seg_ids_from_vqa_data(all_vqa_questions,
                                                          max_num_of_seg_ids_per_empty_count=max_num_of_seg_ids_per_empty_count,
                                                          seed=seed)
    for index in range(len(filtered_vqa_questions)):
        question = filtered_vqa_questions[index]
        base_dir = os.path.basename(os.path.dirname(question["seg_file"]))
        question["img_id"] = filtered_vqa_questions[index]["seg_id"]
        if "img_name" not in question:
            base_img_file = os.path.basename(question["seg_file"]).replace("seg", default_modality)
            question["img_name"] = os.path.join(base_dir, base_img_file)
        assert "question" in filtered_vqa_questions[index]
        assert "answer" in filtered_vqa_questions[index]
        question["q_lang"] = "en"
        question["qid"] = index
        question["location"] = "Brain"
        if "modality" not in question:
            question["modality"] = default_modality
        question["answer_type"] = "OPEN"
        question["base_type"] = "VQA"
        question["content_type"] = question["type"]
        question["qid"] = index
        question["study_name"] = "-".join(base_dir.split("-")[:-1])

    with open(save_vqa_file, 'w') as f:
        json.dump(filtered_vqa_questions, f, indent=2)

    return filtered_vqa_questions


def filter_seg_ids_from_vqa_data(all_vqa_questions, max_num_of_seg_ids_per_empty_count=100, seed=0):
    seg_ids_empty_counts_map = get_seg_ids_empty_counts(all_vqa_questions)
    random_state = np.random.RandomState(seed)

    # 1) Group seg_ids by their empties count
    count_to_segids = defaultdict(list)
    for seg_id, empties_val in seg_ids_empty_counts_map.items():
        count_to_segids[empties_val].append(seg_id)

    # 2) For each empties_val group, if it has more than max_segids_per_count seg_ids,
    #    we randomly sample up to that limit. Otherwise, keep all.
    final_seg_ids = []
    for empties_val, segids in count_to_segids.items():
        if len(segids) > max_num_of_seg_ids_per_empty_count:
            chosen = random_state.choice(segids, size=max_num_of_seg_ids_per_empty_count, replace=False)
        else:
            chosen = segids
        final_seg_ids.extend(chosen)
    return [q for q in all_vqa_questions if q["seg_id"] in final_seg_ids]


def summarize_vqa_data(all_vqa_questions,
                       max_seg_id_list=(10, 20, 25, 30, 40, 50, 70, 77, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)):
    """
    Summarize basic statistics about the generated VQA data:
      - Count total questions
      - Distribution of question types
      - Distribution of label names
      - Parse adjacency and area percentages for min/max/avg
      - Count occurrences where we have 'none' labels or 0% adjacency
    """
    total_questions = len(all_vqa_questions)
    question_type_counts = Counter(q.get("type", "unknown") for q in all_vqa_questions)
    label_name_counts = Counter(q.get("label_name", "unknown") for q in all_vqa_questions)

    # We'll try to parse numeric percentages for adjacency (adj_area) questions:
    adjacency_percentages = []
    # Keep track of how many adjacency questions are effectively 0% => "none" adjacency
    zero_adjacency_count = 0
    none_adj_quadrant_count = 0

    # We'll do the same for area questions:
    area_percentages = []
    # Keep track of how many times area=0% => "none" label
    zero_area_count = 0
    none_area_count = 0

    extent_percentages = []
    solidity_percentages = []

    # We'll also check how often bounding box = "none" or quadrant = "none"
    # in quadrant-related or bbox-related questions
    none_quadrant_count = 0
    none_bbox_count = 0
    none_extent_count = 0
    none_solidity_count = 0

    completely_empty_map = dict()
    empty_count_map = defaultdict(int)
    no_tumor_core_map = dict()

    for q in all_vqa_questions:
        q_type = q.get("type", "unknown")
        answer = q.get("answer", "")
        seg_id = q.get("seg_id", "unknown")
        label_name = q.get("label_name", "unknown")
        empty_count_map[seg_id]

        # Parse adjacency area questions
        if q_type == "adj_area":
            # Typical answer format: "25.0%, which is the majority"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                adj_val = float(match.group(1))
                adjacency_percentages.append(adj_val)
                if adj_val == 0.0:
                    zero_adjacency_count += 1
                    empty_count_map[seg_id] += 1

        # Parse adjacency quadrant questions
        if q_type == "adj_quadrants":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_adj_quadrant_count += 1
                empty_count_map[seg_id] += 1

        # Parse area questions
        if q_type == "area":
            # Typical answer format: "25.0%, which is a small portion"
            match = re.search(r'([\d.]+)%,', answer)
            area_val = None
            if match:
                area_val = float(match.group(1))
                area_percentages.append(area_val)
                if area_val == 0.0:
                    zero_area_count += 1
                    empty_count_map[seg_id] += 1
                    completely_empty_map[seg_id] = completely_empty_map[seg_id] and True if seg_id in completely_empty_map else True
                    if "tumor core" in label_name.lower():
                        no_tumor_core_map[seg_id] = True
                else:
                    completely_empty_map[seg_id] = False
            if "none" in answer.lower():
                none_area_count += 1
            if "none" not in answer.lower() and area_val == 0.0:
                print("Anomaly: ", answer)

        # Check bounding box questions
        if q_type == "bbox":
            # Typical answer format: "top-left, bottom-right" or "none"
            if "none" in answer.lower():
                none_bbox_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "quadrant":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_quadrant_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "extent":
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                extent_val = float(match.group(1))
                extent_percentages.append(extent_val)
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_extent_count += 1
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "solidity":
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                solidity_val = float(match.group(1))
                solidity_percentages.append(solidity_val)
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                none_solidity_count += 1
                empty_count_map[seg_id] += 1

    # Build the summary report lines
    lines = []
    lines.append("===== VQA DATA SUMMARY =====")
    lines.append(f"Total questions: {total_questions}")

    lines.append("\nQuestion type distribution:")
    for t, c in question_type_counts.items():
        lines.append(f"  - {t}: {c}")

    lines.append("\nLabel name distribution:")
    for lbl, c in label_name_counts.items():
        lines.append(f"  - {lbl}: {c}")

    # Summaries of adjacency values
    if adjacency_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=adjacency_percentages,
                                                zero_score_count=zero_adjacency_count,
                                                none_score_count=none_adj_quadrant_count, metric_name="adjacency_area"))

    # Summaries of area values
    if area_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=area_percentages, zero_score_count=zero_area_count,
                                                none_score_count=none_area_count, metric_name="area"))

    if extent_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=extent_percentages, zero_score_count=np.nan,
                                                none_score_count=none_extent_count, metric_name="extent"))

    if solidity_percentages:
        lines.append(get_descriptive_statistics(list_of_scores=solidity_percentages, zero_score_count=np.nan,
                                                none_score_count=none_solidity_count, metric_name="solidity"))

    # Summaries of "none" answers for quadrant and bounding box
    lines.append(f"\n# of questions that returned 'none' quadrant: {none_quadrant_count}")
    lines.append(f"# of questions that returned 'none' bounding box: {none_bbox_count}")
    lines.append(f"# of questions that returned 'none' extent: {none_extent_count}")
    lines.append(f"# of questions that returned 'none' solidity: {none_solidity_count}")

    lines.append(f"# of seg maps: {len(completely_empty_map)}, seg maps that are empty: {sum(list(completely_empty_map.values()))}, seg maps without tumor core: {sum(list(no_tumor_core_map.values()))}")

    if empty_count_map is not None:
        #for remove_empty_counts in [[], [33, 28], [33, 28, 26]]:
        #    for empty_count in remove_empty_counts:
        #        empty_count_map = {seg_id: count for seg_id, count in list(empty_count_map.items()) if count != empty_count}
        empty_count_map_ = empty_count_map.copy()
        for max_seg_id in max_seg_id_list:
            empty_count_tracker = dict()
            empty_count_map = dict()
            for seg_id, count in list(empty_count_map_.items()):
                if count not in empty_count_tracker:
                    empty_count_tracker[count] = 1
                    empty_count_map[seg_id] = count
                else:
                    if empty_count_tracker[count] < max_seg_id:
                        empty_count_tracker[count] += 1
                        empty_count_map[seg_id] = count
            empty_counts = list(empty_count_map.values())
            avg_empty_counts = sum(empty_counts) / len(empty_counts)
            q1_empty_counts = np.quantile(empty_counts, 0.25)
            q2_empty_counts = np.quantile(empty_counts, 0.5)
            q3_empty_counts = np.quantile(empty_counts, 0.75)
            min_empty_counts = min(empty_counts)
            max_empty_counts = max(empty_counts)
            empty_count_distribution = Counter(empty_counts)
            seg_id_counts = list(empty_count_distribution.values())
            avg_seg_id_counts = sum(seg_id_counts) / len(seg_id_counts)
            q1_seg_id_counts = np.quantile(seg_id_counts, 0.25)
            q2_seg_id_counts = np.quantile(seg_id_counts, 0.5)
            q3_seg_id_counts = np.quantile(seg_id_counts, 0.75)
            min_seg_id_counts = min(seg_id_counts)
            max_seg_id_counts = max(seg_id_counts)
            lines.append(
                f"\nEmpty counts per seg_id (over 4 image types with 1 modality question and 5 questions for 5 organs and 2 questions for 4 relationships. total=136):\n"
                #f"\nEmpty counts per seg_id (over 5 questions for 5 organs and 2 questions for 4 relationships. total=33):\n"
                #f"  Removing seg_ids with empty counts: " + ", ".join([str(k) for k in remove_empty_counts]) + "\n"
                f"  Max # of seg_ids: {max_seg_id}\n"
                f"  Seg_id Count: {len(empty_count_map)}\n"
                f"  Avg Empty Count:   {avg_empty_counts:.2f}\n"
                f"  Empty Count 25-50-75: [{q1_empty_counts:.2f}, {q2_empty_counts:.2f}, {q3_empty_counts:.2f}]\n"
                f"  Empty Count Range: [{min_empty_counts:.2f}, {max_empty_counts:.2f}]\n"
                f"  Empty Count Distribution: {[(k, empty_count_distribution[k]) for k in sorted(empty_count_distribution.keys())]}\n"
                f"  Avg Seg_id Count:   {avg_seg_id_counts:.2f}\n"
                f"  Seg_id 25-50-75: [{q1_seg_id_counts:.2f}, {q2_seg_id_counts:.2f}, {q3_seg_id_counts:.2f}]\n"
                f"  Seg_id Range: [{min_seg_id_counts:.2f}, {max_seg_id_counts:.2f}]\n"
            )
    return "\n".join(lines)


def get_descriptive_statistics(list_of_scores, zero_score_count, none_score_count, metric_name):
    lines = []
    avg_score = sum(list_of_scores) / len(list_of_scores)
    q1_score = np.quantile(list_of_scores, 0.25)
    q2_score = np.quantile(list_of_scores, 0.5)
    q3_score = np.quantile(list_of_scores, 0.75)
    min_score = min(list_of_scores)
    max_score = max(list_of_scores)

    # add non-zero scores
    non_zero_scores = [p for p in list_of_scores if p != 0.0]
    non_zero_avg_score = sum(non_zero_scores) / len(non_zero_scores)
    non_zero_q1_score = np.quantile(non_zero_scores, 0.25)
    non_zero_q2_score = np.quantile(non_zero_scores, 0.5)
    non_zero_q3_score = np.quantile(non_zero_scores, 0.75)
    non_zero_min_score = min(non_zero_scores)
    non_zero_max_score = max(non_zero_scores)

    return (f"\n{metric_name} questions:\n"
            f"  Count: {len(list_of_scores)}\n"
            f"  Avg:   {avg_score:.2f}%\n"
            f"  25-50-75: [{q1_score:.2f}, {q2_score:.2f}, {q3_score:.2f}]\n"
            f"  Range: [{min_score:.2f}%, {max_score:.2f}%]\n"
            f"  # with 0% {metric_name}: {zero_score_count}\n"
            f"  # with none {metric_name}: {none_score_count}\n"

            f"\n (non-zero) {metric_name} questions:\n"
            f"  Count: {len(non_zero_scores)}\n"
            f"  Avg:   {non_zero_avg_score:.2f}%\n"
            f"  25-50-75: [{non_zero_q1_score:.2f}, {non_zero_q2_score:.2f}, {non_zero_q3_score:.2f}]\n"
            f"  Range: [{non_zero_min_score:.2f}%, {non_zero_max_score:.2f}%]\n")


def generate_modality_question(modality):
    question = f"What is the modality of the brain image?"
    answer = modality
    question_dict = {"question": question, "answer": answer, "type": "modality", "label_name": "NA"}
    return [question_dict]


def generate_labal_vqa_questions(summ, include_area=True, include_quadrant=True, include_bbox=True, include_extent=True,
                                 include_solidity=True, subjective_only=False):
    vqa_questions = []
    if include_area:
        question = f"How large is the area covered by {summ['name']}?"
        if subjective_only:
            answer = f"{summ['area_interp']}"
        else:
            answer = f"{summ['area_pct']:.1f}%, which is {summ['area_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "area", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_quadrant:
        question = f"Which quadrant is {summ['name']} centered in?"
        answer = f"{summ['centroid_quadrant']}"
        question_dict = {"question": question, "answer": answer, "type": "quadrant", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_bbox:
        question = f"The smallest bounding box surrounding {summ['name']} is in which quadrants?"
        answer = f"{summ['bbox_str']}"
        question_dict = {"question": question, "answer": answer, "type": "bbox", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_extent:
        question = f"Within the smallest bounding box surrounding {summ['name']}, to what extent is the bounding box region filled?"
        if subjective_only:
            answer = f"{summ['extent_interp']}"
        else:
            answer = f"{summ['extent_value']:.1f}%, which is {summ['extent_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "extent", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    if include_solidity:
        question = f"Within the smallest bounding box surrounding {summ['name']}, how solid is the region?"
        if subjective_only:
            answer = f"{summ['solidity_interp']}"
        else:
            answer = f"{summ['solidity_value']:.1f}%, which is {summ['solidity_interp']}"
        question_dict = {"question": question, "answer": answer, "type": "solidity", "label_name": summ['name']}
        vqa_questions.append(question_dict)
    return vqa_questions


def generate_single_relationship_vqa_questions(label1_name, label2_name, mask1, mask2, total_pixels, height, width,
                                               subjective_only=False):
    relation_str = f"{label1_name}_vs_{label2_name}"
    relation_dict = analyze_label_relationship(mask1, mask2, total_pixels, height, width)

    vqa_questions = []
    question = f"How large is the area covered by {label1_name} adjacent to {label2_name}?"
    if subjective_only:
        answer = f"{relation_dict['adjacent_interpretation']}"
    else:
        answer = f"{relation_dict['adjacent_percentage']:.1f}%, which is {relation_dict['adjacent_interpretation']}"
    question_dict = {"question": question, "answer": answer, "type": "adj_area", "label_name": relation_str}
    vqa_questions.append(question_dict)
    question = f"For the region of {label1_name} which is adjacent to {label2_name}, what quadrant(s) is it in?"
    answer = f"{relation_dict['adjacent_quadrants']}"
    question_dict = {"question": question, "answer": answer, "type": "adj_quadrants", "label_name": relation_str}
    vqa_questions.append(question_dict)
    return vqa_questions


def generate_all_relationship_vqa_questions(seg_map_2d, height, width, total_pixels, image=None,
                                            abs_intensity_diff_thresh=10, include_nonenh_vs_enh=True,
                                            include_flair_vs_core=True, include_rec_vs_core=True,
                                            include_rec_vs_flair=True, subjective_only=False):
    label1_mask = (seg_map_2d == 1)  # Non-Enh
    label2_mask = (seg_map_2d == 2)  # FLAIR
    label3_mask = (seg_map_2d == 3)  # Enh
    label4_mask = (seg_map_2d == 4)  # Resection cavity
    label5_mask = np.logical_or(label1_mask, label3_mask)  # Tumor Core (Non-Enh + Enh)
    if image is not None:
        if image is not None:
            label1_mask, _, _, _ = extract_label_intensity_components(image=image, mask=label1_mask,
                                                                      abs_intensity_diff_thresh=abs_intensity_diff_thresh)
            label2_mask, _, _, _ = extract_label_intensity_components(image=image, mask=label2_mask,
                                                                      abs_intensity_diff_thresh=abs_intensity_diff_thresh)
            label3_mask, _, _, _ = extract_label_intensity_components(image=image, mask=label3_mask,
                                                                      abs_intensity_diff_thresh=abs_intensity_diff_thresh)
            label4_mask, _, _, _ = extract_label_intensity_components(image=image, mask=label4_mask,
                                                                      abs_intensity_diff_thresh=abs_intensity_diff_thresh)
            label5_mask, _, _, _ = extract_label_intensity_components(image=image, mask=label5_mask,
                                                                      abs_intensity_diff_thresh=abs_intensity_diff_thresh)

    vqa_questions = []
    if include_nonenh_vs_enh:
        question_dict = generate_single_relationship_vqa_questions(label_names.get(1), label_names.get(3), label1_mask,
                                                                   label3_mask, total_pixels, height, width,
                                                                   subjective_only=subjective_only)
        vqa_questions.extend(question_dict)
    if include_flair_vs_core:
        question_dict = generate_single_relationship_vqa_questions(label_names.get(2), label_names.get(5), label2_mask,
                                                                   label5_mask, total_pixels, height, width,
                                                                   subjective_only=subjective_only)
        vqa_questions.extend(question_dict)
    if include_rec_vs_core:
        question_dict = generate_single_relationship_vqa_questions(label_names.get(4), label_names.get(5), label4_mask,
                                                                   label5_mask, total_pixels, height, width,
                                                                   subjective_only=subjective_only)
        vqa_questions.extend(question_dict)
    if include_rec_vs_flair:
        question_dict = generate_single_relationship_vqa_questions(label_names.get(4), label_names.get(2), label4_mask,
                                                                   label2_mask, total_pixels, height, width,
                                                                   subjective_only=subjective_only)
        vqa_questions.extend(question_dict)
    return vqa_questions


def generate_vqa_from_seg_map_and_sequence(seg_file, seg_id, include_area=True, include_quadrant=False,
                                           include_bbox=True, include_extent=True, include_solidity=True,
                                           include_nonenh_vs_enh=True, include_flair_vs_core=True,
                                           include_rec_vs_core=True, include_rec_vs_flair=True, subjective_only=False,
                                           abs_intensity_diff_thresh=10):
    """
    Master function to produce a textual report combining:
      - Label summaries (area %, quadrant, bounding box, extent-based compactness)
        with subjective interpretations.
      - Non-Enh vs Enh tumor adjacency info.
      - FLAIR vs Tumor Core adjacency info.
      - Resection cavity vs tumor core & FLAIR.
    """
    seg_map_2d = load_color_seg_png_as_labels(seg_file)

    height, width = seg_map_2d.shape
    total_pixels = seg_map_2d.size

    all_vqa_questions = []
    # extract labels per mri sequence
    for modality in ["t1c", "t1n", "t2w", "t2f"]:
        img_file = seg_file.replace("seg", modality)
        image = np.array(Image.open(img_file))

        # Summaries of labels
        label_summaries = analyze_label_summary(seg_map_2d=seg_map_2d, image=image, height=height, width=width,
                                                total_pixels=total_pixels,
                                                abs_intensity_diff_thresh=abs_intensity_diff_thresh)
        vqa_questions = []
        modality_question = generate_modality_question(modality)
        vqa_questions.extend(modality_question)
        # get single label questions
        for summ in label_summaries:
            label_vqa_questions = generate_labal_vqa_questions(summ=summ, include_area=include_area,
                                                               include_quadrant=include_quadrant,
                                                               include_bbox=include_bbox,
                                                               include_extent=include_extent,
                                                               include_solidity=include_solidity,
                                                               subjective_only=subjective_only)
            vqa_questions.extend(label_vqa_questions)
        # get label relationship questions
        relationship_vqa_questions = generate_all_relationship_vqa_questions(seg_map_2d=seg_map_2d, height=height,
                                                                             width=width, total_pixels=total_pixels,
                                                                             include_nonenh_vs_enh=include_nonenh_vs_enh,
                                                                             include_flair_vs_core=include_flair_vs_core,
                                                                             include_rec_vs_core=include_rec_vs_core,
                                                                             include_rec_vs_flair=include_rec_vs_flair,
                                                                             subjective_only=subjective_only)
        vqa_questions.extend(relationship_vqa_questions)
        for q in vqa_questions:
            q['img_name'] = img_file
            q['modality'] = modality
            q["seg_id"] = seg_id
            q["seg_file"] = seg_file
        all_vqa_questions.extend(vqa_questions)
    return all_vqa_questions


def generate_vqa_from_seg_map(seg_file, seg_id, include_area=True, include_quadrant=True, include_bbox=True,
                              include_extent=True, include_solidity=True, include_nonenh_vs_enh=True,
                              include_flair_vs_core=True, include_rec_vs_core=True, include_rec_vs_flair=True,
                              subjective_only=False):
    """
    Master function to produce a textual report combining:
      - Label summaries (area %, quadrant, bounding box, extent-based compactness)
        with subjective interpretations.
      - Non-Enh vs Enh tumor adjacency info.
      - FLAIR vs Tumor Core adjacency info.
      - Resection cavity vs tumor core & FLAIR.
    """
    seg_map_2d = load_color_seg_png_as_labels(seg_file)
    height, width = seg_map_2d.shape
    total_pixels = seg_map_2d.size

    # Summaries of labels
    label_summaries = analyze_label_summary(seg_map_2d=seg_map_2d, height=height, width=width,
                                            total_pixels=total_pixels)
    vqa_questions = []
    # get single label questions
    for summ in label_summaries:
        label_vqa_questions = generate_labal_vqa_questions(summ=summ, include_area=include_area,
                                                           include_quadrant=include_quadrant, include_bbox=include_bbox,
                                                           include_extent=include_extent,
                                                           include_solidity=include_solidity,
                                                           subjective_only=subjective_only)
        vqa_questions.extend(label_vqa_questions)
    # get label relationship questions
    relationship_vqa_questions = generate_all_relationship_vqa_questions(seg_map_2d=seg_map_2d, height=height,
                                                                         width=width, total_pixels=total_pixels,
                                                                         include_nonenh_vs_enh=include_nonenh_vs_enh,
                                                                         include_flair_vs_core=include_flair_vs_core,
                                                                         include_rec_vs_core=include_rec_vs_core,
                                                                         include_rec_vs_flair=include_rec_vs_flair,
                                                                         subjective_only=subjective_only)
    vqa_questions.extend(relationship_vqa_questions)
    for q in vqa_questions:
        q["seg_id"] = seg_id
        q["seg_file"] = seg_file
    return vqa_questions


def generate_vqa_data_from_seg_file(seg_files, include_area=True, include_quadrant=True, include_bbox=True,
                                    include_extent=True, include_solidity=True, include_nonenh_vs_enh=True,
                                    include_flair_vs_core=True, include_rec_vs_core=True, include_rec_vs_flair=True):
    all_vqa_questions = []
    for seg_id, seg_file in tqdm(enumerate(seg_files)):
        vqa_data = generate_vqa_from_seg_map(seg_file=seg_file, seg_id=seg_id, include_area=include_area,
                                             include_quadrant=include_quadrant, include_bbox=include_bbox,
                                             include_extent=include_extent, include_solidity=include_solidity,
                                             include_nonenh_vs_enh=include_nonenh_vs_enh,
                                             include_flair_vs_core=include_flair_vs_core,
                                             include_rec_vs_core=include_rec_vs_core,
                                             include_rec_vs_flair=include_rec_vs_flair)
        all_vqa_questions.extend(vqa_data)
    return all_vqa_questions


def generate_vqa_data_from_seg_file_joblib(
    seg_files,
    n_jobs=-1,
    include_area=True,
    include_quadrant=True,
    include_bbox=True,
    include_extent=True,
    include_solidity=True,
    include_nonenh_vs_enh=True,
    include_flair_vs_core=True,
    include_rec_vs_core=True,
    include_rec_vs_flair=True,
    subjective_only=False,
    abs_intensity_diff_thresh=10
):
    """
    Parallelized version of generating VQA data from a list of seg_files,
    with a progress bar (tqdm_joblib).

    Parameters
    ----------
    seg_files : list of str
        Paths to segmentation files (PNG).
    n_jobs : int, default=-1
        Number of parallel jobs. -1 => use all cores.
    include_area, include_quadrant, include_bbox, ...
        Configuration flags passed down to generate_vqa_from_seg_map.
    """
    all_vqa_questions = []

    # Wrap Parallel execution with tqdm_joblib for the progress bar:
    with tqdm_joblib(desc="Processing segmentation files", total=len(seg_files)):
        results = Parallel(n_jobs=n_jobs)(
            #delayed(generate_vqa_from_seg_map)(
            delayed(generate_vqa_from_seg_map_and_sequence)(
                seg_file,
                seg_id,
                include_area,
                include_quadrant,
                include_bbox,
                include_extent,
                include_solidity,
                include_nonenh_vs_enh,
                include_flair_vs_core,
                include_rec_vs_core,
                include_rec_vs_flair,
                subjective_only,
                abs_intensity_diff_thresh
            )
            for seg_id, seg_file in enumerate(seg_files)
        )

    # Combine results
    for r in results:
        all_vqa_questions.extend(r)

    return all_vqa_questions


if __name__ == "__main__":
    """"
    # generate relevant sample output for some seg files
    seg_files = ['/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02358-101/BraTS-GLI-02358-101_seg_slice_100_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02103-103/BraTS-GLI-02103-103_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02111-105/BraTS-GLI-02111-105_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02198-104/BraTS-GLI-02198-104_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02209-101/BraTS-GLI-02209-101_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-03064-100/BraTS-GLI-03064-100_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02741-100/BraTS-GLI-02741-100_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02761-100/BraTS-GLI-02761-100_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02882-101/BraTS-GLI-02882-101_seg_slice_120_y.png',
                 '/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02894-101/BraTS-GLI-02894-101_seg_slice_120_y.png']
    for seg_file in seg_files:
        seg_map_2d = load_color_seg_png_as_labels(seg_file)
        report = analyze_segmentation_map(seg_map_2d)
        print(report)
    """
    subjective_only = True
    vqa_file = f"brats_gli_vqa_subj{subjective_only}_data_v2_1.json"
    clean_vqa_file = f"brats_gli_vqa_subj{subjective_only}_clean_data_v2_1.json"
    train_file = f"brats_gli_vqa_subj{subjective_only}_train_v2_1.json"
    val_file = f"brats_gli_vqa_subj{subjective_only}_val_v2_1.json"
    test_file = f"brats_gli_vqa_subj{subjective_only}_test_v2_1.json"
    slice_idx = 120
    seg_files_ = sorted(list(glob(f'/local2/amvepa91/MedTrinity-25M/output_pngs/*/*seg_slice_{slice_idx}_y.png')))
    vqa_data_ = generate_vqa_data_from_seg_file_joblib(seg_files_, subjective_only=subjective_only, include_quadrant=False,
                                                       n_jobs=8)
    with open(vqa_file, 'w') as f:
        json.dump(vqa_data_, f, indent=2)
    with open(vqa_file, 'r') as f:
        vqa_data = json.load(f)
    print(summarize_vqa_data(vqa_data))
    processed_vqa_data = postprocess_vqa_data(vqa_data, max_num_of_seg_ids_per_empty_count=10, save_vqa_file=clean_vqa_file)
    generate_train_val_test_splits(processed_vqa_data, train_file=train_file, val_file=val_file, test_file=test_file)

