import os.path
from glob import glob
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from collections import defaultdict
import json
from create_brats_imaging_dataset import load_color_seg_png_as_labels
from vqa_utils import label_names, analyze_label_relationship, analyze_label_summary, get_seg_ids_empty_counts, \
    extract_label_intensity_components, summarize_vqa_data


def generate_train_val_test_splits(all_vqa_questions, seed=0, train_seg_ids=(), val_seg_ids=(), test_seg_ids=(),
                                   train_frac=0.8, val_frac=0.1, train_file="brats_gli_vqa_train.json",
                                   val_file="brats_gli_vqa_val.json", test_file="brats_gli_vqa_test.json"):
    if (train_seg_ids is not None) and (val_seg_ids is not None) and (test_seg_ids is not None):
        train_questions = [q for q in all_vqa_questions if q["seg_id"] in train_seg_ids]
        val_questions = [q for q in all_vqa_questions if q["seg_id"] in val_seg_ids]
        test_questions = [q for q in all_vqa_questions if q["seg_id"] in test_seg_ids]
        train_studies = list({q["study_name"] for q in train_questions})
        val_studies = list({q["study_name"] for q in val_questions})
        test_studies = list({q["study_name"] for q in test_questions})
    else:
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

def postprocess_vqa_data(all_vqa_questions, seg_id_list=(), max_num_of_seg_ids_per_empty_count=100,
                         default_modality="t1c", save_vqa_file="brats_gli_vqa_clean_data.json", seed=0):
    if seg_id_list:
        filtered_vqa_questions = [q for q in all_vqa_questions if q["seg_id"] in seg_id_list]
    elif max_num_of_seg_ids_per_empty_count is not None:
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
    # reference vqa files to line up seg_ids and train/val/test splits
    ref_vqa_file = "brats_gli_vqa_subjFalse_clean_data_v2.json"
    ref_train_vqa_file = "brats_gli_vqa_subjFalse_train_v2.json"
    ref_val_vqa_file = "brats_gli_vqa_subjFalse_val_v2.json"
    ref_test_vqa_file = "brats_gli_vqa_subjFalse_test_v2.json"
    # rest of the parameters
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
    if ref_vqa_file is not None:
        with open(ref_vqa_file, 'r') as f:
            ref_vqa_data = json.load(f)
            ref_seg_ids = [q["seg_id"] for q in ref_vqa_data]
        processed_vqa_data = postprocess_vqa_data(vqa_data, seg_id_list=ref_seg_ids, save_vqa_file=clean_vqa_file)
        with open(ref_train_vqa_file, 'r') as f:
            ref_train_vqa_data = json.load(f)
            ref_train_seg_ids = [q["seg_id"] for q in ref_train_vqa_data]
        with open(ref_val_vqa_file, 'r') as f:
            ref_val_vqa_data = json.load(f)
            ref_val_seg_ids = [q["seg_id"] for q in ref_val_vqa_data]
        with open(ref_test_vqa_file, 'r') as f:
            ref_test_vqa_data = json.load(f)
            ref_test_seg_ids = [q["seg_id"] for q in ref_test_vqa_data]
        generate_train_val_test_splits(processed_vqa_data, train_seg_ids=ref_train_seg_ids, val_seg_ids=ref_val_seg_ids,
                                       test_seg_ids=ref_test_seg_ids, train_file=train_file, val_file=val_file,
                                       test_file=test_file)
    else:
        processed_vqa_data = postprocess_vqa_data(vqa_data, max_num_of_seg_ids_per_empty_count=10, save_vqa_file=clean_vqa_file)
        generate_train_val_test_splits(processed_vqa_data, train_file=train_file, val_file=val_file, test_file=test_file)

