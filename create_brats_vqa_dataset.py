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
    extract_label_intensity_components, summarize_vqa_data, postprocess_vqa_data, generate_train_val_test_splits, \
    generate_labal_vqa_questions


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

