import os.path
from glob import glob
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import json
from create_brats_imaging_dataset import get_nifti_seg_file_from_dir, get_nifti_non_seg_file_from_dir, load_lab_map_from_nifti
from vqa_utils import analyze_3d_label_summary, summarize_3d_vqa_data


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


def generate_vqa_from_seg_map(volume_file_dir, volume_id, include_area=True, include_quadrant=False,
                              include_bbox=True, include_extent=True, include_solidity=True, subjective_only=False):
    """
    Master function to produce a textual report combining:
      - Label summaries (area %, quadrant, bounding box, extent-based compactness)
        with subjective interpretations.
      - Non-Enh vs Enh tumor adjacency info.
      - FLAIR vs Tumor Core adjacency info.
      - Resection cavity vs tumor core & FLAIR.
    """
    nii_seg_file = get_nifti_seg_file_from_dir(volume_file_dir)
    seg_map_3d = load_lab_map_from_nifti(nii_seg_file)

    height, width, depth = seg_map_3d.shape
    total_pixels = seg_map_3d.size

    all_vqa_questions = []

    # Summaries of labels
    label_summaries = analyze_3d_label_summary(seg_map_3d=seg_map_3d,height=height, width=width, depth=depth, total_pixels=total_pixels)
    vqa_questions = []
    # get single label questions
    for summ in label_summaries:
        label_vqa_questions = generate_labal_vqa_questions(summ=summ, include_area=include_area,
                                                           include_quadrant=include_quadrant,
                                                           include_bbox=include_bbox,
                                                           include_extent=include_extent,
                                                           include_solidity=include_solidity,
                                                           subjective_only=subjective_only)
        vqa_questions.extend(label_vqa_questions)
    non_seg_files_dict = get_nifti_non_seg_file_from_dir(volume_file_dir)
    for q in vqa_questions:
        q["volume_file_id"] = volume_id
        q["volume_file_dir"] = volume_file_dir
        q["volume_seg_file"] = nii_seg_file
        q["volume_non_seg_files"] = non_seg_files_dict
    all_vqa_questions.extend(vqa_questions)
    return all_vqa_questions


def generate_vqa_data_from_seg_file(seg_files, include_area=True, include_quadrant=True, include_bbox=True,
                                    include_extent=True, include_solidity=True):
    all_vqa_questions = []
    for volume_id, volume_file_dir in tqdm(enumerate(seg_files)):
        vqa_data = generate_vqa_from_seg_map(volume_file_dir=volume_file_dir, volume_id=volume_id, include_area=include_area,
                                             include_quadrant=include_quadrant, include_bbox=include_bbox,
                                             include_extent=include_extent, include_solidity=include_solidity,)
        all_vqa_questions.extend(vqa_data)
    return all_vqa_questions


def generate_vqa_data_from_seg_file_joblib(
    volume_file_dirs,
    n_jobs=-1,
    include_area=True,
    include_quadrant=True,
    include_bbox=True,
    include_extent=True,
    include_solidity=True,
    subjective_only=False
):
    """
    Parallelized version of generating VQA data from a list of seg_files,
    with a progress bar (tqdm_joblib).

    Parameters
    ----------
    volume_file_dirs : list of str
        Paths to volume file dirs (NIFTI).
    n_jobs : int, default=-1
        Number of parallel jobs. -1 => use all cores.
    include_area, include_quadrant, include_bbox, ...
        Configuration flags passed down to generate_vqa_from_seg_map.
    """
    all_vqa_questions = []

    # Wrap Parallel execution with tqdm_joblib for the progress bar:
    with tqdm_joblib(desc="Processing segmentation files", total=len(volume_file_dirs)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(generate_vqa_from_seg_map)(
                volume_file_dir,
                volume_id,
                include_area,
                include_quadrant,
                include_bbox,
                include_extent,
                include_solidity,
                subjective_only,
            )
            for volume_id, volume_file_dir in enumerate(volume_file_dirs)
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
    ref_vqa_file = None
    ref_train_vqa_file = None
    ref_val_vqa_file = None
    ref_test_vqa_file = None
    # rest of the parameters
    subjective_only = True
    vqa_file = f"brats_gli_3d_vqa_subj{subjective_only}_data.json"
    clean_vqa_file = f"brats_3d_gli_vqa_subj{subjective_only}_clean_data.json"
    train_file = f"brats_gli_3d_vqa_subj{subjective_only}_train.json"
    val_file = f"brats_gli_3d_vqa_subj{subjective_only}_val.json"
    test_file = f"brats_gli_3d_vqa_subj{subjective_only}_test.json"
    volume_file_dirs = sorted(list(glob(f'/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/*')))
    vqa_data_ = generate_vqa_data_from_seg_file_joblib(volume_file_dirs, subjective_only=subjective_only,
                                                       include_quadrant=False, n_jobs=8)
    with open(vqa_file, 'w') as f:
        json.dump(vqa_data_, f, indent=2)
    with open(vqa_file, 'r') as f:
        vqa_data = json.load(f)
    print(summarize_3d_vqa_data(vqa_data))
    """
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
    """
