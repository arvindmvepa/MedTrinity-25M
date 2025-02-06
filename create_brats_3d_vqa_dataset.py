from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import json
from create_brats_imaging_dataset import get_nifti_seg_file_from_dir, get_nifti_non_seg_file_from_dir, \
    load_lab_map_from_nifti
from vqa_utils import analyze_3d_label_summary, summarize_3d_vqa_data, generate_labal_vqa_questions, \
    postprocess_3d_vqa_data, generate_train_val_test_splits


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
    ref_train_vqa_file = f"brats_gli_3d_vqa_subjFalse_train_v2.json"
    ref_val_vqa_file = f"brats_gli_3d_vqa_subjFalse_val_v2.json"
    ref_test_vqa_file = f"brats_gli_3d_vqa_subjFalse_test_v2.json"
    # rest of the parameters
    subjective_only = True
    vqa_file = f"brats_gli_3d_vqa_subj{subjective_only}_data_v2.json"
    clean_vqa_file = f"brats_3d_gli_vqa_subj{subjective_only}_clean_data_v2.json"
    train_file = f"brats_gli_3d_vqa_subj{subjective_only}_train_v2.json"
    val_file = f"brats_gli_3d_vqa_subj{subjective_only}_val_v2.json"
    test_file = f"brats_gli_3d_vqa_subj{subjective_only}_test_v2.json"
    volume_file_dirs = sorted(list(glob(f'/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/*')))
    question_key = "volume_file_id"
    vqa_data_ = generate_vqa_data_from_seg_file_joblib(volume_file_dirs, subjective_only=subjective_only,
                                                       include_quadrant=False, n_jobs=8)
    with open(vqa_file, 'w') as f:
        json.dump(vqa_data_, f, indent=2)
    with open(vqa_file, 'r') as f:
        vqa_data_ = json.load(f)
    print(summarize_3d_vqa_data(vqa_data_))
    processed_vqa_data = postprocess_3d_vqa_data(vqa_data_, save_vqa_file=clean_vqa_file)
    if (ref_train_vqa_file is not None) and (ref_val_vqa_file is not None) and (ref_test_vqa_file is not None):
        with open(ref_train_vqa_file, 'r') as f:
            ref_train_vqa_data = json.load(f)
            ref_train_seg_ids = [q[question_key] for q in ref_train_vqa_data]
        with open(ref_val_vqa_file, 'r') as f:
            ref_val_vqa_data = json.load(f)
            ref_val_seg_ids = [q[question_key] for q in ref_val_vqa_data]
        with open(ref_test_vqa_file, 'r') as f:
            ref_test_vqa_data = json.load(f)
            ref_test_seg_ids = [q[question_key] for q in ref_test_vqa_data]
        generate_train_val_test_splits(processed_vqa_data, train_seg_ids=ref_train_seg_ids, val_seg_ids=ref_val_seg_ids,
                                       test_seg_ids=ref_test_seg_ids, train_file=train_file, val_file=val_file,
                                       test_file=test_file)
    else:
        generate_train_val_test_splits(processed_vqa_data, question_key=question_key, train_file=train_file,
                                       val_file=val_file, test_file=test_file)
