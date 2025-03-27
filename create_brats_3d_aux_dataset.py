import re
from glob import glob
import json
from vqa_utils import label_names, ped_label_names, goat_label_names


def build_gt_lookup(vqa_questions, question_types=("area", "bbox", "extent", "solidity")):
    gt_lookup = {}
    for entry in vqa_questions:
        seg_file = entry["volume_seg_file"]
        label_name = entry["label_name"]
        q_type = entry["type"]
        answer = entry["answer"].strip()
        if q_type not in question_types:
            continue
        key = (seg_file, label_name, q_type)
        gt_lookup[key] = answer
    return gt_lookup


def build_aux_tasks(all_vqa_questions, dataset="gli", question_types=("area", "bbox", "extent", "solidity")):
    """
    Convert the original Q&A JSON into
    one row per (volume_seg_file, label_name, type),
    with a fixed set of 4 labels x 4 types = 16 rows per volume_seg_file.
    """
    # 1) Build the ground-truth lookup from the Q&A
    gt_lookup = build_gt_lookup(all_vqa_questions)

    # 2) Identify all seg_files in the data
    seg_files_set = set(entry["volume_seg_file"] for entry in all_vqa_questions)

    # 3) Get the Target Labels for the dataset
    if dataset == "gli":
        target_labels = list(label_names.values())[:5]
    elif dataset == "met":
        target_labels = list(label_names.values())[:4]
    elif dataset == "goat":
        target_labels = list(goat_label_names.values())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 5) Build the final list of rows
    aux_dict = {}
    for seg_file in sorted(seg_files_set):
        label_dict = {}
        for lbl_name in target_labels:
            q_type_dict = {}
            for q_type in question_types:
                key = (seg_file, lbl_name, q_type)
                if key in gt_lookup:
                    gt_value = gt_lookup[key]
                else:
                    continue
                q_type_dict[q_type] = gt_value
            label_dict[lbl_name] = q_type_dict
        aux_dict[seg_file] = label_dict

    return aux_dict


if __name__ == "__main__":
    # reference vqa files to line up seg_ids and train/val/test splits
    ref_train_vqa_file = None
    ref_val_vqa_file = None
    ref_test_vqa_file = None
    # rest of the parameters
    subjective_only = True

    vqa_file = "brats_{}_3d_vqa_subj{}_data_{}.json"
    clean_vqa_file = "brats_{}_3d_vqa_subj{}_clean_data_{}.json"
    train_file = "brats_{}_3d_vqa_subj{}_train_{}.json"
    train_aux_file = "brats_{}_3d_vqa_subj{}_train_aux_{}.json"
    val_file = "brats_{}_3d_vqa_subj{}_val_{}.json"
    val_aux_file = "brats_{}_3d_vqa_subj{}_val_aux_{}.json"
    test_file = "brats_{}_3d_vqa_subj{}_test_{}.json"
    test_aux_file = "brats_{}_3d_vqa_subj{}_test_aux_{}.json"
    seed = 0

    # GLI dataset settings
    dataset_type = "gli"
    version = f"v6_seed{seed}"
    volume_file_dirs = sorted(list(glob(f'/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/*')))
    labels_order = (1, 2, 3, 4)
    pediatric = False
    goat = False
    """
    # MET dataset settings
    dataset_type = "met"
    version = "v2_seed{seed}"
    volume_file_dirs = sorted(list(glob(f'/local2/shared_data/BraTS2024-BraTS-MET/MICCAI-BraTS2024-MET-Challenge-Training_overall/*')))
    labels_order = (1, 2, 3)
    pediatric = False
    goat = False

    # GoAT dataset settings
    dataset_type = "goat"
    version = "v2_seed{seed}"
    volume_file_dirs = sorted(list(glob(f'/local2/shared_data/BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/*')))
    labels_order = (1, 2, 3)
    pediatric = False
    goat = True
    """
    train_file = train_file.format(dataset_type, subjective_only, version)
    train_aux_file = train_aux_file.format(dataset_type, subjective_only, version)
    val_file = val_file.format(dataset_type, subjective_only, version)
    val_aux_file = val_aux_file.format(dataset_type, subjective_only, version)
    test_file = test_file.format(dataset_type, subjective_only, version)
    test_aux_file = test_aux_file.format(dataset_type, subjective_only, version)

    with open(train_file, 'r') as f:
        train_vqa_data = json.load(f)
    with open(val_file, 'r') as f:
        val_vqa_data = json.load(f)
    with open(test_file, 'r') as f:
        test_vqa_data = json.load(f)

    train_vqa_aux_data = build_aux_tasks(train_vqa_data)
    val_vqa_aux_data = build_aux_tasks(val_vqa_data)
    test_vqa_aux_data = build_aux_tasks(test_vqa_data)

    with open(train_aux_file, "w") as f:
        json.dump(train_vqa_aux_data, f, indent=4)
    with open(val_aux_file, "w") as f:
        json.dump(val_vqa_aux_data, f, indent=4)
    with open(test_aux_file, "w") as f:
        json.dump(test_vqa_aux_data, f, indent=4)

    print(f"Wrote {len(train_vqa_aux_data)} auxiliary rows to {train_aux_file}")
    print(f"Wrote {len(val_vqa_aux_data)} auxiliary rows to {val_aux_file}")
    print(f"Wrote {len(test_vqa_aux_data)} auxiliary rows to {test_aux_file}")
