import json
import os


if __name__ == "__main__":
    report_file = "/local2/amvepa91/RadGenome-Brain_MRI/BraTS_MET/impression.json"

    subjective_only = True
    train_vqa_file = "brats_{}_3d_vqa_subj{}_train_updated_{}_multitask_fixed.json"
    val_vqa_file = "brats_{}_3d_vqa_subj{}_val_updated_{}_multitask_fixed.json"
    test_vqa_file = "brats_{}_3d_vqa_subj{}_test_updated_{}_multitask_fixed.json"
    seed = 0

    # MET dataset settings
    dataset_type = "met"
    version = f"v11_seed{seed}"
    train_file = train_vqa_file.format(dataset_type, subjective_only, version)
    val_file = val_vqa_file.format(dataset_type, subjective_only, version)
    test_file = test_vqa_file.format(dataset_type, subjective_only, version)

    with open(report_file, "r") as f:
        report_data = json.load(f)
    
    with open(train_file, "r") as f:
        train_data = json.load(f)

    with open(val_file, "r") as f:
        val_data = json.load(f)

    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    train_reports = []
    previously_seen = set()
    for item in train_data:
        seg_id = os.path.basename(item["volume_file_dir"])
        if seg_id in report_data and seg_id not in previously_seen:
            item["question"] = "What is the impression of the radiology report for this MRI scan?"
            item["answer_gen"] = report_data[seg_id]
            train_reports.append(item["answer_gen"])
            previously_seen.add(seg_id)
    val_reports = []
    for item in val_data:
        seg_id = os.path.basename(item["volume_file_dir"])
        if seg_id in report_data and seg_id not in previously_seen:
            item["question"] = "What is the impression of the radiology report for this MRI scan?"
            item["answer_gen"] = report_data[seg_id]
            val_reports.append(item["answer_gen"])
            previously_seen.add(seg_id)
    test_reports = []
    for item in test_data:
        seg_id = os.path.basename(item["volume_file_dir"])
        if seg_id in report_data and seg_id not in previously_seen:
            item["question"] = "What is the impression of the radiology report for this MRI scan?"
            item["answer_gen"] = report_data[seg_id]
            test_reports.append(item["answer_gen"])
            previously_seen.add(seg_id) 
    print(f"Number of training samples with reports: {len(train_reports)}")
    print(f"Number of validation samples with reports: {len(val_reports)}")
    print(f"Number of test samples with reports: {len(test_reports)}") 