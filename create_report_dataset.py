import json
import os


if __name__ == "__main__":
    report_file = "/local2/amvepa91/RadGenome-Brain_MRI/BraTS_MET/impression.json"

    subjective_only = True
    seed = 0

    train_vqa_file = "brats_{}_3d_vqa_subj{}_train_updated_{}_multitask_fixed.json"
    val_vqa_file = "brats_{}_3d_vqa_subj{}_val_updated_{}_multitask_fixed.json"
    test_vqa_file = "brats_{}_3d_vqa_subj{}_test_updated_{}_multitask_fixed.json"

    train_report_file = "brats_{}_3d_vqa_subj{}_train_updated_{}_report.json"
    val_report_file = "brats_{}_3d_vqa_subj{}_val_updated_{}_report.json"
    test_report_file = "brats_{}_3d_vqa_subj{}_test_updated_{}_report.json"

    # MET dataset settings
    dataset_type = "met"
    version = f"v11_seed{seed}"
    train_file = train_vqa_file.format(dataset_type, subjective_only, version)
    val_file = val_vqa_file.format(dataset_type, subjective_only, version)
    test_file = test_vqa_file.format(dataset_type, subjective_only, version)

    train_report_file = train_report_file.format(dataset_type, subjective_only, version)
    val_report_file = val_report_file.format(dataset_type, subjective_only, version)
    test_report_file = test_report_file.format(dataset_type, subjective_only, version)

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
            item["answer"] = report_data[seg_id]["impression"]
            item["answer_gen"] = report_data[seg_id]["impression"]
            item['answer_vqa'] = []
            item['label_name'] = "NA"
            item['type'] = 'report'
            item['base_type'] = 'report'
            item['content_type'] = 'report'
            item['combo'] = []
            item['answer_vqa_numeric'] = []
            train_reports.append(item)
            previously_seen.add(seg_id)
    val_reports = []
    for item in val_data:
        seg_id = os.path.basename(item["volume_file_dir"])
        if seg_id in report_data and seg_id not in previously_seen:
            item["question"] = "What is the impression of the radiology report for this MRI scan?"
            item["answer"] = report_data[seg_id]["impression"]
            item["answer_gen"] = report_data[seg_id]["impression"]
            item['answer_vqa'] = []
            item['label_name'] = "NA"
            item['type'] = 'report'
            item['base_type'] = 'report'
            item['content_type'] = 'report'
            item['combo'] = []
            item['answer_vqa_numeric'] = []
            val_reports.append(item)
            previously_seen.add(seg_id)
    test_reports = []
    for item in test_data:
        seg_id = os.path.basename(item["volume_file_dir"])
        if seg_id in report_data and seg_id not in previously_seen:
            item["question"] = "What is the impression of the radiology report for this MRI scan?"
            item["answer"] = report_data[seg_id]["impression"]
            item["answer_gen"] = report_data[seg_id]["impression"]
            item['answer_vqa'] = []
            item['label_name'] = "NA"
            item['type'] = 'report'
            item['base_type'] = 'report'
            item['content_type'] = 'report'
            item['combo'] = []
            item['answer_vqa_numeric'] = []
            test_reports.append(item)
            previously_seen.add(seg_id) 
    print(f"Number of training samples with reports: {len(train_reports)}")
    print(f"Number of validation samples with reports: {len(val_reports)}")
    print(f"Number of test samples with reports: {len(test_reports)}")
    with open(train_report_file, "w") as f:
        json.dump(train_reports, f, indent=4)
    with open(val_report_file, "w") as f:
        json.dump(val_reports, f, indent=4)
    with open(test_report_file, "w") as f:
        json.dump(test_reports, f, indent=4)