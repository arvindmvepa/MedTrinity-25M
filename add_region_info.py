import json


def add_region_info(vqa_data):
    region_info_dict = dict()
    for entry in vqa_data:
        content_type = entry["type"]
        if content_type == "region":
            study_name = entry["study_name"]
            # extract index of region answer from the combo list
            region_answer_index = entry["combo"].index(2)
            region_value = entry["answer_vqa"][region_answer_index][0]
            region_info_dict[study_name] = region_value
    for q_i in range(len(vqa_data)):
        entry = vqa_data[q_i]
        study_name = entry["study_name"]
        region_info = region_info_dict[study_name]
        question = entry["question"]
        updated_question = f"Region(s): {region_info}\n{question}"
        vqa_data[q_i]["question"] = updated_question
    return vqa_data


if __name__ == "__main__":
    ref_train_vqa_file = "brats_{}_3d_vqa_subj{}_train_{}_multitask_fixed.json"
    ref_val_vqa_file = "brats_{}_3d_vqa_subj{}_val_{}_multitask_fixed.json"
    ref_test_vqa_file = "brats_{}_3d_vqa_subj{}_test_{}_multitask_fixed.json"

    train_vqa_file = "brats_{}_3d_vqa_subj{}_train_{}_multitask_fixed_region_info.json"
    val_vqa_file = "brats_{}_3d_vqa_subj{}_val_{}_multitask_fixed_region_info.json"
    test_vqa_file = "brats_{}_3d_vqa_subj{}_test_{}_multitask_fixed_region_info.json"

    # rest of the parameters
    subjective_only = True
    dataset_seed = 0
    new_dataset_seed = 0

    # GLI dataset settings
    dataset_type = "gli"
    version = f"updated_v11_seed{dataset_seed}"
    labels_order = (1, 2, 3, 4)
    pediatric = False
    goat = False

    # MET dataset settings
    #dataset_type = "met"
    #version = f"updated_v2_seed{dataset_seed}"
    #labels_order = (1, 2, 3)
    #pediatric = False
    #goat = False

    # GoAT dataset settings
    #dataset_type = "goat"
    #version = f"updated_v2_seed{dataset_seed}"
    #labels_order = (1, 2, 3)
    #pediatric = False
    #goat = True

    ref_train_vqa_file = ref_train_vqa_file.format(dataset_type, subjective_only, version)
    ref_val_vqa_file = ref_val_vqa_file.format(dataset_type, subjective_only, version)
    ref_test_vqa_file = ref_test_vqa_file.format(dataset_type, subjective_only, version)

    train_vqa_file = train_vqa_file.format(dataset_type, subjective_only, version)
    val_vqa_file = val_vqa_file.format(dataset_type, subjective_only, version)
    test_vqa_file = test_vqa_file.format(dataset_type, subjective_only, version)

    question_key = "volume_file_id"
    type_key = "type"

    with open(ref_train_vqa_file, 'r') as f:
        ref_train_vqa_data = json.load(f)
        ref_train_vqa_data = add_region_info(ref_train_vqa_data)
    with open(ref_val_vqa_file, 'r') as f:
        ref_val_vqa_data = json.load(f)
        ref_val_vqa_data = add_region_info(ref_val_vqa_data)
    with open(ref_test_vqa_file, 'r') as f:
        ref_test_vqa_data = json.load(f)
        ref_test_vqa_data = add_region_info(ref_test_vqa_data)
    
    # save updated vqa files with region info in questions
    with open(train_vqa_file, 'w') as f:
        json.dump(ref_train_vqa_data, f, indent=2)
    with open(val_vqa_file, 'w') as f:
        json.dump(ref_val_vqa_data, f, indent=2)
    with open(test_vqa_file, 'w') as f:
        json.dump(ref_test_vqa_data, f, indent=2)