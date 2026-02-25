import json
from collections import Counter


def summarize_aux(aux_list):
    """
    Produces summary statistics from the final aux list of dictionaries
    """
    cancer_list = []

    for it in aux_list:
        content_info = it["content_info"]
        cancer = content_info["cancer"]

        # Collect all the values into lists
        cancer_list.append(cancer)

    cancer_counter = Counter([result for result in cancer_list])

    print("Summary of auxiliary data:")
    print(f"Total entries: {len(aux_list)}")
    print(f"Cancer: {cancer_counter}")


def convert_dict_to_numeric(original_data, na_string="NA", nan_string="nan", sep_string="|"):
    """
    Given the nested dictionary structure shown above,
    return a new dictionary with numeric codes for
    area, extent, and solidity, plus a list of integer codes for bbox.
    """
    new_data = {}

    for (pid, embedding_path, init_study_yr, final_study_yr), content_type_dict in original_data.items():
        location = content_type_dict["location"].split(sep_string)
        interval_change = content_type_dict["interval_change"].split(sep_string)
        interval_growth = content_type_dict["interval_growth"].split(sep_string)
        margins = content_type_dict["margins"].split(sep_string)
        predominant_attenuation = content_type_dict["predominant_attenuation"].split(sep_string)
        longest_diameter = content_type_dict["longest_diameter"].split(sep_string)
        longest_perpendicular_diameter = content_type_dict["longest_perpendicular_diameter"].split(sep_string)

        # Convert each one to numeric / codes
        location = [location_map[item.strip()] for item in location]
        interval_change = [interval_change_map[item.strip()] for item in interval_change]
        interval_growth = [interval_growth_map[item.strip()] for item in interval_growth]
        margins = [margins_map[item.strip()] for item in margins]
        predominant_attenuation = [pre_att_map[item.strip()] for item in predominant_attenuation]
        longest_diameter = [float(item.strip()) for item in longest_diameter]
        longest_perpendicular_diameter = [float(item.strip()) for item in longest_perpendicular_diameter]

        # Build the new metrics
        new_content_type_dict = {
            "location": location,
            "interval_change": interval_change,
            "interval_growth": interval_growth,
            "margins": margins,
            "predominant_attenuation": predominant_attenuation,
            "longest_diameter": longest_diameter,
            "longest_perpendicular_diameter": longest_perpendicular_diameter,
        }
        new_data[(pid, embedding_path, init_study_yr, final_study_yr)] = new_content_type_dict

    return new_data


def convert_numeric_dict_to_list(numeric_data):
    """
    Given the 'numeric_data' dict from convert_dict_to_numeric(),
    produce a list of dicts, one per seg_file, sorted by seg_file.
    Each dict has keys: id, seg_file, labels (the label metrics).
    """
    keys_sorted = sorted(numeric_data.keys(), key= lambda x: str(x[0]))  # sort by seg_file path
    result_list = []

    for i, (pid, embedding_path, init_study_yr, final_study_yr) in enumerate(keys_sorted):
        content_info = numeric_data[(pid, embedding_path, init_study_yr, final_study_yr)]
        entry = {
            "id": i,
            "pid": pid,
            "embedding_path": embedding_path,
            "init_study_yr": init_study_yr,
            "final_study_yr": final_study_yr,
            "content_info": content_info
        }
        result_list.append(entry)

    return result_list


def build_gt_lookup(vqa_questions, content_types=("cancer")):
    gt_lookup = {}
    for entry in vqa_questions:
        pid = entry["pid"]
        embedding_path = entry["embedding_path"]
        study_yr = entry["study_yr"]
        content_type = entry["content_type"]
        answer = entry["answer"].strip()
        if content_type not in content_types:
            continue
        key = (pid, embedding_path, study_yr, content_type)
        gt_lookup[key] = answer
    return gt_lookup


def build_aux_tasks(all_vqa_questions, content_types=("cancer",)):
    """
    Convert the original Q&A JSON into
    one row per (volume_seg_file, label_name, type),
    with a fixed set of 4 labels x 4 types = 16 rows per volume_seg_file.
    """
    # 1) Build the ground-truth lookup from the Q&A
    gt_lookup = build_gt_lookup(all_vqa_questions)

    # 2) Identify all seg_files in the data
    pid_embedding_path_set = set((entry["pid"], entry["embedding_path"]) for entry in all_vqa_questions)

    # 5) Build the final list of rows
    aux_dict = {}
    for pid, embedding_path in sorted(pid_embedding_path_set):
        for study_yr in [0, 1, 2]:
            content_type_dict = {}
            for content_type in content_types:
                key = (pid, embedding_path, study_yr, content_type)
                if key in gt_lookup:
                    gt_value = gt_lookup[key]
                else:
                    continue
                content_type_dict[content_type] = gt_value
            # Only add to aux_dict if content_type_dict is not empty
            if content_type_dict:
                aux_dict[(pid, embedding_path, study_yr)] = content_type_dict
    return aux_dict


if __name__ == "__main__":
    # reference vqa files to line up seg_ids and train/val/test splits
    ref_train_vqa_file = None
    ref_val_vqa_file = None
    ref_test_vqa_file = None

    # params
    add_time_delta2 = True
    tag = "v0"


    vqa_file = "nlst_cancer_vqa_add_time_delta2{}_{}.json"
    clean_vqa_file = "nlst_cancer_vqa_filt_delta2{}_{}.json"
    train_file = "nlst_cancer_train_vqa_delta2{}_{}.json"
    train_aux_file = "nlst_cancer_train_aux_vqa_delta2{}_{}.json"
    val_file = "nlst_cancer_val_vqa_delta2{}_{}.json"
    val_aux_file = "nlst_cancer_val_aux_vqa_delta2{}_{}.json"
    test_file = "nlst_cancer_test_vqa_delta2{}_{}.json"
    test_aux_file = "nlst_cancer_test_aux_vqa_delta2{}_{}.json"

    train_file = train_file.format(add_time_delta2, tag)
    train_aux_file = train_aux_file.format(add_time_delta2, tag)
    val_file = val_file.format(add_time_delta2, tag)
    val_aux_file = val_aux_file.format(add_time_delta2, tag)
    test_file = test_file.format(add_time_delta2, tag)
    test_aux_file = test_aux_file.format(add_time_delta2, tag)

    with open(train_file, 'r') as f:
        train_vqa_data = json.load(f)
    with open(val_file, 'r') as f:
        val_vqa_data = json.load(f)
    with open(test_file, 'r') as f:
        test_vqa_data = json.load(f)

    train_vqa_aux_data = convert_numeric_dict_to_list(convert_dict_to_numeric(build_aux_tasks(train_vqa_data)))
    val_vqa_aux_data = convert_numeric_dict_to_list(convert_dict_to_numeric(build_aux_tasks(val_vqa_data)))
    test_vqa_aux_data = convert_numeric_dict_to_list(convert_dict_to_numeric(build_aux_tasks(test_vqa_data)))

    with open(train_aux_file, "w") as f:
        json.dump(train_vqa_aux_data, f, indent=4)
    with open(val_aux_file, "w") as f:
        json.dump(val_vqa_aux_data, f, indent=4)
    with open(test_aux_file, "w") as f:
        json.dump(test_vqa_aux_data, f, indent=4)

    print(f"Wrote {len(train_vqa_aux_data)} auxiliary rows to {train_aux_file}")
    print(f"Wrote {len(val_vqa_aux_data)} auxiliary rows to {val_aux_file}")
    print(f"Wrote {len(test_vqa_aux_data)} auxiliary rows to {test_aux_file}")

    # Summarize the auxiliary data
    print("Summary of all auxiliary data")
    summarize_aux(train_vqa_aux_data + val_vqa_aux_data + test_vqa_aux_data)
    print("Summary of train auxiliary data")
    summarize_aux(train_vqa_aux_data)
    print("Summary of val auxiliary data")
    summarize_aux(val_vqa_aux_data)
    print("Summary of test auxiliary data")
    summarize_aux(test_vqa_aux_data)
