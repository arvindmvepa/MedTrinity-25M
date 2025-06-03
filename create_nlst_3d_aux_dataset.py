import json
import numpy as np
from create_nlst_3d_dataset import sct_ab_code_dict, sct_epi_loc_dict, sct_margins_dict, sct_pre_att_dict, \
    sct_ab_attn_dict, sct_ab_gwth_dict, sct_ab_invg_dict, sct_ab_preexist_dict



abnormality_type_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_ab_code_dict.values()))}
location_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_epi_loc_dict.values()))}
margins_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_margins_dict.values()))}
pre_att_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_pre_att_dict.values()))}
interval_change_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_ab_attn_dict.values()))}
interval_growth_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_ab_gwth_dict.values()))}
further_investigation_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_ab_invg_dict.values()))}
ab_preexist_map = {val: index for index, val in enumerate(["NA"] + sorted(sct_ab_preexist_dict.values()))}

EXTENT_MAP = {
    "none": 0,
    "very sparse": 1,
    "somewhat scattered": 2,
    "partially filled": 3,
    "nearly filled": 4,
    "almost fully filled": 5,
}

SOLIDITY_MAP = {
    "none": 0,
    "highly irregular and scattered": 1,
    "somewhat compact but irregular": 2,
    "mostly compact": 3,
}


def convert_dict_to_numeric(original_data, na_string="NA", nan_string="nan", sep_string="|"):
    """
    Given the nested dictionary structure shown above,
    return a new dictionary with numeric codes for
    area, extent, and solidity, plus a list of integer codes for bbox.
    """
    new_data = {}

    for (img_files, init_study_yr, final_study_yr), content_type_dict in original_data.items():

        abnormality_type = content_type_dict.get("abnormality_type", na_string).split(sep_string)
        pre_existing = content_type_dict.get("pre-existing", na_string).split(sep_string)
        location = content_type_dict.get("location", na_string).split(sep_string)
        interval_change = content_type_dict.get("interval_change", na_string).split(sep_string)
        interval_growth = content_type_dict.get("interval_growth", na_string).split(sep_string)
        further_investigation = content_type_dict.get("further_investigation", na_string).split(sep_string)
        margins = content_type_dict.get("margins", na_string).split(sep_string)
        predominant_attenuation = content_type_dict.get("predominant_attenuation", na_string).split(sep_string)
        longest_diameter = content_type_dict.get("longest_diameter", nan_string).split(sep_string)
        longest_perpendicular_diameter = content_type_dict.get("longest_perpendicular_diameter", nan_string).split(sep_string)

        # Convert each one to numeric / codes
        abnormality_type = [abnormality_type_map[item.strip()] for item in abnormality_type]
        pre_existing = [ab_preexist_map[item.strip()] for item in pre_existing]
        location = [location_map[item.strip()] for item in location]
        interval_change = [interval_change_map[item.strip()] for item in interval_change]
        interval_growth = [interval_growth_map[item.strip()] for item in interval_growth]
        further_investigation = [further_investigation_map[item.strip()] for item in further_investigation]
        margins = [margins_map[item.strip()] for item in margins]
        predominant_attenuation = [pre_att_map[item.strip()] for item in predominant_attenuation]
        longest_diameter = [float(item.strip()) for item in longest_diameter]
        longest_perpendicular_diameter = [float(item.strip()) for item in longest_perpendicular_diameter]

        # Build the new metrics
        new_content_type_dict = {
            "abnormality_type": abnormality_type,
            "pre_existing": pre_existing,
            "location": location,
            "interval_change": interval_change,
            "interval_growth": interval_growth,
            "further_investigation": further_investigation,
            "margins": margins,
            "predominant_attenuation": predominant_attenuation,
            "longest_diameter": longest_diameter,
            "longest_perpendicular_diameter": longest_perpendicular_diameter,
        }
        new_data[(tuple(img_files), init_study_yr, final_study_yr)] = new_content_type_dict

    return new_data


def convert_numeric_dict_to_list(numeric_data):
    """
    Given the 'numeric_data' dict from convert_dict_to_numeric(),
    produce a list of dicts, one per seg_file, sorted by seg_file.
    Each dict has keys: id, seg_file, labels (the label metrics).
    """
    keys_sorted = sorted(numeric_data.keys(), key= lambda x: str(x[0]))  # sort by seg_file path
    result_list = []

    for i, (img_files, init_study_yr, final_study_yr) in enumerate(keys_sorted):
        content_info = numeric_data[(img_files, init_study_yr, final_study_yr)]
        entry = {
            "id": i,
            "img_files": img_files,
            "init_study_yr": init_study_yr,
            "final_study_yr": final_study_yr,
            "content_info": content_info
        }
        result_list.append(entry)

    return result_list


def build_gt_lookup(vqa_questions, content_types=("abnormality_type", "pre-existing", "location", "interval_change",
                                                  "interval_growth", "further_investigation", "margins",
                                                  "predominant_attenuation", "longest_diameter",
                                                  "longest_perpendicular_diameter")):
    gt_lookup = {}
    for entry in vqa_questions:
        img_files = tuple(entry["img_files"])
        content_type = entry["content_type"]
        init_study_yr = entry["init_study_yr"]
        final_study_yr = entry["final_study_yr"]
        answer = entry["answer"].strip()
        if content_type not in content_types:
            continue
        key = (img_files, content_type, init_study_yr, final_study_yr)
        gt_lookup[key] = answer
    return gt_lookup


def build_aux_tasks(all_vqa_questions, content_types=("abnormality_type", "pre-existing", "location", "interval_change",
                                                      "interval_growth", "further_investigation", "margins",
                                                      "predominant_attenuation", "longest_diameter",
                                                      "longest_perpendicular_diameter")):
    """
    Convert the original Q&A JSON into
    one row per (volume_seg_file, label_name, type),
    with a fixed set of 4 labels x 4 types = 16 rows per volume_seg_file.
    """
    # 1) Build the ground-truth lookup from the Q&A
    gt_lookup = build_gt_lookup(all_vqa_questions)

    # 2) Identify all seg_files in the data
    img_files_set = set(tuple(entry["img_files"]) for entry in all_vqa_questions)


    # 5) Build the final list of rows
    aux_dict = {}
    for img_files in sorted(img_files_set):
        for init_study_yr, final_study_yr in [(0, 1), (1, 2), (0, 2)]:
            content_type_dict = {}
            for content_type in content_types:
                key = (img_files, content_type, init_study_yr, final_study_yr)
                if key in gt_lookup:
                    gt_value = gt_lookup[key]
                else:
                    continue
                content_type_dict[content_type] = gt_value
            aux_dict[(img_files, init_study_yr, final_study_yr)] = content_type_dict
    return aux_dict


if __name__ == "__main__":
    # reference vqa files to line up seg_ids and train/val/test splits
    ref_train_vqa_file = None
    ref_val_vqa_file = None
    ref_test_vqa_file = None

    # params
    add_time_delta2 = True
    tag = "v3"


    vqa_file = "nlst_vqa_add_time_delta2{}_{}.json"
    clean_vqa_file = "nlst_vqa_filt_delta2{}_{}.json"
    train_file = "nlst_train_vqa_delta2{}_{}.json"
    train_aux_file = "nlst_train_aux_vqa_delta2{}_{}.json"
    val_file = "nlst_val_vqa_delta2{}_{}.json"
    val_aux_file = "nlst_val_aux_vqa_delta2{}_{}.json"
    test_file = "nlst_test_vqa_delta2{}_{}.json"
    test_aux_file = "nlst_test_aux_vqa_delta2{}_{}.json"

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
