import pyreadstat
import json
import pandas as pd
import random
import os


def get_npy_path(volume_path, img_root="/local/amvepa91/nlst_npy"):
    volume_name = os.path.basename(volume_path)
    time_point_dir = os.path.basename(os.path.dirname(volume_path))
    pid_dir = os.path.basename(os.path.dirname(os.path.dirname(volume_path)))
    volume_path_npy = os.path.join(img_root, pid_dir, time_point_dir, volume_name + ".npy")
    return volume_path_npy


# A small helper to handle "code not found in dict" => "NA"
def get_dict_value(dictionary, key, na_string="NA"):
    return dictionary.get(key, na_string)


def get_string_from_item_lst(rows, key, key_dict, na_string="NA", sep_string="|"):
    if len(rows) == 0:
        return na_string
    return sep_string.join([get_dict_value(key_dict, row[key]) for _, row in rows.iterrows()])


def get_string_from_numeric_lst(rows, key, nan_string="nan", sep_string="|"):
    if len(rows) == 0:
        return nan_string
    return sep_string.join([str(row.get(key, nan_string)) for _, row in rows.iterrows()])


def train_val_test_split_by_pid(final_vqa, val_pct=0.1, test_pct=0.1, seed=0):
    """
    Splits a list of VQA dicts into train, val, and test sets by PID.
    The val set is val_pct of unique PIDs,
    the test set is test_pct of unique PIDs,
    and the remainder goes to train.

    - final_vqa: list of dictionaries, each must have 'pid' key
    - val_pct: fraction of PIDs for validation
    - test_pct: fraction of PIDs for test
    - seed: random seed for reproducibility

    Returns: (train_list, val_list, test_list)
    """

    # 1) Collect unique PIDs from the list
    unique_pids = sorted({item["pid"] for item in final_vqa})
    total_pids = len(unique_pids)

    # 2) Shuffle PIDs
    random.seed(seed)
    random.shuffle(unique_pids)

    # 3) Determine how many PIDs go to val/test
    val_size = int(val_pct * total_pids)
    test_size = int(test_pct * total_pids)
    train_size = total_pids - val_size - test_size

    # 4) Slice the shuffled PIDs
    train_pids = set(unique_pids[:train_size])
    val_pids = set(unique_pids[train_size:train_size + val_size])
    test_pids = set(unique_pids[train_size + val_size:train_size + val_size + test_size])

    # 5) Partition the original list by checking pid membership
    train_list = [entry for entry in final_vqa if entry["pid"] in train_pids]
    val_list = [entry for entry in final_vqa if entry["pid"] in val_pids]
    test_list = [entry for entry in final_vqa if entry["pid"] in test_pids]

    return train_list, val_list, test_list


def filter_by_instution(all_vqas, inst_list):
    """
    Filter the VQA list by institution.
    """
    filt_inst_list = [qa for qa in all_vqas if qa["inst"] in inst_list]
    return filt_inst_list


def build_cancer_question(img_files, filters, pid, study_yr, inst, question_index, question, answer, numeric_answer,
                          content_type="cancer"):
    """
    Build a single Q–A dictionary with the relevant fields.
    """
    return {
        "pid": pid,
        "study_yr": study_yr,
        "inst": inst,
        "img_files": img_files,
        "filters": filters,
        "question": question,
        "answer": answer,
        "numeric_answer": numeric_answer,
        "qid": question_index,
        "content_type": content_type
    }


def get_cancer_question(img_files, filters, pid, study_yr, inst, question_index, has_cancer):
    qa = build_cancer_question(
        pid=pid,
        study_yr=study_yr,
        inst=inst,
        question="Will this patient develop cancer?",
        answer="yes" if has_cancer else "no",
        numeric_answer=int(has_cancer),
        img_files=img_files,
        filters=filters,
        question_index=question_index,
        content_type="cancer"
    )
    return qa


def generate_cancer_aux_from_df(index_df, ann_df):
    """
    Main function: iterates over the rows of 'df' and
    creates VQA Q–A pairs in a modular way.
    """
    cancer_aux_list = []
    question_index = 0
    for pid, group in index_df.groupby('pid'):
        pid_ann_df = ann_df.loc[ann_df["pid"] == pid]
        inst = pid_ann_df['cen'].iloc[0]
        cancyr = pid_ann_df['cancyr'].iloc[0]
        has_cancer = False
        if not pd.isna(cancyr):
            has_cancer = True

        grp_t0 = group["dicom_t0"].loc[~group["dicom_t0"].isnull()].tolist()
        grp_t0_filters = group["dicom_filter"].loc[~group["dicom_t0"].isnull()].tolist()

        grp_t1 = group["dicom_t1"].loc[~group["dicom_t1"].isnull()].tolist()
        grp_t1_filters = group["dicom_filter"].loc[~group["dicom_t1"].isnull()].tolist()

        grp_t2 = group["dicom_t2"].loc[~group["dicom_t2"].isnull()].tolist()
        grp_t2_filters = group["dicom_filter"].loc[~group["dicom_t2"].isnull()].tolist()

        if len(grp_t0) > 0:
            question_index += 1
            qa = get_cancer_question(img_files=grp_t0, filters=grp_t0_filters, pid=pid, study_yr=0, inst=inst,
                                     question_index=question_index, has_cancer=has_cancer)
            cancer_aux_list.append(qa)
        if len(grp_t1) > 0:
            question_index += 1
            qa = get_cancer_question(img_files=grp_t1, filters=grp_t1_filters, pid=pid, study_yr=1, inst=inst,
                                     question_index=question_index, has_cancer=has_cancer)
            cancer_aux_list.append(qa)
        if len(grp_t2) > 0:
            question_index += 1
            qa = get_cancer_question(img_files=grp_t2, filters=grp_t2_filters, pid=pid, study_yr=2, inst=inst,
                                     question_index=question_index, has_cancer=has_cancer)
            cancer_aux_list.append(qa)
    return cancer_aux_list


if __name__ == "__main__":
    measurement_file = "nlst_780_ctab_idc_20210527.csv"
    comparison_file = "nlst_780_ctabc_idc_20210527.csv"
    patient_file = "participant_d100814.sas7bdat"
    source_file = "nlst_index.csv"
    tag = "v4"

    save_file = f"nlst_aux_cancer_{tag}.json"
    filter_inst = ["AZ", "AG", "AQ", "AJ", "BA", "AU", "BE", "AC", "BF", "AE", "AP"]
    filt_save_file = f"nlst_aux_cancer_filt_{tag}.json"
    filt_save_pid_list = f"nlst_aux_cancer_filt_pids_{tag}.json"
    train_save_file = f"nlst_aux_cancer_train_{tag}.json"
    val_save_file = f"nlst_aux_cancer_val_{tag}.json"
    test_save_file = f"nlst_aux_cancer_test_{tag}.json"

    (patient_df, _) = pyreadstat.read_sas7bdat(patient_file)

    nlst_index_df = pd.read_csv(source_file)
    all_cancer_vqa = generate_cancer_aux_from_df(nlst_index_df, patient_df)

    print(f"==========OVERALL==========")
    with open(save_file, "w") as f:
        json.dump(all_cancer_vqa, f, indent=4)

    print(f"==========FILTERED==========")
    filtered_vqas = filter_by_instution(all_cancer_vqa, filter_inst)
    with open(filt_save_file, "w") as f:
        json.dump(filtered_vqas, f, indent=4)
    filtered_pids = sorted({qa["pid"] for qa in filtered_vqas})
    with open(filt_save_pid_list, "w") as f:
        json.dump(filtered_pids, f)
    print(f"Wrote {len(filtered_vqas)} auxiliary rows to {filtered_vqas}")
    print(f"Number of cancer rows {len([vqa for vqa in filtered_vqas if vqa['numeric_answer'] == 1])}")

    train_vqas, val_vqas, test_vqas = train_val_test_split_by_pid(filtered_vqas, val_pct=0.1, test_pct=0.1, seed=0)

    with open(train_save_file, "w") as f:
        json.dump(train_vqas, f, indent=4)
    with open(val_save_file, "w") as f:
        json.dump(val_vqas, f, indent=4)
    with open(test_save_file, "w") as f:
        json.dump(test_vqas, f, indent=4)

    print(f"Wrote {len(train_vqas)} auxiliary rows to {train_save_file}")
    print(f"Wrote {len(val_vqas)} auxiliary rows to {val_save_file}")
    print(f"Wrote {len(test_vqas)} auxiliary rows to {test_save_file}")
