import pyreadstat
from collections import defaultdict
import json
from tqdm import tqdm
import pandas as pd
import random
import os


sct_ab_code_dict = {
    51: "Non-calcified nodule or mass (opacity >= 4 mm diameter)",
    52: "Non-calcified micronodule(s) (opacity < 4 mm diameter)",
    53: "Benign lung nodule(s) (benign calcification)",
    54: "Atelectasis, segmental or greater",
    55: "Pleural thickening or effusion",
    56: "Non-calcified hilar/mediastinal adenopathy or mass (>= 10 mm on short axis)",
    57: "Chest wall abnormality",
    58: "Consolidation",
    59: "Emphysema",
    60: "Significant cardiovascular abnormality",
    61: "Reticular/reticulonodular opacities",
    62: "6 or more nodules, not suspicious for cancer (opacity >= 4 mm)",
    63: "Other potentially significant abnormality above the diaphragm",
    64: "Other potentially significant abnormality below the diaphragm",
    65: "Other minor abnormality noted",
    # .M, .N, etc. can be mapped as needed. If numeric codes are stored as strings, adjust keys accordingly
}

sct_epi_loc_dict = {
    1: "Right Upper Lobe",
    2: "Right Middle Lobe",
    3: "Right Lower Lobe",
    4: "Left Upper Lobe",
    5: "Lingula",
    6: "Left Lower Lobe",
    8: "Other (see comments)",
    # .N => "Not Applicable", etc.
}

sct_margins_dict = {
    1: "Spiculated (Stellate)",
    2: "Smooth",
    3: "Poorly defined",
    9: "Unable to determine",
    # .N => "Not applicable", etc.
}

sct_pre_att_dict = {
    1: "Soft Tissue",
    2: "Ground glass",
    3: "Mixed",
    4: "Fluid/water",
    6: "Fat",
    7: "Other",
    9: "Unable to determine"
    # .M => "Missing", .N => "Not applicable", etc.
}

sct_ab_attn_dict = {
    1: "No interval change in attenuation",
    2: "Yes, suspicious change in attenuation",
    9: "Unable to determine"
    # .M => "Missing", .N => "Not applicable", etc.
}

sct_ab_gwth_dict = {
    1: "No interval growth",
    2: "Yes, interval growth",
    9: "Unable to determine"
    # .N => "Not applicable"
}

sct_ab_invg_dict = {
    1: "No further investigation needed",
    2: "Yes, warrants further investigation",
    9: "Unable to determine"
    # .M => "Missing", .N => "Not applicable"
}

sct_ab_preexist_dict = {
    1: "No",
    2: "Yes",
    9: "Unable to determine"
    # .M => "Missing"
}


def get_npy_path(volume_path, img_root="/local/amvepa91/nlst_npy"):
    volume_name = os.path.basename(volume_path)
    time_point_dir = os.path.basename(os.path.dirname(volume_path))
    pid_dir = os.path.basename(os.path.dirname(os.path.dirname(volume_path)))
    volume_path_npy = os.path.join(img_root, pid_dir, time_point_dir, volume_name + ".npy")
    return volume_path_npy


# A small helper to handle "code not found in dict" => "missing"
def get_dict_value(dictionary, key, missing_string="missing"):
    return dictionary.get(key, missing_string)


def get_numeric_value(numeric_value, missing_val=-1):
    if pd.isna(numeric_value):
        return str(missing_val)
    else:
        return str(numeric_value)

    
def get_string_from_item_lst(next_rows, key, key_dict, na_string="NA", sep_string="|"):
    if len(next_rows) == 0:
        return na_string
    return sep_string.join([get_dict_value(key_dict, row[key]) for _, row in next_rows.iterrows()])


def get_string_from_numeric_lst(next_rows, key, nan_string="0", missing_val=-1, sep_string="|"):
    if len(next_rows) == 0:
        return nan_string
    return sep_string.join([get_numeric_value(row[key], missing_val) for _, row in next_rows.iterrows()])


def train_val_test_split_by_pid_split_file(final_vqa, pid_split_file):
    """
    Splits a list of VQA dicts into train, val, and test sets based on a PID split file.
    The PID split file should have columns 'pid' and 'split' with values 'train', 'val', or 'test'.

    - final_vqa: list of dictionaries, each must have 'pid' key
    - pid_split_file: path to CSV file containing PID splits

    Returns: (train_list, val_list, test_list)
    """
    pid_df = pd.read_csv(pid_split_file)
    train_pids = set(pid_df[pid_df['SPLIT'] == 'train']['PID'])
    val_pids = set(pid_df[pid_df['SPLIT'] == 'dev']['PID'])
    test_pids = set(pid_df[pid_df['SPLIT'] == 'test']['PID'])

    train_list = [entry for entry in final_vqa if entry["pid"] in train_pids]
    val_list = [entry for entry in final_vqa if entry["pid"] in val_pids]
    test_list = [entry for entry in final_vqa if entry["pid"] in test_pids]

    return train_list, val_list, test_list


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


def build_question(question, answer, long_topics, long_answers, short_topics, short_answers, pid, inst, embedding_path, question_index, content_type):
    """
    Build a single Q–A dictionary with the relevant fields.
    """
    return {
        "pid": pid,
        "inst": inst,
        "embedding_path": embedding_path,
        "question": question,
        "answer": answer,
        "long_topics": long_topics,
        "long_answers": long_answers,
        "short_topics": short_topics,
        "short_answers": short_answers,
        "qid": question_index,
        "content_type": content_type,
    }


def create_trajectory_question(long_topics, long_answers, short_topics, short_answers, num_timesteps=2):
    if long_topics is not None and short_topics is None:
        question = f"Predict the patient's trajectory of {" and ".join(long_topics)} over the next {num_timesteps} years?"
        answer = f"The predicted trajectory is as follows: {" ; ".join([f"{topic}: {", ".join(answer)}" for topic, answer in zip(long_topics, long_answers)])}."
    elif long_topics is None and short_topics is not None:
        question = f"What will be the eventual patient status for {" and ".join(short_topics)}?"
        answer = f"The patient status will be as follows: {" ; ".join([f"{topic}: {", ".join(answer)}" for topic, answer in zip(short_topics, short_answers)])}."
    else:
        question = f"Predict the patient's trajectory of {" and ".join(long_topics)} over the next {num_timesteps} years and the eventual status of {" and ".join(short_topics)}?"
        answer = f"The predicted trajectory and eventual status are as follows: {" ; ".join([f"{topic}: {", ".join(answer)}" for topic, answer in zip(long_topics + short_topics, long_answers + short_answers)])}."
    return question, answer




def get_questions(rows_ts0, rows_ts1, rows_ts2, pid, inst, question_index, embedding_path, na_string="NA", nan_string="nan", sep_string="|"):
    q_list = []
    # initially sort the next_rows by sct_ab_code, then largest nodule to smallest nodule (cur_rows only for determining if there is a current nodule)
    rows_ts0 = rows_ts0.sort_values(by=["sct_ab_code", "sct_long_dia"],
                            ascending=[False, False],
                            kind="mergesort")
    rows_ts1 = rows_ts1.sort_values(by=["sct_ab_code", "sct_long_dia"],
                            ascending=[False, False],
                            kind="mergesort")
    rows_ts2 = rows_ts2.sort_values(by=["sct_ab_code", "sct_long_dia"],
                            ascending=[False, False],
                            kind="mergesort") 
    # only focus on nodule rows
    rows_ts0 = rows_ts0.loc[rows_ts0["sct_ab_code"] == 51]
    rows_ts1 = rows_ts1.loc[rows_ts1["sct_ab_code"] == 51]
    rows_ts2 = rows_ts2.loc[rows_ts2["sct_ab_code"] == 51]

    ts0_is_lung_nodule = len(rows_ts0) > 0
    ts1_is_lung_nodule = len(rows_ts1) > 0
    ts2_is_lung_nodule = len(rows_ts2) > 0

    cancyr = pid_ann_df['cancyr'].iloc[0]
    has_cancer = False
    if not pd.isna(cancyr):
        has_cancer = True

    qa_margin_answers = [get_string_from_item_lst(rows, key="sct_margins", key_dict=sct_margins_dict, na_string=na_string) for rows in [rows_ts0, rows_ts1, rows_ts2]]
    qa_pre_att_answers = [get_string_from_item_lst(rows, key="sct_pre_att", key_dict=sct_pre_att_dict, na_string=na_string) for rows in [rows_ts0, rows_ts1, rows_ts2]]

    long_topics_answers = ("margins for the nodule", qa_margin_answers), ("predominant attenuation for the nodule", qa_pre_att_answers)
    short_topics_answers = ("cancer", ["yes" if has_cancer else "no"])

    for short_topics_answers_ in [None, short_topics_answers]:
        for long_topics_answers_ in [None, long_topics_answers[0:1], long_topics_answers[1:2], long_topics_answers]:
            if short_topics_answers_ is None and long_topics_answers_ is None:
                continue
            else:
                long_topics_ = [topic for topic, _ in long_topics_answers_] if long_topics_answers_ is not None else None
                short_topics_ = [topic for topic, _ in short_topics_answers_] if short_topics_answers_ is not None else None
                long_answers_ = [answer for _, answer in long_topics_answers_] if long_topics_answers_ is not None else None
                short_answers_ = [answer for _, answer in short_topics_answers_] if short_topics_answers_ is not None else None

                traj_q, traj_a = create_trajectory_question(long_topics=long_topics_, short_topics=short_topics_, long_answers=long_answers_, short_answers=short_answers_)
                q_list.append(build_question(question=traj_q, answer=traj_a, long_topics=long_topics_, long_answers=long_answers_, short_topics=short_topics_, short_answers=short_answers_, pid=pid, inst=inst, embedding_path=embedding_path, question_index=question_index, content_type="trajectory"))
                question_index += 1

    return q_list, question_index


def generate_vqa_from_df(ann_df, add_time_delta2=False, embedding_dir="/hsuraid/avepa/nlst_sybil_embeddings"):
    """
    Main function: iterates over the next_rows of 'df' and
    creates VQA Q–A pairs in a modular way.
    """
    all_vqas = []
    question_index = 0
    for pid, pid_ann_df in tqdm(ann_df.groupby('pid')):
        inst = pid_ann_df['cen'].iloc[0]

        pid_study_yr0_ann_df = pid_ann_df.loc[pid_ann_df["study_yr"] == 0]
        pid_study_yr1_ann_df = pid_ann_df.loc[pid_ann_df["study_yr"] == 1]
        pid_study_yr2_ann_df = pid_ann_df.loc[pid_ann_df["study_yr"] == 2]

        if os.path.exists(embedding_path) and len(pid_study_yr0_ann_df) > 0 and len(pid_study_yr1_ann_df) > 0 and len(pid_study_yr2_ann_df) > 0:
            qas, question_index = get_questions(pid_study_yr0_ann_df, pid_study_yr1_ann_df, pid_study_yr2_ann_df, pid=pid, inst=inst, 
            question_index=question_index, embedding_path=embedding_path)
            all_vqas.extend(qas)
    return all_vqas


def filter_by_instution(all_vqas, inst_list):
    """
    Filter the VQA list by institution.
    """
    filt_inst_list = [qa for qa in all_vqas if qa["inst"] in inst_list]
    return filt_inst_list


if __name__ == "__main__":
    measurement_file = "nlst_780_ctab_idc_20210527.csv"
    comparison_file = "nlst_780_ctabc_idc_20210527.csv"
    patient_file = "participant_d100814.sas7bdat"
    tag = "traj_v0"

    save_file = f"nlst_vqa_add_{tag}.json"
    train_save_file = f"nlst_train_vqa_{tag}.json"
    val_save_file = f"nlst_val_vqa_{tag}.json"
    test_save_file = f"nlst_test_vqa_{tag}.json"
    pid_split_file = "/home/avepa/Sybil/pid2split.csv"
    embedding_dir = "/hsuraid/avepa/nlst_sybil_embeddings"

    measure_df = pd.read_csv(measurement_file)
    compare_df = pd.read_csv(comparison_file)
    combined_measure_comp_df = pd.merge(measure_df, compare_df, on=["pid", "study_yr", "sct_ab_num"], how="inner")
    (patient_df, _) = pyreadstat.read_sas7bdat(patient_file)
    patient_df['pid'] = patient_df['pid'].astype(int)
    patient_info_w_combined_measure_comp_df = pd.merge(patient_df,
                                                       combined_measure_comp_df, on="pid", how="left")
    all_vqas = generate_vqa_from_df(patient_info_w_combined_measure_comp_df, add_time_delta2=add_time_delta2, embedding_dir=embedding_dir)
    print(f"==========OVERALL==========")
    print(f"Total VQA pairs generated: {len(all_vqas)}")
    summarize_vqa(all_vqas)
    with open(save_file, "w") as f:
        json.dump(all_vqas, f, indent=4)

    train_vqas, val_vqas, test_vqas = train_val_test_split_by_pid_split_file(all_vqas, pid_split_file=pid_split_file)


    #print(f"==========TRAIN==========")
    #summarize_vqa(train_vqas)
    #print(f"==========VAL==========")
    #summarize_vqa(val_vqas)
    #print(f"==========TEST==========")
    #summarize_vqa(test_vqas)

    with open(train_save_file, "w") as f:
        json.dump(train_vqas, f, indent=4)
    with open(val_save_file, "w") as f:
        json.dump(val_vqas, f, indent=4)
    with open(test_save_file, "w") as f:
        json.dump(test_vqas, f, indent=4)
