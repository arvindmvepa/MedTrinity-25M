import pyreadstat
from collections import defaultdict
import json
from tqdm import tqdm
import pandas as pd
import random
import os
import numpy as np


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

sct_margins_dict = {
    1: "Spiculated (Stellate)",
    2: "Smooth",
    3: "Poorly defined",
    9: "Unable to determine",
    # .N => "Not applicable", etc.
}
sct_margins_numeric = {
    "NA": 1,
    "Spiculated (Stellate)": 2,
    "Smooth": 3,
    "Poorly defined": 4,
    "Unable to determine": 5,
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
sct_margins_numeric = {
    "NA": 1,
    "Soft Tissue": 2,
    "Ground glass": 3,
    "Mixed": 4,
    "Fluid/water": 5,
    "Fat": 6,
    "Other": 7,
    "Unable to determine": 8
    # .M => "Missing", .N => "Not applicable", etc.
}
sct_cancer_numeric = { "NA": 1, "yes": 1, "no": 2 }

def generate_train_val_test_split(
    all_vqa_questions,
    question_key="pid",
    seed=0,
    train_pid_ids=None,
    val_pid_ids=None,
    test_pid_ids=None,
    train_frac=0.8,
    val_frac=0.1,
):
    if (
        (train_pid_ids is not None)
        and (val_pid_ids is not None)
        and (test_pid_ids is not None)
    ):
        train_questions = [
            q for q in all_vqa_questions if q[question_key] in train_pid_ids
        ]
        val_questions = [q for q in all_vqa_questions if q[question_key] in val_pid_ids]
        test_questions = [
            q for q in all_vqa_questions if q[question_key] in test_pid_ids
        ]
        train_pids = list({q["pid"] for q in train_questions})
        val_pids = list({q["pid"] for q in val_questions})
        test_pids = list({q["pid"] for q in test_questions})
    else:
        random_state = np.random.RandomState(seed)
        all_pids = sorted(list({q["pid"] for q in all_vqa_questions}))
        random_state.shuffle(all_pids)
        total_pids = len(all_pids)
        train_end = int(total_pids * train_frac)
        val_end = int(total_pids * (train_frac + val_frac))
        train_pids = all_pids[:train_end]
        val_pids = all_pids[train_end:val_end]
        test_pids = all_pids[val_end:]
        train_questions = [q for q in all_vqa_questions if q["pid"] in train_pids]
        val_questions = [q for q in all_vqa_questions if q["pid"] in val_pids]
        test_questions = [q for q in all_vqa_questions if q["pid"] in test_pids]
    print(
        f"Train PIDs: {len(train_pids)}, Val PIDs: {len(val_pids)}, Test PIDs: {len(test_pids)}"
    )
    print(
        f"Train questions: {len(train_questions)}, Val questions: {len(val_questions)}, Test questions: {len(test_questions)}"
    )

    return train_questions, val_questions, test_questions


def train_val_test_pids(pid_split_file):
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

    return train_pids, val_pids, test_pids


# A small helper to handle "code not found in dict" => "missing"
def get_dict_value(dictionary, key, missing_string="missing"):
    return dictionary.get(key, missing_string)


def get_numeric_value(numeric_value, missing_val=-1):
    if pd.isna(numeric_value):
        return str(missing_val)
    else:
        return str(numeric_value)


def get_string_from_item_lst(rows, key, key_dict, na_string="NA"):
    if len(rows) == 0:
        return na_string
    # only return the value for the first row
    return get_dict_value(key_dict, rows.iloc[0][key])


def get_string_from_numeric_lst(rows, key, nan_string="0", missing_val=-1):
    if len(rows) == 0:
        return nan_string
    # only return the value for the first row
    return get_numeric_value(rows.iloc[0][key], missing_val)


def train_val_test_split_by_pid_split_file(final_vqa, pid_split_file):
    """
    Splits a list of VQA dicts into train, val, and test sets based on a PID split file.
    The PID split file should have columns 'pid' and 'split' with values 'train', 'val', or 'test'.

    - final_vqa: list of dictionaries, each must have 'pid' key
    - pid_split_file: path to CSV file containing PID splits

    Returns: (train_list, val_list, test_list)
    """
    pid_df = pd.read_csv(pid_split_file)
    train_pids = set(pid_df[pid_df["SPLIT"] == "train"]["PID"])
    val_pids = set(pid_df[pid_df["SPLIT"] == "dev"]["PID"])
    test_pids = set(pid_df[pid_df["SPLIT"] == "test"]["PID"])

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
    val_pids = set(unique_pids[train_size : train_size + val_size])
    test_pids = set(
        unique_pids[train_size + val_size : train_size + val_size + test_size]
    )

    # 5) Partition the original list by checking pid membership
    train_list = [entry for entry in final_vqa if entry["pid"] in train_pids]
    val_list = [entry for entry in final_vqa if entry["pid"] in val_pids]
    test_list = [entry for entry in final_vqa if entry["pid"] in test_pids]

    return train_list, val_list, test_list


def build_question(
    question,
    answer,
    long_topics,
    long_answers,
    short_topics,
    short_answers,
    numeric_answers,
    question_template,
    answer_template,
    pid,
    inst,
    embedding_path_ts0,
    embedding_path_ts1,
    embedding_path_ts2,
    question_index,
    content_type,
):
    """
    Build a single Q–A dictionary with the relevant fields.
    """
    return {
        "pid": pid,
        "inst": inst,
        "embedding_path_ts0": embedding_path_ts0,
        "embedding_path_ts1": embedding_path_ts1,
        "embedding_path_ts2": embedding_path_ts2,
        "question": question,
        "answer": answer,
        "long_topics": long_topics,
        "long_answers": long_answers,
        "short_topics": short_topics,
        "short_answers": short_answers,
        "numeric_answers": numeric_answers,
        "qid": question_index,
        "content_type": content_type,
        "question_template": question_template,
        "answer_template": answer_template,
    }


def format_long_topic_only_qa(long_topics, long_answers, num_timesteps=2):
    topics_str = " and ".join(long_topics)
    question = f"Predict the patient's trajectory of {topics_str} over the next {num_timesteps} years?"
    question_template = "Predict the patient's trajectory of {} over the next {} years?"

    # Format trajectory answers with clear year labels
    trajectory_parts = []
    for topic, answer_list in zip(long_topics, long_answers):
        # answer_list contains 3 strings (one for each year)
        year_parts = []
        for i, year_answer in enumerate(answer_list):
            year_parts.append(f"Year {i}: {year_answer}")
        trajectory_parts.append(f"{topic} - {', '.join(year_parts)}")
    
    answer = f"The predicted trajectory for {' and '.join(trajectory_parts)}."
    answer_template = "The predicted trajectory for {}."
    return question, answer, question_template, answer_template


def format_short_topic_only_qa(short_topics, short_answers):
    topics_str = " and ".join(short_topics)
    question = f"What will be the eventual patient status for {topics_str}?"
    question_template = "What will be the eventual patient status for {}?"

    # Format short answers as natural statements
    status_parts = []
    for topic, answer_list in zip(short_topics, short_answers):
        answer_value = answer_list[0] if answer_list else "unknown"
        if topic.lower() == "cancer":
            if answer_value.lower() == "yes":
                status_parts.append("the patient will develop cancer")
            elif answer_value.lower() == "no":
                status_parts.append("the patient will not develop cancer")
            else:
                status_parts.append(f"cancer status is {answer_value}")
        else:
            status_parts.append(f"{topic} will be {answer_value}")
    
    answer = f"The eventual patient status: {', '.join(status_parts)}."
    answer_template = "The eventual patient status: {}."
    return question, answer, question_template, answer_template


def format_long_and_short_topic_qa(long_topics, long_answers, short_topics, short_answers, num_timesteps=2):
    long_topics_str = " and ".join(long_topics)
    short_topics_str = " and ".join(short_topics)
    question = f"Predict the patient's trajectory of {long_topics_str} over the next {num_timesteps} years and the eventual status of {short_topics_str}?"
    question_template = "Predict the patient's trajectory of {} over the next {} years and the eventual status of {}?"

    # Combine both trajectory and status formatting
    trajectory_parts = []
    for topic, answer_list in zip(long_topics, long_answers):
        # answer_list contains 3 strings (one for each year)
        year_parts = []
        for i, year_answer in enumerate(answer_list):
            year_parts.append(f"Year {i}: {year_answer}")
        trajectory_parts.append(f"{topic} - {', '.join(year_parts)}")
    
    status_parts = []
    for topic, answer_list in zip(short_topics, short_answers):
        answer_value = answer_list[0] if answer_list else "unknown"
        if topic.lower() == "cancer":
            if answer_value.lower() == "yes":
                status_parts.append("the patient will develop cancer")
            elif answer_value.lower() == "no":
                status_parts.append("the patient will not develop cancer")
            else:
                status_parts.append(f"cancer status is {answer_value}")
        else:
            status_parts.append(f"{topic} will be {answer_value}")
    
    answer = f"The predicted trajectory for {' and '.join(trajectory_parts)}. The eventual status: {', '.join(status_parts)}."
    answer_template = "The predicted trajectory for {}. The eventual status: {}."

    return question, answer, question_template, answer_template


def get_numeric_answer_from_string(long_topics, long_answers, short_topics, short_answers):
    numeric_answers = {}
    if long_topics is None:
        numeric_answers["margins_ts0"] = 0
        numeric_answers["margins_ts1"] = 0
        numeric_answers["margins_ts2"] = 0
        numeric_answers["att_ts0"] = 0
        numeric_answers["att_ts1"] = 0
        numeric_answers["att_ts2"] = 0
    for topic, answer_list in zip(long_topics, long_answers):
        if topic == "margins for the nodule":
            numeric_answers["margins_ts0"] = answer_list[0]
            numeric_answers["margins_ts1"] = answer_list[1]
            numeric_answers["margins_ts2"] = answer_list[2]
        else:
            numeric_answers["margins_ts0"] = 0
            numeric_answers["margins_ts1"] = 0
            numeric_answers["margins_ts2"] = 0
        if topic == "predominant attenuation for the nodule":
            numeric_answers["att_ts0"] = answer_list[0]
            numeric_answers["att_ts1"] = answer_list[1]
            numeric_answers["att_ts2"] = answer_list[2]
        else:
            numeric_answers["att_ts0"] = 0
            numeric_answers["att_ts1"] = 0
            numeric_answers["att_ts2"] = 0
    if short_topics is None:
        numeric_answers["cancer"] = 0
    for topic, answer_list in zip(short_topics, short_answers):
        if topic == "cancer":
            numeric_answers["cancer"] = answer_list[0]
        else:
            numeric_answers["cancer"] = 0
    return numeric_answers


def create_trajectory_question(
    long_topics, long_answers, short_topics, short_answers, num_timesteps=2
):
    if long_topics is not None and short_topics is None:
        return format_long_topic_only_qa(long_topics, long_answers, num_timesteps=num_timesteps)
    elif long_topics is None and short_topics is not None:
        return format_short_topic_only_qa(short_topics, short_answers)
    else:
        return format_long_and_short_topic_qa(long_topics, long_answers, short_topics, short_answers, num_timesteps=num_timesteps)


def get_questions(
    rows_ts0,
    rows_ts1,
    rows_ts2,
    pid,
    inst,
    question_index,
    embedding_path_ts0,
    embedding_path_ts1,
    embedding_path_ts2,
    na_string="NA",
):
    q_list = []

    cancyr = rows_ts0["cancyr"].iloc[0]
    has_cancer = False
    if not pd.isna(cancyr):
        has_cancer = True

    # initially sort the next_rows by sct_ab_code, then largest nodule to smallest nodule (cur_rows only for determining if there is a current nodule)
    rows_ts0 = rows_ts0.sort_values(
        by=["sct_ab_desc", "sct_long_dia"], ascending=[False, False], kind="mergesort"
    )
    rows_ts1 = rows_ts1.sort_values(
        by=["sct_ab_desc", "sct_long_dia"], ascending=[False, False], kind="mergesort"
    )
    rows_ts2 = rows_ts2.sort_values(
        by=["sct_ab_desc", "sct_long_dia"], ascending=[False, False], kind="mergesort"
    )
    # only focus on nodule rows
    rows_ts0 = rows_ts0.loc[rows_ts0["sct_ab_desc"] == 51]
    rows_ts1 = rows_ts1.loc[rows_ts1["sct_ab_desc"] == 51]
    rows_ts2 = rows_ts2.loc[rows_ts2["sct_ab_desc"] == 51]

    qa_margin_answers = [
        get_string_from_item_lst(
            rows, key="sct_margins", key_dict=sct_margins_dict, na_string=na_string
        )
        for rows in [rows_ts0, rows_ts1, rows_ts2]
    ]
    qa_pre_att_answers = [
        get_string_from_item_lst(
            rows, key="sct_pre_att", key_dict=sct_pre_att_dict, na_string=na_string
        )
        for rows in [rows_ts0, rows_ts1, rows_ts2]
    ]

    long_topics_answers = [("margins for the nodule", qa_margin_answers), 
                           ("predominant attenuation for the nodule", qa_pre_att_answers)]
    short_topics_answers = short_topics_answers = [("cancer", ["yes" if has_cancer else "no"])]

    for short_topics_answers_ in [None, short_topics_answers]:
        for long_topics_answers_ in [
            None,
            long_topics_answers[0:1],
            long_topics_answers[1:2],
            long_topics_answers,
        ]:
            if short_topics_answers_ is None and long_topics_answers_ is None:
                continue
            else:
                long_topics_ = (
                    [topic for topic, _ in long_topics_answers_]
                    if long_topics_answers_ is not None
                    else None
                )
                short_topics_ = (
                    [topic for topic, _ in short_topics_answers_]
                    if short_topics_answers_ is not None
                    else None
                )
                long_answers_ = (
                    [answer for _, answer in long_topics_answers_]
                    if long_topics_answers_ is not None
                    else None
                )
                short_answers_ = (
                    [answer for _, answer in short_topics_answers_]
                    if short_topics_answers_ is not None
                    else None
                )

                numeric_answers = get_numeric_answer_from_string(
                    long_topics=long_topics_,
                    long_answers=long_answers_,
                    short_topics=short_topics_,
                    short_answers=short_answers_
                )

                traj_q, traj_a, templ_q, templ_a = create_trajectory_question(
                    long_topics=long_topics_,
                    short_topics=short_topics_,
                    long_answers=long_answers_,
                    short_answers=short_answers_,
                )
                q_list.append(
                    build_question(
                        question=traj_q,
                        answer=traj_a,
                        question_template=templ_q,
                        answer_template=templ_a,
                        long_topics=long_topics_,
                        long_answers=long_answers_,
                        short_topics=short_topics_,
                        short_answers=short_answers_,
                        numeric_answers=numeric_answers,
                        pid=pid,
                        inst=inst,
                        embedding_path_ts0=embedding_path_ts0,
                        embedding_path_ts1=embedding_path_ts1,
                        embedding_path_ts2=embedding_path_ts2,
                        question_index=question_index,
                        content_type="trajectory",
                    )
                )
                question_index += 1

    return q_list, question_index


def generate_vqa_from_df(ann_df, train_pids, val_pids, test_pids, embedding_dir="/hsuraid/avepa/m3fm_embeddings"):
    """
    Main function: iterates over the next_rows of 'df' and
    creates VQA Q–A pairs in a modular way.
    """
    all_vqas = []
    train_vqas = []
    val_vqas = []
    test_vqas = []
    question_index = 0
    valid_df_count = 0
    print("Number of unique pids with timepoint 0: ", ann_df.loc[ann_df['study_yr'] == 0]['pid'].nunique())
    print("Number of unique pids with timepoint 1: ", ann_df.loc[ann_df['study_yr'] == 1]['pid'].nunique())
    print("Number of unique pids with timepoint 2: ", ann_df.loc[ann_df['study_yr'] == 2]['pid'].nunique())
    split = None
    for pid, pid_ann_df in tqdm(ann_df.groupby("pid")):
        if pid in train_pids:
            embedding_pid_dir = os.path.join(embedding_dir, "train")
            split = "train"
        elif pid in val_pids:
            embedding_pid_dir = os.path.join(embedding_dir, "val")
            split = "val"
        elif pid in test_pids:
            embedding_pid_dir = os.path.join(embedding_dir, "test")
            split = "test"
        else:
            split = None
            continue
        inst = pid_ann_df["cen"].iloc[0]

        pid_study_yr0_ann_df = pid_ann_df.loc[pid_ann_df["study_yr"] == 0]
        pid_study_yr1_ann_df = pid_ann_df.loc[pid_ann_df["study_yr"] == 1]
        pid_study_yr2_ann_df = pid_ann_df.loc[pid_ann_df["study_yr"] == 2]

        embedding_path_ts0 = os.path.join(embedding_pid_dir, f"pid{pid}_ts0.st")
        embedding_path_ts1 = os.path.join(embedding_pid_dir, f"pid{pid}_ts1.st")
        embedding_path_ts2 = os.path.join(embedding_pid_dir, f"pid{pid}_ts2.st")

        if len(pid_study_yr0_ann_df) > 0 and len(pid_study_yr1_ann_df) > 0 and len(pid_study_yr2_ann_df) > 0 and \
            os.path.exists(embedding_path_ts0):
            qas, question_index = get_questions(
                pid_study_yr0_ann_df,
                pid_study_yr1_ann_df,
                pid_study_yr2_ann_df,
                pid=pid,
                inst=inst,
                question_index=question_index,
                embedding_path_ts0=embedding_path_ts0,
                embedding_path_ts1=embedding_path_ts1,
                embedding_path_ts2=embedding_path_ts2,
            )
            all_vqas.extend(qas)
            if split == "train":
                train_vqas.extend(qas)
            elif split == "val":
                val_vqas.extend(qas)
            elif split == "test":
                test_vqas.extend(qas)
            else:
                raise ValueError(f"split is missing! for pid {pid}")
            valid_df_count += 1
        #else:
        #    print(f"PID {pid} does not have all 3 time points. {pid_ann_df['study_yr'].tolist()}")
    print(f"Total valid PIDs with all 3 time points: {valid_df_count}")
    return all_vqas, train_vqas, val_vqas, test_vqas


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
    seed = 0
    tag = "traj_v3"

    save_file = f"nlst_vqa_add_{tag}.json"
    train_save_file = f"nlst_train_vqa_{tag}_seed{seed}.json"
    val_save_file = f"nlst_val_vqa_{tag}_seed{seed}.json"
    test_save_file = f"nlst_test_vqa_{tag}_seed{seed}.json"
    pid_split_file = "/home/avepa/Sybil/pid2split.csv"
    embedding_dir = "/hsuraid/avepa/m3fm_embeddings"

    measure_df = pd.read_csv(measurement_file)
    print(f"Number of unique pids in measurement file: {measure_df['pid'].nunique()}")
    print(f"Number of unique study years in measurement file: {measure_df['study_yr'].nunique()}")
    (patient_df, _) = pyreadstat.read_sas7bdat(patient_file)
    print(f"Number of unique pids in patient file: {patient_df['pid'].nunique()}")
    patient_df["pid"] = patient_df["pid"].astype(int)
    patient_info_w_measure_df = pd.merge(
        patient_df, measure_df, on="pid", how="left"
    )
    patient_info_w_measure_df_ = pd.merge(
        patient_df, measure_df, on="pid", how="inner"
    )
    print(f"Number of unique pids in merged patient df and measurement df: {patient_info_w_measure_df_['pid'].nunique()}")
    train_pids, val_pids, test_pids = train_val_test_pids(pid_split_file)
    all_vqas, train_vqas, val_vqas, test_vqas = generate_vqa_from_df(
        patient_info_w_measure_df, embedding_dir=embedding_dir, train_pids=train_pids, val_pids=val_pids, test_pids=test_pids
    )
    print(f"==========OVERALL==========")
    print(f"Total VQA pairs generated: {len(all_vqas)}")
    with open(save_file, "w") as f:
        json.dump(all_vqas, f, indent=4)
    
    print(f"VQA pairs generated for train {len(train_vqas)}, val {len(val_vqas)}, and test {len(test_vqas)}")

    with open(train_save_file, "w") as f:
        json.dump(train_vqas, f, indent=4)
    with open(val_save_file, "w") as f:
        json.dump(val_vqas, f, indent=4)
    with open(test_save_file, "w") as f:
        json.dump(test_vqas, f, indent=4)
