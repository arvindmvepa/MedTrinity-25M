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


def summarize_vqa(final_vqa, na_string="NA", nan_string="nan", sep_string="|", print_inst_results=False):
    """
    Produces summary statistics from the final VQA list of dictionaries,
    including the percentage of Code 51 questions.
    """

    # 1) Convert to DataFrame
    df = pd.DataFrame(final_vqa)

    # 2) Overall Statistics
    n_questions = len(df)

    n_init_year0 = (df["init_study_yr"] == 0).sum()
    n_init_year1 = (df["init_study_yr"] == 1).sum()
    n_final_year1 = (df["final_study_yr"] == 1).sum()
    n_final_year2 = (df["final_study_yr"] == 2).sum()
    n_time_delta_1 = (df["time_delta"] == 1).sum()
    n_time_delta_2 = (df["time_delta"] == 2).sum()

    n_nodule_cur_n_nodule_next = ((~df["cur_is_lung_nodule"]) & (~df["next_is_lung_nodule"])).sum()
    n_nodule_cur_nodule_next = ((~df["cur_is_lung_nodule"]) & (df["next_is_lung_nodule"])).sum()
    nodule_cur_n_nodule_next = ((df["cur_is_lung_nodule"]) & (~df["next_is_lung_nodule"])).sum()
    nodule_cur_nodule_next = ((df["cur_is_lung_nodule"]) & (df["next_is_lung_nodule"])).sum()

    # statistics on different question types
    location_counts = df.loc[df['content_type'] == 'location']['answer'].str.split(pat=sep_string).explode().value_counts()
    interval_change_counts = df.loc[df['content_type'] == 'interval_change']['answer'].str.split(pat=sep_string).explode().value_counts()
    interval_growth_counts = df.loc[df['content_type'] == 'interval_growth']['answer'].str.split(pat=sep_string).explode().value_counts()
    further_investigation_counts = df.loc[df['content_type'] == 'further_investigation']['answer'].str.split(pat=sep_string).explode().value_counts()
    margins_counts = df.loc[df['content_type'] == 'margins']['answer'].str.split(pat=sep_string).explode().value_counts()
    predominant_attenuation_counts = df.loc[df['content_type'] == 'predominant_attenuation']['answer'].str.split(pat=sep_string).explode().value_counts()
    longest_diameter_counts = df.loc[df['content_type'] == 'longest_diameter']['answer'].str.split(pat=sep_string).explode().value_counts()
    longest_perpendicular_diameter_counts = df.loc[df['content_type'] == 'longest_perpendicular_diameter']['answer'].str.split(pat=sep_string).explode().value_counts()

    n_pids = df["pid"].nunique()

    print("=== Overall Statistics ===")
    print(f"Total number of questions: {n_questions}")
    print(f"Number of questions with initial year 0: {n_init_year0}")
    print(f"Number of questions with initial year 1: {n_init_year1}")
    print(f"Number of questions with final year 1: {n_final_year1}")
    print(f"Number of questions with final year 2: {n_final_year2}")
    print(f"Number of questions with time delta 1: {n_time_delta_1}")
    print(f"Number of questions with time delta 2: {n_time_delta_2}")

    print(f"Number of questions with no nodule at current and next time points: {n_nodule_cur_n_nodule_next}")
    print(f"Number of questions with no nodule at current but nodule at next time point: {n_nodule_cur_nodule_next}")
    print(f"Number of questions with nodule at current but no nodule at next time point: {nodule_cur_n_nodule_next}")
    print(f"Number of questions with nodule at current and next time points: {nodule_cur_nodule_next}")

    print(f"Value Counts for location_counts: {location_counts}")
    print(f"Value Counts for interval_change_counts: {interval_change_counts}")
    print(f"Value Counts for interval_growth_counts: {interval_growth_counts}")
    print(f"Value Counts for further_investigation_counts: {further_investigation_counts}")
    print(f"Value Counts for margins_counts: {margins_counts}")
    print(f"Value Counts for predominant_attenuation_counts: {predominant_attenuation_counts}")
    print(f"Value Counts for longest_diameter_counts: {longest_diameter_counts}")
    print(f"Value Counts for longest_perpendicular_diameter_counts: {longest_perpendicular_diameter_counts}")

    print(f"Number of unique pids: {n_pids}\n")

    # 3) Per-Institution Statistics
    if print_inst_results:
        df["init_year0_flag"] = df["init_study_yr"] == 0
        df["init_year1_flag"] = df["init_study_yr"] == 1
        df["final_year1_flag"] = df["final_study_yr"] == 1
        df["final_year2_flag"] = df["final_study_yr"] == 2
        df["time_delta1_flag"] = df["time_delta"] == 1
        df["time_delta2_flag"] = df["time_delta"] == 2

        grouped = df.groupby("inst").agg(
            total_questions=("question", "count"),
            total_init_year0=("init_year0_flag", "sum"),
            total_init_year1=("init_year1_flag", "sum"),
            total_final_year1=("final_year1_flag", "sum"),
            total_final_year2=("final_year2_flag", "sum"),
            total_time_delta1=("time_delta1_flag", "sum"),
            total_time_delta2=("time_delta2_flag", "sum"),
            unique_pids=("pid", "nunique")
        ).reset_index()

        # 5) Sort descending by total questions
        grouped_sorted = grouped.sort_values(by="total_questions", ascending=False)

        print("=== Per-Institution Statistics (sorted by most questions) ===")
        # Display as a string table
        print(grouped_sorted.to_string(index=False))

        return grouped_sorted


def build_question(question, answer, pid, init_study_yr, final_study_yr, inst, next_is_lung_nodule, time_delta, embedding_path, question_index, content_type, cur_is_lung_nodule):
    """
    Build a single Q–A dictionary with the relevant fields.
    """
    return {
        "pid": pid,
        "init_study_yr": init_study_yr,
        "final_study_yr": final_study_yr,
        "time_delta": time_delta,
        "inst": inst,
        "next_is_lung_nodule": next_is_lung_nodule,
        "embedding_path": embedding_path,
        "question": question,
        "answer": answer,
        "qid": question_index,
        "content_type": content_type,
        "cur_is_lung_nodule": cur_is_lung_nodule
    }


def get_questions(cur_rows, next_rows, time_delta, pid, init_study_yr, final_study_yr, inst, question_index, embedding_path,
                  na_string="NA", nan_string="nan", sep_string="|"):
    q_list = []

    # initially sort the next_rows by sct_ab_code, then largest nodule to smallest nodule (cur_rows only for determining if there is a current nodule)
    next_rows = next_rows.sort_values(by=["sct_ab_code", "sct_long_dia"],
                            ascending=[False, False],
                            kind="mergesort")
    # only focus on nodule rows
    cur_nodule_rows = cur_rows.loc[cur_rows["sct_ab_code"] == 51]
    next_nodule_rows = next_rows.loc[next_rows["sct_ab_code"] == 51]

    cur_is_lung_nodule = len(cur_nodule_rows) > 0
    next_is_lung_nodule = len(next_nodule_rows) > 0

    # 1) Where will the abnormality be located?
    qa_loc_answer = get_string_from_item_lst(next_nodule_rows, key="sct_epi_loc", key_dict=sct_epi_loc_dict,na_string=na_string)
    qa_loc = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"Where will the predicted nodule(s) epicenter be located after {time_delta} years?",
        answer=qa_loc_answer,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="location"
    )
    q_list.append(qa_loc)
    question_index += 1

    # 2) Will it have a suspicious interval change in attenuation?
    qa_attn_answer = get_string_from_item_lst(next_nodule_rows, key="sct_ab_attn", key_dict=sct_ab_attn_dict, na_string=na_string)
    qa_attn = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"Will there be suspicious interval change in attenuation for the nodule(s) after {time_delta} years?",
        answer=qa_attn_answer,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="interval_change"
    )
    q_list.append(qa_attn)
    question_index += 1

    # 3) Will the abnormality have interval growth?
    qa_gwth_answer = get_string_from_item_lst(next_nodule_rows, key="sct_ab_gwth", key_dict=sct_ab_gwth_dict, na_string=na_string)
    qa_gwth = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"Will the nodule(s) have interval growth after {time_delta} years?",
        answer=qa_gwth_answer,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="interval_growth"
    )
    q_list.append(qa_gwth)
    question_index += 1

    # 4) Will there be an interval change that warrants further investigation?
    qa_invg_answer = get_string_from_item_lst(next_nodule_rows, key="sct_ab_invg", key_dict=sct_ab_invg_dict, na_string=na_string)
    qa_invg = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"Will there be a predicted interval change in the nodule(s) after {time_delta} years that warrants further investigation?",
        answer=qa_invg_answer,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="further_investigation"
    )
    q_list.append(qa_invg)
    question_index += 1

    # 5) What will be the margins?
    qa_margin_answer = get_string_from_item_lst(next_nodule_rows, key="sct_margins", key_dict=sct_margins_dict, na_string=na_string)
    qa_margin = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"What will be the predicted margins for the nodule(s) after {time_delta} years?",
        answer=qa_margin_answer,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="margins"
    )
    q_list.append(qa_margin)
    question_index += 1

    # 6) What will be the predominant attenuation?
    qa_pre_att_answer = get_string_from_item_lst(next_nodule_rows, key="sct_pre_att", key_dict=sct_pre_att_dict, na_string=na_string)
    qa_pre_att = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"What will be the predicted predominant attenuation for the nodule(s) after {time_delta} years?",
        answer=qa_pre_att_answer,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="predominant_attenuation"
    )
    q_list.append(qa_pre_att)
    question_index += 1

    # 7) What will be the longest diameter (in mm)?
    long_dia_str = get_string_from_numeric_lst(next_nodule_rows, key="sct_long_dia", sep_string=sep_string)
    qa_long = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"What will be the predicted longest diameter (mm) for the nodule(s) after {time_delta} years?",
        answer=long_dia_str,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="longest_diameter"
    )
    q_list.append(qa_long)
    question_index += 1

    # 8) What will be the longest perpendicular diameter (in mm)?
    perp_dia_str = get_string_from_numeric_lst(next_nodule_rows, key="sct_perp_dia", sep_string=sep_string)
    qa_perp = build_question(
        pid=pid,
        init_study_yr=init_study_yr,
        final_study_yr=final_study_yr,
        time_delta=time_delta,
        inst=inst,
        question=f"What will be the predicted longest perpendicular diameter (mm) for the nodule(s) after {time_delta} years?",
        answer=perp_dia_str,
        embedding_path=embedding_path,
        next_is_lung_nodule=next_is_lung_nodule,
        cur_is_lung_nodule=cur_is_lung_nodule,
        question_index=question_index,
        content_type="longest_perpendicular_diameter"
    )
    q_list.append(qa_perp)
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

        # create t0 to t1 questions
        embedding_path = os.path.join(embedding_dir, f"pid{pid}_ts0.st")
        print(f"Processing PID {pid} with embedding path: {embedding_path}")
        if os.path.exists(embedding_path):
            qas, question_index = get_questions(pid_study_yr0_ann_df, pid_study_yr1_ann_df, time_delta=1, pid=pid, init_study_yr=0, 
            final_study_yr=1, inst=inst, question_index=question_index, embedding_path=embedding_path)
            all_vqas.extend(qas)
            print(1)
        # create t1 to t2 questions
        embedding_path = os.path.join(embedding_dir, f"pid{pid}_ts1.st")
        if os.path.exists(embedding_path):
            qas, question_index = get_questions(pid_study_yr1_ann_df, pid_study_yr2_ann_df, time_delta=1, pid=pid,init_study_yr=1, 
            final_study_yr=2, inst=inst, question_index=question_index, embedding_path=embedding_path)
            all_vqas.extend(qas)
            print(2)
        # create t0 to t2 questions
        embedding_path = os.path.join(embedding_dir, f"pid{pid}_ts0.st")
        if os.path.exists(embedding_path):
            qas, question_index = get_questions(pid_study_yr0_ann_df, pid_study_yr2_ann_df, time_delta=2, pid=pid, init_study_yr=0, 
            final_study_yr=2, inst=inst, question_index=question_index, embedding_path=embedding_path)
            all_vqas.extend(qas)
            print(3)
    return all_vqas


if __name__ == "__main__":
    measurement_file = "nlst_780_ctab_idc_20210527.csv"
    comparison_file = "nlst_780_ctabc_idc_20210527.csv"
    patient_file = "participant_d100814.sas7bdat"
    add_time_delta2 = True
    tag = "v1_m3fm"

    save_file = f"nlst_vqa_add_time_delta2{add_time_delta2}_{tag}.json"
    train_save_file = f"nlst_train_vqa_delta2{add_time_delta2}_{tag}.json"
    val_save_file = f"nlst_val_vqa_delta2{add_time_delta2}_{tag}.json"
    test_save_file = f"nlst_test_vqa_delta2{add_time_delta2}_{tag}.json"
    pid_split_file = "/home/avepa/Sybil/pid2split.csv"
    embedding_dir = "/hsuraid/avepa/nlst_npy_m3fm"

    measure_df = pd.read_csv(measurement_file)
    compare_df = pd.read_csv(comparison_file)
    combined_measure_comp_df = pd.merge(measure_df, compare_df, on=["pid", "study_yr", "sct_ab_num"], how="inner")
    (patient_df, _) = pyreadstat.read_sas7bdat(patient_file)
    patient_df['pid'] = patient_df['pid'].astype(int)
    patient_info_w_combined_measure_comp_df = pd.merge(patient_df,
                                                       combined_measure_comp_df, on="pid", how="left")
    all_vqas = generate_vqa_from_df(patient_info_w_combined_measure_comp_df, add_time_delta2=add_time_delta2, embedding_dir=embedding_dir)
    with open(save_file, "w") as f:
        json.dump(all_vqas, f, indent=4)
    print(f"==========OVERALL==========")
    print(f"Total VQA pairs generated: {len(all_vqas)}")
    summarize_vqa(all_vqas)
    #with open(save_file, "r") as f:
    #    all_vqas = json.load(f)

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