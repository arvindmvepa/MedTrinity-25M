import pandas as pd
import pyreadstat
from collections import defaultdict
import json


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
    9: "Unable to determine",
    # .M => "Missing", .N => "Not applicable", etc.
}

sct_ab_attn_dict = {
    1: "No interval change in attenuation",
    2: "Yes, suspicious change in attenuation",
    9: "Unable to determine",
    # .M => "Missing", .N => "Not applicable", etc.
}

sct_ab_gwth_dict = {
    1: "No interval growth",
    2: "Yes, interval growth",
    9: "Unable to determine",
    # .N => "Not applicable"
}

sct_ab_invg_dict = {
    1: "No further investigation needed",
    2: "Yes, warrants further investigation",
    9: "Unable to determine",
    # .M => "Missing", .N => "Not applicable"
}

sct_ab_preexist_dict = {
    1: "No",
    2: "Yes",
    9: "Unable to determine",
    # .M => "Missing"
}

# A small helper to handle "code not found in dict" => "NA"
def get_dict_value(dictionary, key):
    return dictionary.get(key, "NA")


def postprocess_qas(all_qas):
    """
    1) Group Q–A by (pid, study_yr, sct_ab_code, question).
    2) If sct_ab_code == 51, sort by "largest diameter first" among that group.
    3) Merge answers with commas.
    """

    # 2) We'll store a dictionary so we can group by (pid, study_yr, sct_ab_code, question).
    grouped = defaultdict(list)

    for qa in all_qas:
        # We'll keep the entire Q–A dictionary,
        # so we can retrieve sct_ab_num or any other data in sorting logic.
        pid = qa.get("pid")
        study_yr = qa.get("study_yr")
        code = qa.get("sct_ab_code")
        question = qa.get("question", "")
        # Build the group key
        group_key = (pid, study_yr, code, question)
        grouped[group_key].append(qa)

    # 3) We need a way to figure out "largest diameter" to sort for code=51.
    #    We'll do a quick pass to map each (pid, study_yr, code, sct_ab_num) to a numeric diameter.
    #    Typically, you'd store numeric diameter in the QA dict or parse it from row data.
    #    For minimal example, let's parse from question if it contains "longest diameter", etc.

    # We'll keep a dictionary of (pid, study_yr, code, sct_ab_num) -> diameter
    diam_dict = {}
    for qa in all_qas:
        code = qa.get("sct_ab_code")
        if code == 51:
            question = qa.get("question", "").lower()
            if "longest diameter" in question:
                # Attempt to parse the answer as a float
                try:
                    val = float(qa["answer"])
                except ValueError:
                    val = 0.0
                # Store by (pid, study_yr, code, sct_ab_num)
                key = (qa["pid"], qa["study_yr"], qa["sct_ab_code"], qa["sct_ab_num"])
                diam_dict[key] = val

    # 4) Now build the final output
    final_list = []

    for (pid, study_yr, code, question), qlist in grouped.items():
        if code == 51:
            # Sort by descending diameter. Each QA in qlist has "sct_ab_num".
            # We'll look up diam_dict if present; default to 0.0
            def get_diameter(qa):
                subkey = (qa["pid"], qa["study_yr"], qa["sct_ab_code"], qa["sct_ab_num"])
                return diam_dict.get(subkey, 0.0)

            qlist_sorted = sorted(qlist, key=get_diameter, reverse=True)
        else:
            # For codes != 51, keep the original order
            qlist_sorted = qlist

        # Now combine all answers from that group into one comma-separated string
        combined_answers = ", ".join(qa["answer"] for qa in qlist_sorted)

        # We'll take the "first" QA as a base. Or you could copy minimal fields.
        # We'll just clone it so we have all fields like 'inst', etc.
        base_qa = qlist_sorted[0].copy()
        base_qa["answer"] = combined_answers

        final_list.append(base_qa)

    return final_list


def build_question(row, question, answer):
    """
    Build a single Q–A dictionary with the relevant fields.
    """
    return {
        "pid": row['pid'],
        "study_yr": row['study_yr'],
        "sct_ab_num": row['sct_ab_num'],
        "sct_ab_code": row['sct_ab_code'],
        "inst": row['cen'],
        "question": question,
        "answer": answer
    }


def get_general_questions(row):
    """
    Returns a list of Q–A dictionaries always asked for any sct_ab_code.
    """
    ab_code = row["sct_ab_code"]
    lesion_name = get_dict_value(sct_ab_code_dict, ab_code)

    # Q1: What type of abnormality is this?
    qa1 = build_question(
        row,
        question="What type of abnormality is seen?",
        answer=lesion_name
    )

    # Q2: Was the abnormality pre-existing?
    qa2 = build_question(
        row,
        question=f"Was this {lesion_name} pre-existing?",
        answer=get_dict_value(sct_ab_preexist_dict, row["sct_ab_preexist"])
    )

    return [qa1, qa2]


def get_code51_questions(row):
    """
    Returns a list of Q–A dictionaries relevant only for sct_ab_code == 51.
    If called for code != 51, you can return an empty list or
    handle logic in the caller function.
    """
    ab_code = row["sct_ab_code"]
    # Make sure it's only for code=51
    if ab_code != 51:
        return []

    lesion_name = get_dict_value(sct_ab_code_dict, ab_code)

    # 1) Where is the abnormality located?
    qa_loc = build_question(
        row,
        question=f"Where is the {lesion_name} epicenter located?",
        answer=get_dict_value(sct_epi_loc_dict, row["sct_epi_loc"])
    )

    # 2) Did it have a suspicious interval change in attenuation?
    qa_attn = build_question(
        row,
        question=f"Any suspicious interval change in attenuation for {lesion_name}?",
        answer=get_dict_value(sct_ab_attn_dict, row["sct_ab_attn"])
    )

    # 3) Did the abnormality have interval growth?
    qa_gwth = build_question(
        row,
        question=f"Did the {lesion_name} have interval growth?",
        answer=get_dict_value(sct_ab_gwth_dict, row["sct_ab_gwth"])
    )

    # 4) Does interval change warrant further investigation?
    qa_invg = build_question(
        row,
        question=f"Does the interval change in {lesion_name} warrant further investigation?",
        answer=get_dict_value(sct_ab_invg_dict, row["sct_ab_invg"])
    )

    # 5) What are the margins?
    qa_marg = build_question(
        row,
        question=f"What are the margins for {lesion_name}?",
        answer=get_dict_value(sct_margins_dict, row["sct_margins"])
    )

    # 6) What is the predominant attenuation?
    qa_pre_att = build_question(
        row,
        question=f"What is the predominant attenuation for {lesion_name}?",
        answer=get_dict_value(sct_pre_att_dict, row["sct_pre_att"])
    )

    # 7) What is the longest diameter (in mm)?
    long_dia_str = str(row["sct_long_dia"]) if pd.notnull(row["sct_long_dia"]) else "NA"
    qa_long = build_question(
        row,
        question=f"What is the longest diameter (mm) for {lesion_name}?",
        answer=long_dia_str
    )

    # 8) What is the longest perpendicular diameter (in mm)?
    perp_dia_str = str(row["sct_perp_dia"]) if pd.notnull(row["sct_perp_dia"]) else "NA"
    qa_perp = build_question(
        row,
        question=f"What is the longest perpendicular diameter (mm) for {lesion_name}?",
        answer=perp_dia_str
    )

    return [qa_loc, qa_attn, qa_gwth, qa_invg, qa_marg, qa_pre_att, qa_long, qa_perp]


def generate_vqa_from_df(df):
    """
    Main function: iterates over the rows of 'df' and
    creates VQA Q–A pairs in a modular way.
    """
    all_vqas = []

    for idx, row in df.iterrows():
        # Always-asked questions
        general_qas = get_general_questions(row)
        all_vqas.extend(general_qas)

        # Code-51-specific questions
        code51_qas = get_code51_questions(row)
        all_vqas.extend(code51_qas)

    post_processed_qas = postprocess_qas(all_vqas)

    return post_processed_qas


if __name__ == "__main__":
    measurement_file = "nlst_780_ctab_idc_20210527.csv"
    comparison_file = "nlst_780_ctabc_idc_20210527.csv"
    patient_file = "participant_d100814.sas7bdat"
    save_file = "nlst_vqa.json"

    measure_df = pd.read_csv(measurement_file)
    compare_df = pd.read_csv(comparison_file)
    combined_measure_comp_df = pd.merge(measure_df, compare_df, on=["pid", "study_yr", "sct_ab_num"], how="inner")
    (patient_df, _) = pyreadstat.read_sas7bdat(patient_file)
    combined_measure_comp_w_patient_info_df = pd.merge(combined_measure_comp_df,
                                                       patient_df, on="pid", how="inner")
    all_vqas = generate_vqa_from_df(combined_measure_comp_w_patient_info_df)
    all_pids = {qa["pid"] for qa in all_vqas}
    print(f"Generated {len(all_vqas)} VQA pairs with {len(all_pids)} unique patients.")

    with open(save_file, "w") as f:
        json.dump(all_vqas, f, indent=4)
    """
    for inst in ["BF", "AC", "AP", "AJ", "AX", "AB"]:
        df_inst = combined_measure_comp_w_patient_info_df.loc[combined_measure_comp_w_patient_info_df['cen'] == inst]
        df_inst['pid'] = df_inst['pid'].astype(int)
        print(f"Number of patients in {inst}: {len(df_inst['pid'].unique())}")
        df_inst.to_csv(f"nlst_{inst}.csv", index=False)
    """





