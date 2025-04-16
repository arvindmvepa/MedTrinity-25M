import itertools, random
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path


base_types = [1, 2, 3, 4]
all_combos = [
    tuple(sorted(c))
    for r in range(1, len(base_types) + 1)
    for c in itertools.combinations(base_types, r)
]

combo_question_type_map = {(1,): "Q: How large is the volume covered by {label}? A: The overall volume of {label} is {area}.",
                           (2,): "Q: Which region(s) of the brain is {label} located in? A: The {label} is located in {regions}.",
                           (3,): "Q: What is the shape of {label}? A: The shape of {label} is {shape}.",
                           (4,): "Q: How spread out is {label}? A: The spread of {label} is {satellite}.",
                           (1, 2): "Q: How large is the volume of {label} and where is it located? A: The overall volume of {label} is {area}, and it is located in {regions}.",
                           (1, 3): "Q: How large is the volume of {label} and what is its shape? A: The overall volume of {label} is {area}, and its shape is described as {shape}.",
                           (1, 4): "Q: How large is the volume of {label} and how spread out is it? A: The overall volume of {label} is {area}, and it is characterized as {satellite}.",
                           (2, 3): "Q: In which region is {label} and what is its shape? A: The {label} is located in {regions}, and its shape is described as {shape}.",
                           (2, 4): "Q: In which region is {label} and how spread out is it? A: The {label} is located in {regions}, and it is characterized as {satellite}.",
                           (3, 4): "Q: What is the shape of {label} and how spread out is it? A: The shape of {label} is described as {shape}, and it is characterized as {satellite}.",
                           (1, 2, 3): "Q: What is the volume, region, and shape of {label}? A: The overall volume of {label} is {area}, it is located in {regions}, and its shape is described as {shape}.",
                           (1, 2, 4): "Q: What is the volume, region, and spread of {label}? A: The overall volume of {label} is {area}, it is located in {regions}, and it is characterized as {satellite}.",
                           (1, 3, 4): "Q: What is the volume, shape, and spread of {label}? A: The overall volume of {label} is {area}, its shape is described as {shape}, and it is characterized as {satellite}.",
                           (2, 3, 4): "Q: What is the region, shape, and spread of {label}? A: The {label} is located in {regions}, its shape is described as {shape}, and it is characterized as {satellite}.",
                           (1, 2, 3, 4): "Q: What is the volume, region, shape, and spread of {label}? A: The overall volume of {label} is {area}, it is located in {regions}, its shape is described as {shape}, and it is characterized as {satellite}."}
question_type_combo_map = {v: k for k, v in combo_question_type_map.items()}


def validate_vqa_lists(vqa_list, save_dir=None):
    """
    Compute and (optionally) persist statistics that sanity‑check your VQA data.

    Parameters
    ----------
    vqa_list : list[dict]
        Each element must contain at least these keys
            question, answer, label_name, type, combo
        where
            question : str   – rendered "Q: …"
            answer   : str   – rendered "A: …"
            label_name : str – e.g. "Enhancing"
            type       : str – one of {"area","region","shape","satellite"}
            combo      : tuple[int] – e.g. (1, 3, 4)
    save_dir : str | Path | None, default None
        If provided, CSV versions of the tables are written there.

    Returns
    -------
    dict[str, pd.DataFrame]  – the six summary tables for further inspection.
    """
    QUESTION_COL = "question"
    ANSWER_COL = "answer"
    LABEL_COL = "label_name"
    TYPE_COL = "type"
    COMBO_COL = "combo"

    required_cols = {QUESTION_COL, ANSWER_COL, LABEL_COL, TYPE_COL, COMBO_COL}

    df = pd.DataFrame(vqa_list)
    print(f"Loaded {len(df):,} rows  ({len(vqa_list):,})")

    # Basic schema check
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Each VQA dict must include {sorted(required_cols)}. Missing: {missing}")

    df[COMBO_COL] = df[COMBO_COL].apply(
        lambda c: tuple(c) if isinstance(c, list) else c
    )
    combo_overall = (df[COMBO_COL]
          .value_counts()
          .rename_axis("combo")
          .reset_index(name="n_questions")
          .sort_values("combo", key=lambda s: s.apply(str))
    )

    combo_per_label = (
        df.groupby([LABEL_COL, COMBO_COL])
          .size()
          .rename("n_questions")
          .reset_index()
    )

    combo_per_label_type = (
        df.groupby([LABEL_COL, TYPE_COL, COMBO_COL])
          .size()
          .rename("n_questions")
          .reset_index()
    )

    # ─────────────────────────────────────────────────────────────
    # 4.  Unique‑question / unique‑answer counts
    # ─────────────────────────────────────────────────────────────
    unique_q_overall = df[QUESTION_COL].nunique()
    unique_a_overall = df[ANSWER_COL].nunique()

    unique_q_per_label = (
        df.groupby(LABEL_COL)[QUESTION_COL]
          .nunique()
          .rename("n_unique_questions")
          .reset_index()
    )

    unique_a_per_label = (
        df.groupby(LABEL_COL)[ANSWER_COL]
          .nunique()
          .rename("n_unique_answers")
          .reset_index()
    )

    unique_q_per_label_type = (
        df.groupby([LABEL_COL, TYPE_COL])[QUESTION_COL]
          .nunique()
          .rename("n_unique_questions")
          .reset_index()
    )

    unique_a_per_label_type = (
        df.groupby([LABEL_COL, TYPE_COL])[ANSWER_COL]
          .nunique()
          .rename("n_unique_answers")
          .reset_index()
    )

    print("\n▶ Combo distribution (overall)")
    print(combo_overall.to_string(index=False))

    print("\n▶ Unique Q/A counts (overall)")
    print(f"   • questions : {unique_q_overall:,}")
    print(f"   • answers   : {unique_a_overall:,}")

    # ─────────────────────────────────────────────────────────────
    # 6.  Optional CSV export
    # ─────────────────────────────────────────────────────────────
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        combo_overall.to_csv(save_dir / "combo_overall.csv", index=False)
        combo_per_label.to_csv(save_dir / "combo_per_label.csv", index=False)
        combo_per_label_type.to_csv(save_dir / "combo_per_label_type.csv", index=False)
        unique_q_per_label.to_csv(save_dir / "unique_q_per_label.csv", index=False)
        unique_a_per_label.to_csv(save_dir / "unique_a_per_label.csv", index=False)
        unique_q_per_label_type.to_csv(save_dir / "unique_q_per_label_type.csv", index=False)
        unique_a_per_label_type.to_csv(save_dir / "unique_a_per_label_type.csv", index=False)

        print(f"\nCSV tables written to → {save_dir.resolve()}")

    # ─────────────────────────────────────────────────────────────
    # 7.  Return tables for programmatic inspection
    # ─────────────────────────────────────────────────────────────
    return {
        "combo_overall": combo_overall,
        "combo_per_label": combo_per_label,
        "combo_per_label_type": combo_per_label_type,
        "unique_q_per_label": unique_q_per_label,
        "unique_a_per_label": unique_a_per_label,
        "unique_q_per_label_type": unique_q_per_label_type,
        "unique_a_per_label_type": unique_a_per_label_type,
    }


def map_df_cols_to_combo(df):
    # iterate over the rows of the dataframe
    for i, row in df.iterrows():
        # get the combo for the current row
        original_qa_prompt = row["original_qa"][row["original_qa"].index("Q: "):]
        combo = question_type_combo_map[original_qa_prompt]
        df.at[i, "combo"] = str(combo)
    return df


def pick_num_question_types_combos_and_rows(df, rng):
    shuffled_base_types = base_types[:]
    rng.shuffle(shuffled_base_types)
    shuffled_combos = all_combos[:]
    rng.shuffle(shuffled_combos)

    used_combos = list()
    qas = []

    for t in shuffled_base_types:
        for combo in shuffled_combos:
            str_combo = str(tuple(combo))
            filt_df = df[df["combo"] == str_combo]
            if (t in combo) and (combo not in used_combos) and (len(filt_df) > 0):
                row = filt_df.iloc[0]
                question = row["transformed_q"]
                answer = row["transformed_a"]
                qas.append((question, answer))
                used_combos.append(combo)
                row_idx = row.name
                df.drop(row_idx, inplace=True)
                break
        else:
            raise ValueError(
                f"No available question containing type {t} for this pair."
            )

    return qas, used_combos


def organize_vqa_data_by_seg_id_and_label_and_type(vqa_data, question_key="volume_file_id", type_key="type",
                                                   label_key="label_name"):
    vqa_data_dict = dict()
    for vqa_datum in vqa_data:
        seg_id = vqa_datum[question_key]
        label = vqa_datum[label_key]
        question_type = vqa_datum[type_key]
        if seg_id not in vqa_data_dict:
            vqa_data_dict[seg_id] = {}
        if label not in vqa_data_dict[seg_id]:
            vqa_data_dict[seg_id][label] = {}
        vqa_data_dict[seg_id][label][question_type] = vqa_datum
    return vqa_data_dict


def unorganize_vqa_data_by_seg_id_and_label_and_type(vqa_data):
    vqa_data_list = []
    for seg_id, seg_id_vqa_datum in vqa_data.items():
        for label, label_vqa_datum in seg_id_vqa_datum.items():
            for question_type, vqa_datum in label_vqa_datum.items():
                vqa_data_list.append(vqa_datum)
    return vqa_data_list


def generate_updated_vqa_data(vqa_data_dict, df, seed):
    rng = random.Random(seed)
    for seg_id, labels_question_types_vqa_datum in tqdm(vqa_data_dict.items()):
        for label, question_types_vqa_datum in labels_question_types_vqa_datum.items():
            qas, used_combos = pick_num_question_types_combos_and_rows(df=df, rng=rng)
            # collect all the answers for all the types
            for i, (question_type, vqa_datum) in enumerate(question_types_vqa_datum.items()):
                answer_vqa = vqa_datum["answer_vqa"]
                if question_type == "area":
                    area = answer_vqa
                if question_type == "region":
                    regions = answer_vqa
                if question_type == "shape":
                    shape = answer_vqa
                if question_type == "satellite":
                    satellite = answer_vqa
            for i, (question_type, vqa_datum) in enumerate(question_types_vqa_datum.items()):
                question, answer = qas[i]
                question = question.replace("{label}", label)
                answer = answer.replace("{label}", label)
                new_answer_vqa = []
                if "{area}" in answer:
                    new_answer_vqa = new_answer_vqa + [area]
                    area_str = area[0]
                    answer = answer.replace("{area}", area_str)
                if "{regions}" in answer:
                    new_answer_vqa = new_answer_vqa + [regions]
                    region_str = regions[0]
                    answer = answer.replace("{regions}", region_str)
                if "{shape}" in answer:
                    new_answer_vqa = new_answer_vqa + [shape]
                    shape_str = shape[0]
                    answer = answer.replace("{shape}", shape_str)
                if "{satellite}" in answer:
                    new_answer_vqa = new_answer_vqa + [satellite]
                    satellite_str = satellite[0]
                    answer = answer.replace("{satellite}", satellite_str)
                vqa_datum["question"] = question
                vqa_datum["answer"] = answer
                vqa_datum["answer_vqa"] = new_answer_vqa
                vqa_datum["answer_gen"] = answer
                vqa_datum["combo"] = used_combos[i]
    return vqa_data_dict


if __name__ == "__main__":
    ref_train_vqa_file = "brats_{}_3d_vqa_subj{}_train_{}.json"
    ref_val_vqa_file = "brats_{}_3d_vqa_subj{}_val_{}.json"
    ref_test_vqa_file = "brats_{}_3d_vqa_subj{}_test_{}.json"

    train_vqa_file = "brats_{}_3d_vqa_subj{}_train_{}_new.json"
    val_vqa_file = "brats_{}_3d_vqa_subj{}_val_{}_new.json"
    test_vqa_file = "brats_{}_3d_vqa_subj{}_test_{}_new.json"

    openai_df_file = "mri_dataset_draft_v1_combined_clean.csv"
    # rest of the parameters
    subjective_only = True
    dataset_seed = 0
    new_dataset_seed = 0

    # GLI dataset settings
    dataset_type = "gli"
    version = f"updated_v2_seed{dataset_seed}"
    labels_order = (1, 2, 3, 4)
    pediatric = False
    goat = False

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
        ref_train_vqa_data_dict = organize_vqa_data_by_seg_id_and_label_and_type(ref_train_vqa_data)
    with open(ref_val_vqa_file, 'r') as f:
        ref_val_vqa_data = json.load(f)
        ref_val_vqa_data_dict = organize_vqa_data_by_seg_id_and_label_and_type(ref_val_vqa_data)
    with open(ref_test_vqa_file, 'r') as f:
        ref_test_vqa_data = json.load(f)
        ref_test_vqa_data_dict = organize_vqa_data_by_seg_id_and_label_and_type(ref_test_vqa_data)

    # read the openai_df_file
    openai_df = pd.read_csv(openai_df_file, header=0)
    openai_df = map_df_cols_to_combo(openai_df)
    openai_df = openai_df.sample(frac=1, random_state=new_dataset_seed)

    # generate updated vqa dataset
    train_vqa_data_dict = generate_updated_vqa_data(ref_train_vqa_data_dict, openai_df, seed=new_dataset_seed)
    train_vqa = unorganize_vqa_data_by_seg_id_and_label_and_type(train_vqa_data_dict)
    val_vqa_data_dict = generate_updated_vqa_data(ref_val_vqa_data_dict, openai_df, seed=new_dataset_seed)
    val_vqa = unorganize_vqa_data_by_seg_id_and_label_and_type(val_vqa_data_dict)
    test_vqa_data_dict = generate_updated_vqa_data(ref_test_vqa_data_dict, openai_df, seed=new_dataset_seed)
    test_vqa = unorganize_vqa_data_by_seg_id_and_label_and_type(test_vqa_data_dict)

    # TODO: Add some validation to ensure the performance is good (check for question duplicates, frequency of combo/question types, etc.).
    # Can use ChatGPT to help

    # save the updated vqa dataset
    with open(train_vqa_file, 'w') as f:
        json.dump(train_vqa, f, indent=2)
    with open(val_vqa_file, 'w') as f:
        json.dump(val_vqa, f, indent=2)
    with open(test_vqa_file, 'w') as f:
        json.dump(test_vqa, f, indent=2)
