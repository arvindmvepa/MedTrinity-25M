import itertools, random
import pandas as pd
import json


base_types = (1, 2, 3, 4)
all_combos = [
    sorted(tuple(c))
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


def map_df_cols_to_combo(df):
    # iterate over the rows of the dataframe
    for i, row in df.iterrows():
        # get the combo for the current row
        original_qa_prompt = row["original_qa"][row["original_qa"].index("Q: "):]
        combo = question_type_combo_map[original_qa_prompt]
        df.at[i, "combo"] = str(combo)
    print(f"df['combo'].value_counts() = {df['combo'].value_counts()}")
    return df


def pick_num_question_types_combos_and_rows(df, rng):
    shuffled_combos = all_combos[:]
    rng.shuffle(shuffled_combos)

    used_combos = list()
    qas = []

    for t in base_types:
        for combo in shuffled_combos:
            if t in combo and combo not in used_combos:
                str_combo = str(tuple(combo))
                print(f"Using combo {str_combo} for type {t}")
                row = df[df["combo"] == str_combo].iloc[0]
                df.drop(row.index, inplace=True)
                question = row["transformed_q"]
                answer = row["transformed_a"]
                qas.append((question, answer))
                used_combos.append(combo)
                df.drop(row.index, inplace=True)
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


def unorganize_vqa_data_by_seg_id_and_label_and_type(vqa_data, question_key="volume_file_id", type_key="type",
                                                     label_key="label_name"):
    vqa_data = []
    for seg_id, seg_id_vqa_datum in vqa_data.items():
        for label, label_vqa_datum in seg_id_vqa_datum.items():
            for question_type, vqa_datum in label_vqa_datum.items():
                vqa_data.append(vqa_datum)
    return vqa_data


def generate_updated_vqa_data(vqa_data_dict, df, seed):
    rng = random.Random(seed)
    for label, question_types_vqa_datum in vqa_data_dict.items():
        qas, used_combos = pick_num_question_types_combos_and_rows(df=df, rng=rng)
        # collect all the answers for all the types
        for i, (question_type, vqa_datum) in enumerate(question_types_vqa_datum.items()):
            answer_vqa = vqa_datum["answer_vqa"]
            if vqa_datum == "area":
                area = answer_vqa
            if vqa_datum == "region":
                regions = answer_vqa
            if vqa_datum == "shape":
                shape = answer_vqa
            if vqa_datum == "satellite":
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
    version = f"updated_v0_seed{dataset_seed}"
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
