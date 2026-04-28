import json
from tqdm import tqdm


def build_gt_lookup(vqa_questions):
    gt_lookup = {}
    for entry in vqa_questions:
        pid = entry["pid"]
        embedding_path_ts0 = entry["embedding_path_ts0"]
        embedding_path_ts1 = entry["embedding_path_ts1"]
        embedding_path_ts2 = entry["embedding_path_ts2"]
        answer_vqa_numeric = entry["answer_vqa_numeric"]
        key = pid
        value = (embedding_path_ts0, embedding_path_ts1, embedding_path_ts2, [answer_vqa_numeric])
        if key in gt_lookup:
            gt_lookup[key][3].append(answer_vqa_numeric)
        else:
            gt_lookup[key] = [value]
    return gt_lookup


def build_aux_tasks(all_vqa_questions):
    """
    Convert the original Q&A JSON into
    one row per (volume_seg_file, label_name, type),
    with a fixed set of 4 labels x 4 types = 16 rows per volume_seg_file.
    """
    # 1) Build the ground-truth lookup from the Q&A
    gt_lookup = build_gt_lookup(all_vqa_questions)

    # 2) Identify all seg_files in the data
    pid_set = set(entry["pid"] for entry in all_vqa_questions)

    # 5) Build the final list of rows
    aux_lst = []
    for pid in tqdm(sorted(pid_set)):
            key = pid
            if key in gt_lookup:
                embedding_path_ts0, embedding_path_ts1, embedding_path_ts2, answer_vqa_numeric_lst = gt_lookup[key]
            else:
                continue
            gt_keys = sorted(answer_vqa_numeric_lst[0].keys())
            aux_numeric_dict = {gt_key: 0 for gt_key in gt_keys}
            for answer_vqa_numeric in answer_vqa_numeric_lst:
                for gt_key in gt_keys:
                    if aux_numeric_dict[gt_key] == 0 and answer_vqa_numeric[gt_key] != 0:
                        aux_numeric_dict[gt_key] = answer_vqa_numeric[gt_key]
            aux_dict = {}
            aux_dict = {"pid": pid,
                        "numeric_dict": aux_numeric_dict,
                        "embedding_path_ts0": embedding_path_ts0,
                        "embedding_path_ts1": embedding_path_ts1,
                        "embedding_path_ts2": embedding_path_ts2}
            aux_lst.append(aux_dict)
    return aux_lst


if __name__ == "__main__":
    seed = 0
    tag = "traj_v3"

    save_file = f"nlst_vqa_add_{tag}.json"
    train_file = f"nlst_train_vqa_{tag}_seed{seed}.json"
    val_file = f"nlst_val_vqa_{tag}_seed{seed}.json"
    test_file = f"nlst_test_vqa_{tag}_seed{seed}.json"

    train_aux_file = f"nlst_train_aux_vqa_{tag}_seed{seed}.json"
    val_aux_file = f"nlst_val_aux_vqa_{tag}_seed{seed}.json"
    test_aux_file = f"nlst_test_aux_vqa_{tag}_seed{seed}.json"

    with open(train_file, 'r') as f:
        train_vqa_data = json.load(f)
    with open(val_file, 'r') as f:
        val_vqa_data = json.load(f)
    with open(test_file, 'r') as f:
        test_vqa_data = json.load(f)

    train_vqa_aux_data = build_aux_tasks(train_vqa_data)
    val_vqa_aux_data = build_aux_tasks(val_vqa_data)
    test_vqa_aux_data = build_aux_tasks(test_vqa_data)

    with open(train_aux_file, "w") as f:
        json.dump(train_vqa_aux_data, f, indent=4)
    with open(val_aux_file, "w") as f:
        json.dump(val_vqa_aux_data, f, indent=4)
    with open(test_aux_file, "w") as f:
        json.dump(test_vqa_aux_data, f, indent=4)
