import pandas as pd
import json


if __name__ == "__main__":
    ann_question_file = "valid_questions.csv"
    openai_filt_df_file = "mri_dataset_draft_v1_combined_clean_validity4.csv"

    ann_df = pd.read_csv(ann_question_file)
    openai_df = pd.read_csv(openai_filt_df_file)

    tag = "Reworded Q: Q: "
    strip_fun = lambda x: x[x.find(tag)+len(tag):].strip()
    openai_df['stripped_question'] = openai_df['question'].apply(strip_fun)

    tp, fp, fn, tn = 0, 0, 0, 0
    for idx, row in ann_df.iterrows():
        ann_question = row['question'].strip()
        ann_is_valid = row['valid'].strip() == "Y"
        openai_is_valid = openai_df[openai_df['stripped_question'] == ann_question]['answer'].values[0].strip() == "VALID"

        if ann_is_valid and openai_is_valid:
            tp += 1
        elif ann_is_valid and not openai_is_valid:
            fn += 1
        elif not ann_is_valid and openai_is_valid:
            fp += 1
        else:
            tn += 1
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Accuracy {acc:.f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")