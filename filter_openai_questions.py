import re
import pandas as pd
from pathlib import Path
import re



def extract_original_qa(text):
    """Return everything from the first 'Q:' onward (used unchanged)."""
    idx = text.find("Q:")
    return text[idx:].strip() if idx != -1 else text.strip()


def generate_clean_qa_dataset_from_openai(input_csv, filter_file):
    """
    Reads the OpenAI‑style CSV and returns a DataFrame with
      original_qa | original_q | original_a | transformed_qa | transformed_q | transformed_a
    """
    df_raw = pd.read_csv(input_csv)
    filt_raw = pd.read_csv(filter_file)
    records = []

    for (_, row_raw), (_, row_filt) in zip(df_raw.iterrows(), filt_raw.iterrows()):
        filt_qa_index = row_filt['question'].find("Reworded QA: ") + len("Reworded QA: ")
        filt_qa_string = row_filt['question'][filt_qa_index:]
        if filt_qa_string != row_raw['transformed_qa']:
            raise ValueError("Mismatch between filter file and raw file QA blocks.")
        if row_filt['answer'] == "VALID":
            records.append(row_raw)
    print(f"Filtered {len(df_raw)} rows → {len(records)} valid rows.")

    return pd.DataFrame(records)


if __name__ == "__main__":
    for input_csv, int_filter_file, output_csv in [
        ("int_mri_dataset_draft_v1_combined_clean.csv", "int_mri_dataset_draft_v1_combined_clean_validity.csv", "mri_dataset_draft_v1_combined_filt_clean.csv"),
        ("int_mri_dataset_partially_unknown_combined1_clean.csv", "int_mri_dataset_partially_unknown_combined1_clean_validity1.csv", "mri_dataset_partially_unknown_combined1_filt_clean.csv"),
        ("int_mri_dataset_unknown_clean.csv", "int_mri_dataset_unknown_clean_validity1.csv", "mri_dataset_unknown_filt_clean.csv"),
    ]:
        print(f"Processing {input_csv} with filter {int_filter_file} → {output_csv}")
        df = generate_clean_qa_dataset_from_openai(input_csv, int_filter_file)
        # statistics
        print(df.head())
        df.to_csv(output_csv, index=False)