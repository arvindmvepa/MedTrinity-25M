import re
import pandas as pd
from pathlib import Path


QA_BLOCK_RE = re.compile(
    r"Q:.*?(?:\nA:.*?)(?=\nQ:|\Z)",   # one complete “Q: … A: …” block
    flags=re.S | re.M,
)
Q_AND_A_RE = re.compile(
    r"^Q:\s*(.*?)\s*A:\s*(.*)$",      # capture Q and A inside a block
    flags=re.S,
)

def parse_qa_block(block):
    """Return (question, answer) strings stripped of leading/trailing blanks."""
    q, a = Q_AND_A_RE.search(block).groups()
    return q.strip(), a.strip()


def split_transformed_qas(text):
    """
    Return a list of dicts, each with keys
      { 'transformed_qa', 'transformed_q', 'transformed_a' }.
    """
    out = []
    for m in QA_BLOCK_RE.finditer(text):
        block = m.group(0).strip()
        q, a = parse_qa_block(block)
        out.append(
            {
                "transformed_qa": block,
                "transformed_q":  q,
                "transformed_a":  a,
            }
        )
    return out


def extract_original_qa(text):
    """Return everything from the first 'Q:' onward (used unchanged)."""
    idx = text.find("Q:")
    return text[idx:].strip() if idx != -1 else text.strip()


def generate_clean_qa_dataset_from_openai(input_csv, question_col="question", answer_col="answer"):
    """
    Reads the OpenAI‑style CSV and returns a DataFrame with
      original_qa | original_q | original_a | transformed_qa | transformed_q | transformed_a
    """
    df_raw = pd.read_csv(input_csv)
    records = []

    for _, row in df_raw.iterrows():
        # ── original block & split ────────────────────────────────────────────
        original_qa = extract_original_qa(row[question_col])
        original_q, original_a = parse_qa_block(original_qa)

        # ── every transformed variant ────────────────────────────────────────
        for t in split_transformed_qas(row[answer_col]):
            records.append(
                {
                    "original_qa": original_qa,
                    "original_q":  original_q,
                    "original_a":  original_a,
                    **t,                       # merges transformed_* keys
                }
            )

    return pd.DataFrame(records)


if __name__ == "__main__":
    #input_csv = "mri_dataset_draft.csv"
    input_csv = "mri_dataset_draft_v1_combined.csv"
    output_csv = "mri_dataset_draft_v1_combined_clean.csv"
    df_long = generate_clean_qa_dataset_from_openai(input_csv)
    print(f"Parsed {len(df_long):,} rows → {output_csv}")
    print(df_long.head())
    print(df_long.groupby('transformed_qa')['transformed_qa'].nunique())
    print(df_long.groupby('transformed_q')['transformed_q'].nunique())
    print(df_long.groupby('transformed_a')['transformed_a'].nunique())
    df_long.to_csv(output_csv, index=False)
