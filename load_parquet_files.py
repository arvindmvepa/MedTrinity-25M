import pandas as pd
import os

def get_df_from_parquet_files(dir_path="/local2/shared_data/MedTrinity-25M-repo/data"):
    # Get a list of all files in the directory that match the pattern
    all_files = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if file.startswith("train-") and file.endswith(".parquet")
    ]

    # Sort files to maintain order
    all_files.sort()

    # Load all parquet files into a single DataFrame
    combined_df = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True)
    return combined_df
