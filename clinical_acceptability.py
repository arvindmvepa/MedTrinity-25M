import pandas as pd


if __name__ == "__main__":
    user_study_file = "user_study_answered.csv"
    user_study_df = pd.read_csv(user_study_file)

    filt_df = user_study_df.loc[~user_study_df['Model 1 answer sufficient? (Y/N)'].isnull()]
    print(("Model 1 acc ", (filt_df['Model 1 answer sufficient? (Y/N)'] == 'y').mean()))
    print(("Model 2 acc ", (filt_df['Model 2 answer sufficient? (Y/N)'] == 'y').mean()))

    for tag in ['volume', 'region', 'shape', 'spread']:
        tag_filt_df = filt_df.loc[filt_df['question'].str.contains(tag)]
        print((f"Model 1 acc for {tag} ", (tag_filt_df['Model 1 answer sufficient? (Y/N)'] == 'y').mean()))
        print((f"Model 2 acc for {tag} ", (tag_filt_df['Model 2 answer sufficient? (Y/N)'] == 'y').mean()))

