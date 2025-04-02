import pandas as pd
import pyreadstat


if __name__ == "__main__":
    measurement_file = "nlst_780_ctab_idc_20210527.csv"
    comparison_file = "nlst_780_ctabc_idc_20210527.csv"
    patient_file = "participant_d100814.sas7bdat"

    measure_df = pd.read_csv(measurement_file)
    compare_df = pd.read_csv(comparison_file)
    combined_measure_comp_df = pd.merge(measure_df, compare_df, on=["pid", "study_yr", "sct_ab_num"], how="inner")
    (patient_df, _) = pyreadstat.read_sas7bdat(patient_file)
    combined_measure_comp_w_patient_info_df = pd.merge(combined_measure_comp_df,
                                                       patient_df, on="pid", how="inner")
    for inst in ["BF", "AC", "AP", "AJ", "AX", "AB"]:
        df_inst = combined_measure_comp_w_patient_info_df.loc[combined_measure_comp_w_patient_info_df['cen'] == inst]
        df_inst['pid'] = df_inst['pid'].astype(int)
        print(f"Number of patients in {inst}: {len(df_inst['pid'].unique())}")
        df_inst.to_csv(f"nlst_{inst}.csv", index=False)





