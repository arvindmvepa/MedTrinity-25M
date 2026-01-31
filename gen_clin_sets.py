import os
import json


if __name__ == "__main__":
    gli_file = "brats_gli_3d_vqa_subjTrue_test_updated_v11_seed0.json"
    met_file = "brats_met_3d_vqa_subjTrue_test_updated_v11_seed0.json"
    goat_file = "brats_goat_3d_vqa_subjTrue_test_updated_v11_seed0.json"

    save_gli_file = "brats_gli_3d_vqa_subjTrue_test_updated_v11_clin_subset.json"
    save_met_file = "brats_met_3d_vqa_subjTrue_test_updated_v11_clin_subset.json"
    save_goat_file = "brats_goat_3d_vqa_subjTrue_test_updated_v11_clin_subset.json"
    gli_ims = ['BraTS-GLI-02118-100', 'BraTS-GLI-02128-102', 'BraTS-GLI-02135-101', 'BraTS-GLI-02186-103', 
    'BraTS-GLI-02408-100', 'BraTS-GLI-02416-100', 'BraTS-GLI-02832-100', 'BraTS-GLI-02840-100', 
    'BraTS-GLI-02994-101', 'BraTS-GLI-03023-100']
    met_ims = ['BraTS-MET-00255-000', 'BraTS-MET-00538-000', 'BraTS-MET-00548-000', 'BraTS-MET-00559-003', 
    'BraTS-MET-00592-000', 'BraTS-MET-00630-001', 'BraTS-MET-00655-000', 'BraTS-MET-00696-000', 
    'BraTS-MET-00759-000', 'BraTS-MET-00759-001']
    goat_ims = ['BraTS-GoAT-00063', 'BraTS-GoAT-00132', 'BraTS-GoAT-00591', 'BraTS-GoAT-00734', 
    'BraTS-GoAT-01294', 'BraTS-GoAT-01376', 'BraTS-GoAT-01470', 'BraTS-GoAT-01544', 'BraTS-GoAT-01758', 
    'BraTS-GoAT-02206']
    with open(gli_file, 'r') as f:
        gli_data = json.load(f)
        gli_data = [item for item in gli_data if os.path.basename(item['volume_file_dir']) in gli_ims]
    
    with open(met_file, 'r') as f:
        met_data = json.load(f)
        met_data = [item for item in met_data if os.path.basename(item['volume_file_dir']) in met_ims]
    with open(goat_file, 'r') as f:
        goat_data = json.load(f)
        goat_data = [item for item in goat_data if os.path.basename(item['volume_file_dir']) in goat_ims]    
    
    with open(save_gli_file, 'w') as f:
        json.dump(gli_data, f, indent=2)
    with open(save_met_file, 'w') as f:
        json.dump(met_data, f, indent=2)
    with open(save_goat_file, 'w') as f:
        json.dump(goat_data, f, indent=2)




