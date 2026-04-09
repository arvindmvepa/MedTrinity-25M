import os
import json
import random
import shutil
import zipfile
from pathlib import Path


if __name__ == "__main__":
    seed = 42
    output_dir = "./clinical_samples"
    zip_filename = "clinical_samples.zip"

    met_dir = "/local2/shared_data/BraTS2024-BraTS-MET/MICCAI-BraTS2024-MET-Challenge-Training_overall"
    goat_dir = "/local2/shared_data/BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth"

    met_ims = ["BraTS-MET-00255-000", "BraTS-MET-00538-000", "BraTS-MET-00548-000", 
    "BraTS-MET-00559-003", "BraTS-MET-00592-000", "BraTS-MET-00630-001", "BraTS-MET-00655-000", 
    "BraTS-MET-00696-000", "BraTS-MET-00759-000", "BraTS-MET-00759-001"]
    
    goat_ims = ["BraTS-GoAT-00063", "BraTS-GoAT-00132", "BraTS-GoAT-00734", "BraTS-GoAT-01294", 
    "BraTS-GoAT-01376", "BraTS-GoAT-01470", "BraTS-GoAT-01544", "BraTS-GoAT-01758", 
    "BraTS-GoAT-02206", "BraTS-GoAT-00591"]

    all_met_ims = sorted(os.listdir(met_dir))
    all_goat_ims = sorted(os.listdir(goat_dir))

    filtered_met_ims = [im for im in all_met_ims if im not in met_ims]
    filtered_goat_ims = [im for im in all_goat_ims if im not in goat_ims]

    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly select 10 samples from each filtered list
    selected_met_ims = random.sample(filtered_met_ims, min(10, len(filtered_met_ims)))
    selected_goat_ims = random.sample(filtered_goat_ims, min(10, len(filtered_goat_ims)))
    
    print(f"Selected MET images: {selected_met_ims}")
    print(f"Selected GoAT images: {selected_goat_ims}")
    
    # Create output directory structure
    met_output = Path(output_dir) / "MET"
    goat_output = Path(output_dir) / "GoAT"
    met_output.mkdir(parents=True, exist_ok=True)
    goat_output.mkdir(parents=True, exist_ok=True)
    
    # Copy MET samples (excluding files containing 'seg')
    for img_name in selected_met_ims:
        src_dir = Path(met_dir) / img_name
        dst_dir = met_output / img_name
        
        if src_dir.exists():
            dst_dir.mkdir(exist_ok=True)
            for file_path in src_dir.glob("*"):
                if file_path.is_file() and "seg" not in file_path.name.lower():
                    shutil.copy2(file_path, dst_dir / file_path.name)
    
    # Copy GoAT samples (excluding files containing 'seg')
    for img_name in selected_goat_ims:
        src_dir = Path(goat_dir) / img_name
        dst_dir = goat_output / img_name
        
        if src_dir.exists():
            dst_dir.mkdir(exist_ok=True)
            for file_path in src_dir.glob("*"):
                if file_path.is_file() and "seg" not in file_path.name.lower():
                    shutil.copy2(file_path, dst_dir / file_path.name)
    
    # Create zip file
    zip_path = Path(output_dir).parent / zip_filename
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(Path(output_dir).parent)
                zipf.write(file_path, arc_path)
    
    print(f"Successfully created zip file: {zip_path}")
    print(f"Total files copied: MET={len(selected_met_ims)}, GoAT={len(selected_goat_ims)}")







