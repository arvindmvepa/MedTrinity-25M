import pandas as pd
from glob import glob
import os
import pyreadstat
from tqdm import tqdm
import torch
import shutil


# load pid2split
df = pd.read_csv("pid2split.csv")
pids = df['PID'].tolist()

root_npy_dir = "/hsuraid/avepa/nlst_npy_v1"
save_dir = f"/hsuraid/avepa/nlst_npy_m3fm"
validate_embeddings = False

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(f"Saving image files for {len(pids)} patients.")

for pid in tqdm(pids):
    patient_dir = os.path.join(root_npy_dir, str(pid))
    if os.path.exists(patient_dir):
        time_points = sorted(glob(os.path.join(patient_dir, "*")))
        for time_index, time_point in enumerate(time_points):
            img_npy_files = sorted(glob(os.path.join(time_point, "*")))
            if not img_npy_files:
                continue
            img_npy_file = img_npy_files[0]
            save_path = os.path.join(save_dir, f"pid{pid}_ts{time_index}.npy")
            shutil.copyfile(img_npy_file, save_path)
