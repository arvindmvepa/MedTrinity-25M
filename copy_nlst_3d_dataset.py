import pandas as pd
from glob import glob
import os
from tqdm import tqdm

df = pd.read_csv("pid2split.csv")
pids = df['PID'].tolist()

root_npy_dir = "/hsuraid/avepa/nlst_npy_v1"
save_dir = "/hsuraid/avepa/nlst_npy_m3fm"

os.makedirs(save_dir, exist_ok=True)

for pid in tqdm(pids):
    patient_dir = os.path.join(root_npy_dir, str(pid))
    if not os.path.exists(patient_dir):
        continue

    time_points = sorted(glob(os.path.join(patient_dir, "*")))
    for time_index, time_point in enumerate(time_points):
        img_npy_files = sorted(glob(os.path.join(time_point, "*")))
        if not img_npy_files:
            continue
        src = img_npy_files[0]
        dst = os.path.join(save_dir, f"pid{pid}_ts{time_index}.npy")

        if not os.path.exists(dst):
            os.symlink(src, dst)