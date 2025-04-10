import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
import os


def load_labels(label_txt_path):
    label_map = {}
    with open(label_txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                idx, name = parts
                label_map[int(idx)] = name
    return label_map


def compute_overlap(tumour_img, atlas_img):
    tumour_data = tumour_img.get_fdata() > 0
    atlas_data = atlas_img.get_fdata().astype(int)[..., 0]  # Assuming the atlas is a single channel
    assert atlas_data.shape == tumour_data.shape, "Atlas and tumour images must have the same shape."
    print(f"min(atlas_data): {np.min(atlas_data)}, max(atlas_data): {np.max(atlas_data)}")
    print(f"total atlas_data: {np.sum(atlas_data > 0)}")

    # Only look at tumour voxels
    overlapped_labels = atlas_data[tumour_data]

    print(f"overlapping labels: {overlapped_labels.shape}")
    print(f"(non-zero) overlapping labels: {np.sum(overlapped_labels > 0)}")

    print("nonzero tumor data: ", np.nonzero(tumour_data))
    print("nonzero atlas data: ", np.nonzero(atlas_data))

    unique, counts = np.unique(overlapped_labels[overlapped_labels > 0], return_counts=True)
    total_voxels = np.sum(tumour_data)

    return unique, counts, total_voxels


def main(seg, atlas, labels, out):
    # Load images
    tumour_img = nib.load(seg)
    atlas_img = nib.load(atlas)

    # Resample atlas to tumour space if needed
    if atlas_img.shape != tumour_img.shape or not np.allclose(atlas_img.affine, tumour_img.affine):
        print("Resampling atlas to match tumour mask...")
        atlas_img = resample_to_img(atlas_img, tumour_img, interpolation='nearest')

    # Load label map
    label_map = load_labels(labels)

    # Compute overlaps
    labels, counts, total = compute_overlap(tumour_img, atlas_img)


    print(f"\nTotal tumour voxels: {total}\n")
    print("Overlapping anatomical regions (LPBA40):")
    for lbl, cnt in zip(labels, counts):
        name = label_map.get(lbl, f"Unknown ({lbl})")
        percent = (cnt / total) * 100
        print(f"  {lbl:>3}: {name:<35} {cnt:>5} voxels  ({percent:5.2f}%)")

    # Optionally: output to CSV
    if out:
        df = pd.DataFrame({
            "Label": labels,
            "Region": [label_map.get(l, f"Unknown ({l})") for l in labels],
            "VoxelCount": counts,
            "Percentage": (counts / total) * 100
        })
        df.to_csv(out, index=False)
        print(f"\nSaved report to {out}")


if __name__ == "__main__":
    seg = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-00005-100/BraTS-GLI-00005-100-seg.nii.gz"
    #atlas = "/local2/amvepa91/sri24/lpba40.nii"
    atlas = "/local2/amvepa91/sri24/tzo116plus.nii"
    labels = "/local2/amvepa91/sri24/SRI24-tzo116plus.txt"
    out = f"./{os.path.basename(seg)}_report.csv"
    main(seg=seg, atlas=atlas, labels=labels, out=out)
