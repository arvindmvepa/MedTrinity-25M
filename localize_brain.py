import os
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img, new_img_like


# --------------------------------------------------------------------
# 1)  Utility: load the “label‑index → region name” text file
# --------------------------------------------------------------------
def load_atlas_label_map(label_txt_path: str) -> dict[int, str]:
    mapping = {}
    with open(label_txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            idx, name = line.strip().split(maxsplit=1)
            name = name.split("\t")[0]
            name = name.replace('"', "")
            mapping[int(idx)] = name
    return mapping


# --------------------------------------------------------------------
# 2)  Core routine: overlap of ONE tumour label with atlas
# --------------------------------------------------------------------
def localize_to_gyrus(
    tumour_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_label_map: dict[int, str],
    label_index: int = 1,
) -> dict:
    """
    Parameters
    ----------
    tumour_img : nibabel image in patient space (binary or multi‑label seg)
    atlas_img  : nibabel image (anatomical atlas, same space)
    atlas_label_map : {int: str} mapping from atlas label index → region name
    tumour_label_value : which value inside tumour_img is the lesion mask
                         (e.g. 3 = enhancing, 2 = edema …)

    Returns
    -------
    results : dict
        {
          'total_voxels': int,
          'overlap': {
              atlas_index: {'region': str,
                            'voxels': int,
                            'percent': float}
              ...
          }
        }
    """
    # --- 1. affine alignment (translation only) --------------------
    if not np.allclose(tumour_img.affine[:3, 3], atlas_img.affine[:3, 3]):
        corr_aff = tumour_img.affine.copy()
        corr_aff[:3, 3] = atlas_img.affine[:3, 3]
        tumour_img = new_img_like(tumour_img, tumour_img.get_fdata(), corr_aff)

    # --- 2. resample atlas to tumour space if needed ---------------
    if atlas_img.shape != tumour_img.shape or not np.allclose(atlas_img.affine, tumour_img.affine):
        atlas_img = resample_to_img(atlas_img, tumour_img, interpolation="nearest")

    # ---- NEW: drop trailing singleton dim if present --------------
    if atlas_img.ndim == 4 and atlas_img.shape[-1] == 1:
        atlas_img = new_img_like(atlas_img,
                                 atlas_img.get_fdata()[..., 0],  # squeeze
                                 atlas_img.affine)

    # --- 3. compute overlap ---------------------------------------
    tumour_mask = (tumour_img.get_fdata() == label_index)
    atlas_data = atlas_img.get_fdata().astype(np.int16)


    overlapped = atlas_data[tumour_mask]
    unique, counts = np.unique(overlapped[overlapped > 0], return_counts=True)
    total = int(tumour_mask.sum())

    # --- 4. pack results ------------------------------------------
    overlap_dict = {}
    region_list = []
    for idx, cnt in zip(unique, counts):
        region = atlas_label_map.get(int(idx), "unknown")
        if region != "unknown":
            overlap_dict[int(idx)] = {
                "region": region,
                "voxels": int(cnt),
                "percent": float(cnt) * 100.0 / total if total else 0.0,
            }
            region_list.append(region)

    return {"total_voxels": total, "overlap": overlap_dict, "regions": sorted(region_list)}


def get_region_str(region_list):
    """
    Helper function to convert the list of regions into a string
    """
    if len(region_list) == 0:
        return "N/A"
    elif len(region_list) == 1:
        return region_list[0]
    elif len(region_list) == 2:
        return f"{region_list[0]} and {region_list[1]}"
    else:
        # For more than two regions, join them with commas and 'and'
        return ", ".join(region_list[:-1]) + " and " + region_list[-1]


def analyze_label_localization(seg_path, atlas_path, label_txt, tumour_labels):
    """
    seg_path      : path to your multi‑label tumour segmentation (NIfTI)
    atlas_path    : path to LPBA40 (or other) atlas NIfTI
    label_txt     : path to text file mapping atlas indices → region names
    tumour_labels : dict like {'ET': 3, 'SNFH': 2, 'NETC': 1, 'RC': 4}
                    (keys = your internal label names, values = voxel values)
    Returns
    -------
    summary : dict keyed by your tumour label
              e.g. summary['ET']['overlap'][46]['region'] → 'left‑MFG'
    """
    tumour_img = nib.load(seg_path)
    atlas_img = nib.load(atlas_path)
    atlas_map = load_atlas_label_map(label_txt)

    summary = {}
    for name, val in tumour_labels.items():
        summary[name] = localize_to_gyrus(
            tumour_img == val, atlas_img, atlas_map)

    return summary


# --------------------------------------------------------------------
# 4)  Minimal CLI test (optional) -----------------------------------
if __name__ == "__main__":
    seg = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-00005-100/BraTS-GLI-00005-100-seg.nii.gz"
    atlas = "/local2/amvepa91/sri24/lpba40.nii"
    #atlas = "/local2/amvepa91/sri24/tzo116plus.nii"
    labels = "/local2/amvepa91/sri24/LPBA40-labels.txt"
    #labels = "/local2/amvepa91/sri24/SRI24-tzo116plus.txt"
    txt = f"./{os.path.basename(seg)}_report.csv"

    tumour_labels = {"ET": 3, "SNFH": 2, "NETC": 1, "RC": 4}

    summ = analyze_label_localization(seg, atlas, txt, tumour_labels)

    # Pretty‑print ET example
    et = summ["ET"]
    print("Total ET voxels:", et["total_voxels"])
    for idx, info in et["overlap"].items():
        print(f"{idx:3d} {info['region']:<30} {info['voxels']:6d} "
              f"({info['percent']:5.2f}%)")

