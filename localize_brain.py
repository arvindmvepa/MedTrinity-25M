from nilearn import plotting
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img, new_img_like


LOBE_MAP: dict[str, set[int]] = {
    "frontal": {
        21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34,
    },
    "parietal": {
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    },
    "occipital": {
        61, 62, 63, 64, 65, 66, 67, 68, 89, 90,
    },
    "temporal": {
        81, 82, 83, 84, 85, 86, 87, 88, 91, 92,
    },
    "limbic": {121, 122, 165, 166},
    "insula": {101, 102},
    "subcortical": {161, 162, 163, 164},
    "cerebellum": {181},
    "brainstem": {182},
    "background": {0},        # keep 0 → background
}


# Build a quick reverse look‑up once so the function stays O(1)
_ID_TO_LOBE: dict[int, str] = {
    idx: lobe for lobe, indices in LOBE_MAP.items() for idx in indices
}


def load_atlas_label_map(label_txt_path, use_lobes=True):
    mapping = {}
    with open(label_txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            idx, name = line.strip().split(maxsplit=1)
            if use_lobes:
                idx = int(idx)
                mapping[idx] = _ID_TO_LOBE[idx]
            else:
                name = name.split("\t")[0]
                name = name.replace('"', "")
                mapping[int(idx)] = name
    return mapping


def localize_to_brain_regions(
    tumour_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_label_map: dict[int, str],
    label_index: int = 1,
    debug=False
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

    if debug:
        display = plotting.plot_roi(tumour_img,
                                    bg_img=atlas_img,
                                    title=f"Tumour-Affine Alignment Check Label", alpha=0.5)
        display.savefig(f"tumour_affine_alignment_check.png")
        display.close()



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


def analyze_label_localization(seg_path="/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-00005-100/BraTS-GLI-00005-100-seg.nii.gz",
                               atlas_path="/local2/amvepa91/sri24/lpba40.nii",
                               label_txt="/local2/amvepa91/sri24/LPBA40-labels.txt",
                               tumour_labels=None, debug=True):
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
    atlas_label_map = load_atlas_label_map(label_txt)

    summary = {}
    for name, label_index in tumour_labels.items():
        summary[name] = localize_to_brain_regions(tumour_img=tumour_img, atlas_img=atlas_img,
                                                  atlas_label_map=atlas_label_map,
                                                  label_index=label_index, debug=debug)

    return summary


# --------------------------------------------------------------------
# 4)  Minimal CLI test (optional) -----------------------------------
if __name__ == "__main__":
    seg_path = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-03027-101/BraTS-GLI-03027-101-seg.nii.gz"
    atlas_path = "/local2/amvepa91/sri24/lpba40.nii"
    #atlas_path = "/local2/amvepa91/sri24/tzo116plus.nii"
    label_txt = "/local2/amvepa91/sri24/LPBA40-labels.txt"
    #label_txt = "/local2/amvepa91/sri24/SRI24-tzo116plus.txt"

    tumour_labels = {"ET": 3, "SNFH": 2, "NETC": 1, "RC": 4}

    summ = analyze_label_localization(seg_path=seg_path, atlas_path=atlas_path, label_txt=label_txt,
                                      tumour_labels=tumour_labels)

    for tumor_label, info in summ.items():
        print(f"\nTumor label: {tumor_label}")
        print("Total voxels:", info["total_voxels"])
        for idx_, info_ in info["overlap"].items():
            print(f"{idx_:3d} {info_['region']:<30} {info_['voxels']:6d} "
                  f"({info_['percent']:5.2f}%)")
        print("Regions:", get_region_str(info["regions"]))

