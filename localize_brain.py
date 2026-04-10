from nilearn import plotting
import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np
from nilearn.image import resample_to_img, new_img_like
from tqdm import tqdm
import glob
import nilearn


LOBE_MAP = {
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
_ID_TO_LOBE = {
    idx: lobe for lobe, indices in LOBE_MAP.items() for idx in indices
}
_LOBE_TO_ID = {
    lobe: idx for idx, lobe in _ID_TO_LOBE.items()
}

# Dense-region reporting thresholds. These do not affect the original sparse
# overlap statistic or the original `regions` field.
DENSE_REGION_MIN_VOXELS = 50
DENSE_REGION_MIN_PERCENT = 1.0


def _squeeze_singleton_4d(img: nib.Nifti1Image) -> nib.Nifti1Image:
    if img.ndim == 4 and img.shape[-1] == 1:
        return new_img_like(img, img.get_fdata()[..., 0], img.affine)
    return img


def _as_closest_canonical(path_or_img):
    img = nib.load(path_or_img) if isinstance(path_or_img, str) else path_or_img
    img = nib.as_closest_canonical(img)
    return _squeeze_singleton_4d(img)


def _ensure_same_grid(moving_img: nib.Nifti1Image, ref_img: nib.Nifti1Image, interpolation="nearest"):
    moving_img = _squeeze_singleton_4d(moving_img)
    ref_img = _squeeze_singleton_4d(ref_img)
    if moving_img.shape != ref_img.shape or not np.allclose(moving_img.affine, ref_img.affine):
        moving_img = resample_to_img(moving_img, ref_img, interpolation=interpolation)
        moving_img = _squeeze_singleton_4d(moving_img)
    return moving_img


def _derive_reference_path(seg_path: str):
    if seg_path is None:
        return None
    suffixes = [
        "-t1c.nii.gz", "-t1ce.nii.gz", "-t1n.nii.gz", "-t1.nii.gz",
        "_t1c.nii.gz", "_t1ce.nii.gz", "_t1n.nii.gz", "_t1.nii.gz",
    ]
    base = seg_path
    if base.endswith("-seg.nii.gz"):
        base = base[:-len("-seg.nii.gz")]
    elif base.endswith("_seg.nii.gz"):
        base = base[:-len("_seg.nii.gz")]
    elif base.endswith(".nii.gz"):
        base = base[:-len(".nii.gz")]

    for suffix in suffixes:
        candidate = base + suffix
        if os.path.exists(candidate):
            return candidate
    case_dir = os.path.dirname(seg_path)
    stem = os.path.basename(base)
    for suffix in suffixes:
        candidate = os.path.join(case_dir, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return None


def _make_brain_mask(reference_img: nib.Nifti1Image):
    data = reference_img.get_fdata()
    mask = np.abs(data) > 0
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, iterations=1)
    return mask.astype(bool)


def _build_dense_lobe_data(atlas_data: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    sparse_lobe_data = np.zeros_like(atlas_data, dtype=np.int16)
    unique_indices = np.unique(atlas_data.astype(np.int32))
    for atlas_idx in unique_indices:
        atlas_idx = int(atlas_idx)
        lobe_name = AAL_INDEX_TO_LOBE.get(atlas_idx, "unknown")
        lobe_id = _LOBE_TO_ID.get(lobe_name, _LOBE_TO_ID["unknown"])
        if atlas_idx == 0 or lobe_id in (_LOBE_TO_ID["background"], _LOBE_TO_ID["unknown"]):
            continue
        sparse_lobe_data[atlas_data == atlas_idx] = lobe_id

    labeled_mask = sparse_lobe_data > 0
    if not np.any(labeled_mask):
        return sparse_lobe_data

    fill_mask = brain_mask & (~labeled_mask)
    if np.any(fill_mask):
        _, nearest_indices = distance_transform_edt(~labeled_mask, return_indices=True)
        dense_lobe_data = sparse_lobe_data.copy()
        dense_lobe_data[fill_mask] = sparse_lobe_data[tuple(nearest_indices[:, fill_mask])]
    else:
        dense_lobe_data = sparse_lobe_data
    return dense_lobe_data.astype(np.int16)


def _dense_overlap_from_mask(dense_lobe_data: np.ndarray, tumour_mask: np.ndarray):
    unique, counts = np.unique(dense_lobe_data[tumour_mask], return_counts=True)
    total = int(tumour_mask.sum())
    overlap_dict = {}
    region_list = []
    overlap_voxels = 0
    for idx, cnt in zip(unique, counts):
        idx = int(idx)
        if idx <= 0:
            continue
        region = _ID_TO_LOBE.get(idx, "unknown")
        if region == "unknown":
            continue
        overlap_dict[idx] = {
            "region": region,
            "voxels": int(cnt),
            "percent": float(cnt) * 100.0 / total if total else 0.0,
        }
        region_list.append(region)
        overlap_voxels += int(cnt)
    return overlap_voxels, overlap_dict, sorted(set(region_list))


def _threshold_region_list(overlap_dict: dict, min_voxels: int = DENSE_REGION_MIN_VOXELS,
                           min_percent: float = DENSE_REGION_MIN_PERCENT):
    """
    Filter region labels using a minimum voxel threshold and minimum percent
    threshold. This is only used for the new dense-region reporting path.
    """
    kept = []
    for info in overlap_dict.values():
        if info["voxels"] >= min_voxels and info["percent"] >= min_percent:
            kept.append(info["region"])
    return sorted(set(kept))


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


def _squeeze_to_3d(img):
    """Return a 3‑D version of `img`.
       If the 4th dim has length 1, squeeze it;
       otherwise raise, because we don’t know which volume to keep."""
    if img.ndim == 3:
        return img
    if img.ndim == 4 and img.shape[-1] == 1:
        data3d = img.get_fdata()[..., 0]          # drop t‑dim
        return new_img_like(img, data3d, img.affine, copy_header=True)
    raise ValueError(
        f'Expected 3‑D or 4‑D with singleton 4th dim; got shape={img.shape}'
    )


def localize_to_brain_regions(
    tumour_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    atlas_label_map,
    label_index = 1,
    debug=False
):
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

    # --- 0. make both images canonical RAS+, 1 mm³ --------------------------
    tumor_img = _squeeze_to_3d(tumour_img)
    atlas_img = _squeeze_to_3d(atlas_img)
    tumour_img = nib_processing.conform(tumour_img)  # isotropic, RAS
    atlas_img = nib_processing.conform(atlas_img)

    # --- 1. bring atlas FOV to tumour FOV (deal with cropping) -------------
    if not all(np.less_equal(tumour_img.shape, atlas_img.shape)):
        atlas_img = nilearn.image.crop_img(atlas_img, tumour_img.affine,
                                           tumour_img.shape)

    # --- 2. affine alignment (translation only) --------------------
    if not np.allclose(tumour_img.affine[:3, 3], atlas_img.affine[:3, 3]):
        corr_aff = tumour_img.affine.copy()
        corr_aff[:3, 3] = atlas_img.affine[:3, 3]
        tumour_img = new_img_like(tumour_img, tumour_img.get_fdata(), corr_aff)

    # --- 3. resample atlas to tumour space if needed ---------------
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
    nonzero = overlapped[overlapped > 0]
    unique, counts = np.unique(nonzero, return_counts=True)
    total = int(tumour_mask.sum())
    overlap_voxels = int(nonzero.size)
    overlap_fraction = float(overlap_voxels) / float(total) if total else 0.0

    overlap_dict = {}
    sparse_region_list = []
    for idx, cnt in zip(unique, counts):
        region = atlas_label_map.get(int(idx), "unknown")
        if region != "unknown":
            overlap_dict[int(idx)] = {
                "region": region,
                "voxels": int(cnt),
                "percent": float(cnt) * 100.0 / total if total else 0.0,
            }
            sparse_region_list.append(region)

    # --- 4. improved dense lobe assignment for final region labels ---
    reference_path = _derive_reference_path(seg_path)
    if reference_path is not None:
        reference_img = _as_closest_canonical(reference_path)
        reference_img = _ensure_same_grid(reference_img, tumour_img, interpolation="continuous")
        brain_mask = _make_brain_mask(reference_img)
    else:
        # Fallback: allow some spill beyond sparse AAL support without requiring
        # a reference anatomy. This keeps the function signature unchanged.
        brain_mask = atlas_data > 0
        brain_mask = binary_fill_holes(brain_mask)
        brain_mask = binary_closing(brain_mask, iterations=2)

    dense_lobe_data = _build_dense_lobe_data(atlas_data, brain_mask)
    dense_overlap_voxels, dense_overlap_dict, dense_region_list = _dense_overlap_from_mask(
        dense_lobe_data=dense_lobe_data,
        tumour_mask=tumour_mask,
    )
    dense_overlap_fraction = float(dense_overlap_voxels) / float(total) if total else 0.0
    sparse_region_list = sorted(set(sparse_region_list))
    dense_region_list = sorted(set(dense_region_list))
    dense_regions_thresholded = _threshold_region_list(
        dense_overlap_dict,
        min_voxels=DENSE_REGION_MIN_VOXELS,
        min_percent=DENSE_REGION_MIN_PERCENT,
    )

    return {
        # Original fields preserved exactly
        "total_voxels": total,
        "overlap_voxels": overlap_voxels,
        "overlap_fraction": overlap_fraction,
        "overlap": overlap_dict,
        "regions": dense_regions_thresholded,

        # Explicit sparse alias for readability
        "sparse_regions": sparse_region_list,

        # Extra fields for dense-lobe reporting
        "dense_overlap_voxels": dense_overlap_voxels,
        "dense_overlap_fraction": dense_overlap_fraction,
        "dense_overlap": dense_overlap_dict,
        "dense_regions_all": dense_region_list,
        "dense_regions": dense_regions_thresholded,
        "dense_region_thresholds": {
            "min_voxels": DENSE_REGION_MIN_VOXELS,
            "min_percent": DENSE_REGION_MIN_PERCENT,
        },
        "reference_path": reference_path,
    }


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
    #seg_path = "/local2/shared_data/BraTS2024-BraTS-GLI/training_data1_v2/BraTS-GLI-03027-101/BraTS-GLI-03027-101-seg.nii.gz"
    seg_path ="/local2/shared_data/BraTS2024-BraTS-MET/MICCAI-BraTS2024-MET-Challenge-Training_overall/BraTS-MET-00759-000/BraTS-MET-00759-000-seg.nii.gz"
    #seg_path = "/local2/shared_data/BraTS2024-BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/BraTS-GoAT-02235/BraTS-GoAT-02235-seg.nii.gz"
    atlas_path = "/local2/amvepa91/sri24/lpba40.nii"
    #atlas_path = "/local2/amvepa91/sri24/tzo116plus.nii"
    label_txt = "/local2/amvepa91/sri24/LPBA40-labels.txt"
    #label_txt = "/local2/amvepa91/sri24/SRI24-tzo116plus.txt"
    
    """
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
    """

    seg_paths = sorted(glob.glob("/local2/shared_data/BraTS2024-BraTS-MET/MICCAI-BraTS2024-MET-Challenge-Training_overall/BraTS-MET*/BraTS-MET*seg.nii.gz"))
    tumour_labels = {"ET": 3, "SNFH": 2, "NETC": 1}
    atlas_overlap = {"ET": [], "SNFH": [], "NETC": []}
    for seg_path in tqdm(seg_paths[:3]):
        try:
            summ = analyze_label_localization(seg_path=seg_path, tumour_labels=tumour_labels, debug=False)
            for tumor_label, info in summ.items():
                if info['total_voxels'] > 0:
                    atlas_overlap_sparse[tumor_label].append(info['overlap_fraction']*100)
                    atlas_overlap_dense[tumor_label].append(info['dense_overlap_fraction']*100)
        except Exception as e:
            print(f"Error processing {seg_path}: {e}")
    print("\n\nSummary of overlap percentages (%):")
    for tumor_label in tumour_labels.keys():
        sparse_vals = atlas_overlap_sparse[tumor_label]
        dense_vals = atlas_overlap_dense[tumor_label]
        if len(sparse_vals) == 0:
            print(f"{tumor_label}: no samples")
            continue
        print(
            f"{tumor_label}: "
            f"sparse={np.mean(sparse_vals):.2f} ± {np.std(sparse_vals):.2f}, "
            f"dense={np.mean(dense_vals):.2f} ± {np.std(dense_vals):.2f}, "
            f"#samples: {len(sparse_vals)}"
        )

