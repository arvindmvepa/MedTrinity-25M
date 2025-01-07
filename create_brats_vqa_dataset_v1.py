import os.path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure
from create_brats_imaging_dataset import load_color_seg_png_as_labels


def extract_label_intensity(
    image: np.ndarray,
    mask: np.ndarray,
    abs_intensity_diff_thresh: float
):
    """
    Given an MR image (e.g. T1CE) and a label mask, compute:
      1) the average intensity within the label region.
      2) whether this label's region has sufficiently high intensity
         compared to the surrounding area (based on difference_factor).

    :param image: 3D numpy array (or 2D if slice-based) of the MR modality
    :param mask:  3D (or 2D) binary segmentation mask for a particular label
    :param abs_intensity_diff_thresh: Thresh by which the label region must differ
                              from surrounding intensity
    :return: (label_is_present, avg_intensity, avg_surrounding_intensity)
             label_is_present is a boolean indicating if the label is considered present
             avg_intensity is the average intensity inside the mask
             avg_surrounding_intensity is the average intensity in the surrounding area
    """
    # 1) Extract average intensity in the mask region
    masked_intensities = image[mask > 0]
    avg_intensity = np.mean(masked_intensities)


    # 2) Compute the average intensity of the surrounding area
    #    - We perform a binary dilation on the mask to define an approximate neighborhood.
    #    - Then we take all voxels in the dilated region that are not in the original mask.
    structure = generate_binary_structure(rank=mask.ndim, connectivity=1)
    dilated_mask = binary_dilation(mask, structure=structure, iterations=2)  # e.g. 2 iterations
    surrounding_mask = np.logical_and(dilated_mask, np.logical_not(mask))

    surrounding_intensities = image[surrounding_mask > 0]
    if len(surrounding_intensities) == 0:
        return False, avg_intensity, 0.0

    avg_surrounding_intensity = np.mean(surrounding_intensities)

    # 3) Check if label region is significantly different from surroundings
    if np.abs(avg_intensity - avg_surrounding_intensity) < abs_intensity_diff_thresh:
        return False, avg_intensity, avg_surrounding_intensity

    # If all checks passed, we consider the label present
    return True, avg_intensity, avg_surrounding_intensity


def process_segmentation(
    image: np.ndarray,
    mask_non_enh: np.ndarray,
    mask_enh: np.ndarray,
    mask_flair: np.ndarray,
    mask_resection: np.ndarray,
    abs_intensity_diff_thresh=10,
):
    """
    Given a T1CE image and segmentation masks for:
      - non-enhancing tumor
      - enhancing tumor
      - FLAIR hyperintensity
      - resection cavity
    Check if each label is present by comparing mask intensity vs. surrounding.

    :param image: 3D (or 2D) NumPy array of T1-contrast-enhanced MRI
    :param mask_non_enh: Binary mask for non-enhancing tumor
    :param mask_enh:     Binary mask for enhancing tumor
    :param mask_flair:   Binary mask for T2/FLAIR hyperintensity
    :param mask_resection: Binary mask for resection cavity
    :param abs_intensity_diff_thresh: Threshold for intensity difference
    """

    results = {}

    labels = {
        'non_enhancing_tumor': mask_non_enh,
        'enhancing_tumor': mask_enh,
        'flair_region': mask_flair,
        'resection_cavity': mask_resection
    }

    for label_name, label_mask in labels.items():
        present, avg_int, avg_sur_int = extract_label_intensity(
            image=image,
            mask=label_mask,
            abs_intensity_diff_thresh=abs_intensity_diff_thresh,
        )
        results[label_name] = {
            'is_present': present,
            'avg_intensity': avg_int,
            'avg_surrounding_intensity': avg_sur_int
        }

    return results


if __name__ == "__main__":
    img_files = ["/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02358-101/BraTS-GLI-02358-101_t1c_slice_100_y.png",
                 "/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02358-101/BraTS-GLI-02358-101_t1n_slice_100_y.png",
                 "/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02358-101/BraTS-GLI-02358-101_t2w_slice_100_y.png",
                 "/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02358-101/BraTS-GLI-02358-101_t2f_slice_100_y.png"]
    for img_file in img_files:
        print(f"Processing image: {os.path.basename(img_file)}")
        seg_file = "/local2/amvepa91/MedTrinity-25M/output_pngs/BraTS-GLI-02358-101/BraTS-GLI-02358-101_seg_slice_100_y.png"

        image_t1ce = Image.open(img_file)
        seg_map_2d = load_color_seg_png_as_labels(seg_file)

        # Similarly, create dummy masks:
        mask_non_enh = seg_map_2d == 1
        mask_enh = seg_map_2d == 3
        mask_flair = seg_map_2d == 2
        mask_resection = seg_map_2d == 4

        # Process segmentation:
        results = process_segmentation(
            image=image_t1ce,
            mask_non_enh=mask_non_enh,
            mask_enh=mask_enh,
            mask_flair=mask_flair,
            mask_resection=mask_resection,
            abs_intensity_diff_thresh=10
        )

        # Print results
        for label, stats in results.items():
            print(f"Label: {label}")
            print(f"  Is Present: {stats['is_present']}")
            print(f"  Avg Intensity: {stats['avg_intensity']:.2f}")
            print(f"  Avg Surrounding Intensity: {stats['avg_surrounding_intensity']:.2f}")
            print()
