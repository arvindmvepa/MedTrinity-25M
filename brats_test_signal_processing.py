import os.path
import numpy as np
from PIL import Image
from vqa_utils import extract_label_intensity_components
from create_brats_imaging_dataset import load_color_seg_png_as_labels


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
        kept_mask, present, avg_int, avg_sur_int = extract_label_intensity_components(
            image=image,
            mask=label_mask,
            abs_intensity_diff_thresh=abs_intensity_diff_thresh,
        )
        results[label_name] = {
            'old_mask_count': np.sum(label_mask),
            'new_mask_count': np.sum(kept_mask),
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

        image = np.array(Image.open(img_file))
        seg_map_2d = load_color_seg_png_as_labels(seg_file)

        # Similarly, create dummy masks:
        mask_non_enh = seg_map_2d == 1
        mask_enh = seg_map_2d == 3
        mask_flair = seg_map_2d == 2
        mask_resection = seg_map_2d == 4

        # Process segmentation:
        results = process_segmentation(
            image=image,
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
            print(f"  Old Mask Count: {stats['old_mask_count']}")
            print(f"  New Mask Count: {stats['new_mask_count']}")
            print(f"  Avg Intensity: {stats['avg_intensity']:.2f}")
            print(f"  Avg Surrounding Intensity: {stats['avg_surrounding_intensity']:.2f}")
            print()
