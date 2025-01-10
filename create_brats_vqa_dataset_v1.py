import os.path
import numpy as np
from PIL import Image
from scipy.ndimage import label as label_, binary_dilation, generate_binary_structure
from create_brats_imaging_dataset import load_color_seg_png_as_labels


def extract_label_intensity_components(
    image: np.ndarray,
    mask: np.ndarray,
    abs_intensity_diff_thresh: float,
    dilation_iterations: int = 2
):
    """
    Given an MR image (e.g. T1CE) and a segmentation mask (possibly with multiple
    connected components), this function:
      1) Identifies connected components in the mask.
      2) For each connected component, computes:
         - the average intensity within that component.
         - the average intensity in the local 'surrounding' region (via dilation).
         - checks if the absolute difference is above abs_intensity_diff_thresh.
      3) Returns a final mask that is the UNION of all connected components that
         pass the threshold criterion, as well as a boolean flag indicating if
         there is at least one component meeting the criterion.

    :param image: 3D numpy array (or 2D) of the MR modality
    :param mask:  3D (or 2D) binary segmentation mask
    :param abs_intensity_diff_thresh: Threshold by which each connected component
                                      must differ from its surroundings
    :param dilation_iterations: Number of iterations for the surrounding dilation
    :return: kept_mask, label_is_present, avg_intensities, avg_surroundings
             where:
                 - kept_mask is a binary mask containing only those components
                   whose difference from surrounding is >= threshold.
                 - label_is_present is True if any component meets the criterion.
                 - avg_intensities is a list of average intensities for each
                   connected component (in the order of component labels).
                 - avg_surroundings is a list of average surrounding intensities
                   for each connected component (same order).
    """

    # 1) Label the connected components in the mask
    labeled_mask, num_components = label_(mask)
    if num_components == 0:
        # No components at all
        return np.zeros_like(mask, dtype=bool), False, [], []

    # Prepare outputs
    kept_components = []
    avg_intensities = []
    avg_surroundings = []

    # Define the structure for dilation
    structure = generate_binary_structure(rank=mask.ndim, connectivity=1)

    # 2) Iterate over each connected component
    for comp_idx in range(1, num_components + 1):
        component_mask = (labeled_mask == comp_idx)

        # (a) Average intensity in this component
        component_intensities = image[component_mask]
        if component_intensities.size == 0:
            # Should not happen if label() found a component, but just in case
            avg_intensity = 0.0
        else:
            avg_intensity = component_intensities.mean()

        # (b) Average intensity of the surrounding region
        dilated_mask = binary_dilation(
            component_mask,
            structure=structure,
            iterations=dilation_iterations
        )
        surrounding_mask = np.logical_and(dilated_mask, np.logical_not(component_mask))

        surrounding_intensities = image[surrounding_mask]
        if surrounding_intensities.size == 0:
            # If there's no valid surrounding, skip or treat as failing threshold
            avg_surrounding_intensity = 0.0
        else:
            avg_surrounding_intensity = surrounding_intensities.mean()

        # Store these for reference
        avg_intensities.append(avg_intensity)
        avg_surroundings.append(avg_surrounding_intensity)

        # (c) Check if this component's difference meets threshold
        if abs(avg_intensity - avg_surrounding_intensity) >= abs_intensity_diff_thresh:
            kept_components.append(comp_idx)

    # 3) Construct the union mask of all kept connected components
    if len(kept_components) == 0:
        kept_mask = np.zeros_like(mask, dtype=bool)
        label_is_present = False
    else:
        # Create a binary mask that is True where the component label is in kept_components
        kept_mask = np.isin(labeled_mask, kept_components)
        label_is_present = True

    return kept_mask, label_is_present, avg_intensities, avg_surroundings


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
            print(f"  Avg Intensity: {stats['avg_intensity']:.2f}")
            print(f"  Avg Surrounding Intensity: {stats['avg_surrounding_intensity']:.2f}")
            print()
