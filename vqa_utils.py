from skimage.morphology import convex_hull_image
from skimage.measure import label
import numpy as np
import re
from collections import defaultdict
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as label_, binary_dilation, generate_binary_structure


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
        return np.zeros_like(mask, dtype=bool), False, 0.0, 0.0

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

    return kept_mask, label_is_present, np.mean(avg_intensities), np.mean(avg_surroundings)



def get_seg_ids_empty_counts(all_vqa_questions):
    empty_count_map = defaultdict(int)
    for q in all_vqa_questions:
        q_type = q.get("type", "unknown")
        answer = q.get("answer", "")
        seg_id = q.get("seg_id", "unknown")
        empty_count_map[seg_id]

        # Parse adjacency area questions
        if q_type == "adj_area":
            # Typical answer format: "25.0%, which is the majority"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                adj_val = float(match.group(1))
                if adj_val == 0.0:
                    empty_count_map[seg_id] += 1

        # Parse adjacency quadrant questions
        if q_type == "adj_quadrants":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Parse area questions
        if q_type == "area":
            # Typical answer format: "25.0%, which is a small portion"
            match = re.search(r'([\d.]+)%,', answer)
            if match:
                area_val = float(match.group(1))
                if area_val == 0.0:
                    empty_count_map[seg_id] += 1

        # Check bounding box questions
        if q_type == "bbox":
            # Typical answer format: "top-left, bottom-right" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "quadrant":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "extent":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1

        # Check quadrant questions
        if q_type == "solidity":
            # Typical answer format: "top-left" or "none"
            if "none" in answer.lower():
                empty_count_map[seg_id] += 1
    return empty_count_map


def analyze_label_relationship(maskA, maskB, total_pixels, height, width):
    """
    Analyze adjacency between maskB and maskA, reporting:
      - % of maskB that is adjacent to maskA
      - bounding box quadrants for adjacent vs. non-adjacent
      - subjective interpretations
    """
    adjacent_mask_B, non_adjacent_mask_B = compute_connected_component_adjacency_masks(maskA, maskB)
    adjacent_pct = compute_adj_percentage(adjacent_mask_B, maskB)
    non_adjacent_pct = 100 - adjacent_pct

    bbox_adjacent = compute_bounding_box(adjacent_mask_B)
    bbox_non_adjacent = compute_bounding_box(non_adjacent_mask_B)

    quadrants_adjacent = get_bounding_box_quadrants(bbox_adjacent, height, width)
    quadrants_non_adjacent = get_bounding_box_quadrants(bbox_non_adjacent, height, width)

    return {
        "adjacent_percentage": adjacent_pct,
        "non_adjacent_percentage": non_adjacent_pct,
        "adjacent_quadrants": quadrants_adjacent,
        "non_adjacent_quadrants": quadrants_non_adjacent,
        "adjacent_interpretation": interpret_relationship_percentage(adjacent_pct),
        "non_adjacent_interpretation": interpret_relationship_percentage(non_adjacent_pct)
    }


def get_label_mask(seg_map_2d, label):
    if label in [1, 2, 3, 4]:
        return seg_map_2d == label
    elif label == 5:
        return (seg_map_2d == 1) | (seg_map_2d == 3)
    else:
        raise ValueError(f"Invalid label: {label}")


def analyze_label_summary(seg_map_2d, height, width, total_pixels, image=None, abs_intensity_diff_thresh=10,
                          labels_order=(1, 2, 3, 4, 5)):
    """
    For each label (1..4), compute:
      - area percentage + subjective interpretation
      - centroid quadrant
      - bounding box quadrants
      - extent-based "compactness" measure
    """
    label_summaries = []

    for lbl in labels_order:
        mask = get_label_mask(seg_map_2d, lbl)
        if image is not None:
            mask, _, _, _ = extract_label_intensity_components(image=image, mask=mask,
                                                               abs_intensity_diff_thresh=abs_intensity_diff_thresh)
        label_name = label_names.get(lbl, f"Label {lbl}")

        area_pct = compute_area_percentage(mask, total_pixels)
        area_interp = interpret_area_percentage(area_pct)

        if area_interp == "none":
            centroid = None
            quadrant = "none"
            bounding_box_quads = None
            bounding_box_str = "none"
            extent_value = 0.0
            extent_interp = "none"
            solidity_value = 0.0
            solidity_interp = "none"
        else:
            centroid = center_of_mass(mask)
            quadrant = get_quadrant(centroid, height, width)

            bbox = compute_bounding_box(mask)
            bounding_box_quads = get_bounding_box_quadrants(bbox, height, width)
            bounding_box_str = bounding_box_quads if bounding_box_quads else "none"

            # Extent-based compactness
            extent_value, extent_interp = measure_extent_compactness(mask)
            solidity_value, solidity_interp = measure_solidity(mask)

        label_summaries.append({
            "label": lbl,
            "name": label_name,
            "area_pct": area_pct,
            "area_interp": area_interp,
            "centroid_quadrant": quadrant,
            "bbox_quadrants": bounding_box_quads,
            "bbox_str": bounding_box_str,
            "extent_value": extent_value,
            "extent_interp": extent_interp,
            "solidity_value": solidity_value,
            "solidity_interp": solidity_interp
        })
    return label_summaries


def analyze_segmentation_map(seg_map_2d):
    """
    Master function to produce a textual report combining:
      - Label summaries (area %, quadrant, bounding box, extent-based compactness)
        with subjective interpretations.
      - Non-Enh vs Enh tumor adjacency info.
      - FLAIR vs Tumor Core adjacency info.
      - Resection cavity vs tumor core & FLAIR.
    """
    height, width = seg_map_2d.shape
    total_pixels = seg_map_2d.size

    # Summaries of labels
    label_summaries = analyze_label_summary(seg_map_2d, height, width, total_pixels)

    # Create masks for relationships
    mask1 = (seg_map_2d == 1)  # Non-Enh
    mask2 = (seg_map_2d == 2)  # FLAIR
    mask3 = (seg_map_2d == 3)  # Enh
    mask4 = (seg_map_2d == 4)  # Resection cavity
    tumor_core_mask = np.logical_or(mask1, mask3)  # Non-Enh + Enh

    # Relationship: Non-Enh (1) vs. Enh (3)
    rel_1_vs_3 = analyze_label_relationship(mask1, mask3, total_pixels, height, width)

    # Relationship: FLAIR (2) vs. Tumor Core (1+3)
    rel_2_vs_core = analyze_label_relationship(tumor_core_mask, mask2, total_pixels, height, width)

    # Relationship: Resection Cavity (4) vs Tumor Core
    rel_4_vs_core = analyze_label_relationship(tumor_core_mask, mask4, total_pixels, height, width)

    # Relationship: Resection Cavity (4) vs FLAIR (2)
    rel_4_vs_2 = analyze_label_relationship(mask2, mask4, total_pixels, height, width)

    #######################################################################
    # Build the text report
    #######################################################################
    lines = []

    lines.append("===== LABEL SUMMARIES =====")
    for summ in label_summaries:
        lines.append(f"{summ['name']} (Label {summ['label']}):")
        lines.append(f"  - Area covers ~{summ['area_pct']:.2f}% of total pixels, which is {summ['area_interp']}.")
        lines.append(f"  - Centered in the {summ['centroid_quadrant']} quadrant.")
        lines.append(f"  - Bounding box quadrants: {summ['bbox_str'] or 'none'}")
        lines.append(f"  - Within its bounding box, it is {summ['extent_interp']} (extent={summ['extent_value']:.2f}) and {summ['solidity_interp']} (solidity={summ['solidity_value']:.2f})")
        lines.append("")

    # 1) Non-Enh vs. Enh Tumor
    lines.append("===== NON-ENHANCING TUMOR (Label 1) vs. ENHANCING TUMOR (Label 3) =====")
    lines.append(f"Enhancing tumor adjacent to non-enhancing tumor: ~{rel_1_vs_3['adjacent_percentage']:.2f}% ({rel_1_vs_3['adjacent_interpretation']}).")
    lines.append(f"Enhancing tumor NOT adjacent: ~{rel_1_vs_3['non_adjacent_percentage']:.2f}% ({rel_1_vs_3['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_1_vs_3['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_1_vs_3['non_adjacent_quadrants']) or 'none'}\n")

    # 2) FLAIR vs. Tumor Core
    lines.append("===== FLAIR HYPERINTENSITY (Label 2) vs. TUMOR CORE (Labels 1 & 3) =====")
    lines.append(f"FLAIR area adjacent to tumor core: ~{rel_2_vs_core['adjacent_percentage']:.2f}% ({rel_2_vs_core['adjacent_interpretation']}).")
    lines.append(f"FLAIR area NOT adjacent: ~{rel_2_vs_core['non_adjacent_percentage']:.2f}% ({rel_2_vs_core['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_2_vs_core['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_2_vs_core['non_adjacent_quadrants']) or 'none'}\n")

    # 3) Resection Cavity (4) vs. Tumor Core
    lines.append("===== RESECTION CAVITY (Label 4) vs. TUMOR CORE (Labels 1 & 3) =====")
    lines.append(f"Resection cavity area adjacent to tumor core: ~{rel_4_vs_core['adjacent_percentage']:.2f}% ({rel_4_vs_core['adjacent_interpretation']}).")
    lines.append(f"Resection cavity area NOT adjacent: ~{rel_4_vs_core['non_adjacent_percentage']:.2f}% ({rel_4_vs_core['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_4_vs_core['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_4_vs_core['non_adjacent_quadrants']) or 'none'}\n")

    # 4) Resection Cavity (4) vs. FLAIR (2)
    lines.append("===== RESECTION CAVITY (Label 4) vs. FLAIR (Label 2) =====")
    lines.append(f"Resection cavity area adjacent to FLAIR: ~{rel_4_vs_2['adjacent_percentage']:.2f}% ({rel_4_vs_2['adjacent_interpretation']}).")
    lines.append(f"Resection cavity area NOT adjacent: ~{rel_4_vs_2['non_adjacent_percentage']:.2f}% ({rel_4_vs_2['non_adjacent_interpretation']}).")
    lines.append(f"Quadrants of adjacent regions: {', '.join(rel_4_vs_2['adjacent_quadrants']) or 'none'}")
    lines.append(f"Quadrants of non-adjacent regions: {', '.join(rel_4_vs_2['non_adjacent_quadrants']) or 'none'}")

    return "\n".join(lines)


def vqa_round(value):
    """
    Round a value to a specific number of decimal places.
    """
    round_val = np.round(value, 1)
    if round_val >= 0.05 and round_val < 0.1:
        return 0.1
    if round_val < .05:
        return 0.0
    else:
        return round_val


label_names = {
            1: "Non-Enhancing Tumor",
            2: "Surrounding Non-enhancing FLAIR hyperintensity",
            3: "Enhancing Tissue",
            4: "Resection Cavity",
            5: "Tumor Core"}


def compute_bounding_box(mask):
    """
    Returns (min_row, min_col, max_row, max_col) for all True pixels in `mask`.
    If `mask` is empty, returns None.
    """
    coords = np.where(mask)
    if coords[0].size == 0:
        return None
    min_r, max_r = coords[0].min(), coords[0].max() + 1
    min_c, max_c = coords[1].min(), coords[1].max() + 1
    return (min_r, min_c, max_r, max_c)

def compute_area_percentage(mask, total_pixels):
    """
    Returns the percentage of 'mask' pixels relative to the total segmentation size.
    """
    return vqa_round((mask.sum() / total_pixels) * 100)

def compute_adj_percentage(adj_mask, orig_mask):
    """
    Returns the percentage of adject 'mask' pixels relative to the overall mask.
    """
    if orig_mask.sum() == 0:
        return 0
    else:
        return vqa_round((adj_mask.sum() / orig_mask.sum()) * 100)


def interpret_area_percentage(pct):
    """
    Subjective interpretation of area percentage, tuned for smaller values.
    Example thresholds (you can tweak these to your liking):
      - 0.0%:    "none"
      - <0.1%:   "almost negligible"
      - <0.5%:   "tiny fraction"
      - <2%:     "very small fraction"
      - <5%:     "small portion"
      - <10%:    "moderate portion"
      - <20%:    "significant portion"
      - <40%:    "large portion"
      - <70%:    "major portion"
      - >=70%:   "the vast majority"
    """
    if pct == 0.0:
        return "none"
    elif pct < 0.1:
        return "almost negligible"
    elif pct < 0.5:
        return "tiny fraction"
    elif pct < 1:
        return "very small fraction"
    elif pct < 2:
        return "small portion"
    elif pct < 5:
        return "moderate portion"
    elif pct < 12:
        return "significant portion"
    elif pct < 40:
        return "large portion"
    elif pct < 70:
        return "major portion"
    else:
        return "the vast majority"


def get_quadrant(centroid, height, width):
    """
    Maps a (row, col) centroid to one of 9 quadrants (top-left to bottom-right).
    """
    if not centroid or np.isnan(centroid[0]) or np.isnan(centroid[1]):
        return "none"

    row, col = centroid
    third_height = height / 3
    third_width = width / 3

    if row < third_height:  # top row
        if col < third_width:
            return "top-left"
        elif col < 2 * third_width:
            return "top-center"
        else:
            return "top-right"
    elif row < 2 * third_height:  # middle row
        if col < third_width:
            return "center-left"
        elif col < 2 * third_width:
            return "center-center"
        else:
            return "center-right"
    else:  # bottom row
        if col < third_width:
            return "bottom-left"
        elif col < 2 * third_width:
            return "bottom-center"
        else:
            return "bottom-right"

def get_bounding_box_quadrants(bbox, height, width):
    """
    Determine which quadrants are affected by a bounding box
    by sampling corners of the bounding box.
    """
    if not bbox:
        return "none"

    min_r, min_c, max_r, max_c = bbox
    affected_quadrants = set()

    # We check the four corners
    corners = [
        (min_r, min_c),
        (min_r, max_c - 1),
        (max_r - 1, min_c),
        (max_r - 1, max_c - 1),
    ]
    for (r, c) in corners:
        quadrant = get_quadrant((r, c), height, width)
        if quadrant != "none":
            affected_quadrants.add(quadrant)
    if len(affected_quadrants) == 0:
        return "none"
    affected_quadrants_str = ", ".join(sorted(affected_quadrants))
    return affected_quadrants_str


def compute_connected_component_adjacency_masks(maskA, maskB, dilation_iterations=1):
    """
    Identify which connected components of maskB are adjacent to maskA
    (within a specified dilation). Returns two binary masks:
      1) B_adj_mask   -> Union of all B’s connected components that touch A
      2) B_nonadj_mask-> Union of all B’s connected components that do not touch A

    Parameters
    ----------
    maskA : 2D bool array
        Binary mask for label A.
    maskB : 2D bool array
        Binary mask for label B.
    dilation_iterations : int, default=1
        Number of times to dilate A to consider adjacency.
        1 => one-pixel boundary expansion.

    Returns
    -------
    B_adj_mask : 2D bool array
        All connected components in B that are adjacent to A.
    B_nonadj_mask : 2D bool array
        All connected components in B that are not adjacent to A.
    labeledB : 2D int array
        Connected component labels for B (same shape as maskB),
        where 0 => background; 1..N => different connected components.
    """

    # 1) Dilate A so boundary contact is recognized
    dilatedA = maskA.copy()
    for _ in range(dilation_iterations):
        dilatedA = binary_dilation(dilatedA)

    # 2) Label each connected component in B
    labeledB = label(maskB, connectivity=2)  # 8-connected

    # Initialize empty masks for adjacency
    B_adj_mask = np.zeros_like(maskB, dtype=bool)
    B_nonadj_mask = np.zeros_like(maskB, dtype=bool)

    max_label = labeledB.max()
    for cc_id in range(1, max_label + 1):
        component_mask = (labeledB == cc_id)

        # Check overlap with dilated A
        overlap = np.logical_and(dilatedA, component_mask).any()

        if overlap:
            # Entire connected component belongs to the "adjacent" mask
            B_adj_mask |= component_mask
        else:
            # Entire connected component belongs to the "non-adjacent" mask
            B_nonadj_mask |= component_mask

    return B_adj_mask, B_nonadj_mask



###############################################################################
# Measuring Compactness via "Extent" (Region's fill within its bounding box)
###############################################################################

def measure_extent_compactness(mask):
    """
    Extent = region_area / bounding_box_area.
    Returns a float in [0..1], plus a subjective interpretation:
      - <0.2 => "very sparse"
      - 0.2..0.5 => "somewhat scattered"
      - 0.5..0.8 => "partially filling"
      - 0.8..0.95 => "nearly filling"
      - >0.95 => "almost fully filling"
    """
    bbox = compute_bounding_box(mask)
    area = mask.sum()
    if not bbox:
        return 0.0, "none"

    min_r, min_c, max_r, max_c = bbox
    bbox_h = max_r - min_r
    bbox_w = max_c - min_c

    bbox_area = bbox_h * bbox_w
    if bbox_area == 0:
        return 0.0, "none"

    extent = (area / bbox_area) * 100
    interpretation = interpret_extent(extent)
    return extent, interpretation

def interpret_extent(value):
    """
    Subjective interpretation of how well the region fills its bounding box.
    """
    if value == 0.0:
        return "none"
    elif value < 20.0:
        return "very sparse"
    elif value < 50.0:
        return "somewhat scattered"
    elif value < 80.0:
        return "partially filled"
    elif value < 95.0:
        return "nearly filled"
    else:
        return "almost fully filled"


def measure_solidity(mask):
    """
    Computes Solidity = area / convex_hull_area.
    """
    convex_hull = convex_hull_image(mask)
    region_area = mask.sum()
    convex_area = convex_hull.sum()
    solidity = vqa_round(region_area / convex_area) * 100 if convex_area > 0 else 0.0
    return solidity, interpret_solidity(solidity)


def interpret_solidity(value):
    """
    Subjective interpretation of solidity.
    """
    if value == 0.0:
        return "none"
    elif value < 50.0:
        return "highly irregular and scattered"
    elif value < 80.0:
        return "somewhat compact but irregular"
    else:
        return "mostly compact"


def interpret_relationship_percentage(pct):
    """
    Subjective interpretation of adjacency / overlap percentages:
     - <5% => "minimal"
     - 5-20% => "some"
     - 20-50% => "a moderate amount"
     - 50-80% => "the majority"
     - >80% => "the vast majority"
    """
    if pct == 0:
        return "none"
    if pct < 5:
        return "minimal"
    elif pct < 20:
        return "some"
    elif pct < 50:
        return "a moderate amount"
    elif pct < 80:
        return "the majority"
    else:
        return "the vast majority"

