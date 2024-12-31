from skimage.morphology import convex_hull_image
from skimage.measure import label
import numpy as np
import re
from collections import defaultdict
from scipy.ndimage import binary_dilation
from scipy.ndimage import center_of_mass


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


def analyze_label_summary(seg_map_2d, height, width, total_pixels):
    """
    For each label (1..4), compute:
      - area percentage + subjective interpretation
      - centroid quadrant
      - bounding box quadrants
      - extent-based "compactness" measure
    """
    labels_order = [1, 2, 3, 4, 5]
    label_summaries = []

    for lbl in labels_order:
        if lbl == 5:
            mask = (seg_map_2d == 1) | (seg_map_2d == 3)
        else:
            mask = (seg_map_2d == lbl)

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
            2: "FLAIR Hyperintensity",
            3: "Enhancing Tumor",
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
    elif pct < 2:
        return "very small fraction"
    elif pct < 5:
        return "small portion"
    elif pct < 10:
        return "moderate portion"
    elif pct < 20:
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

    extent = area / bbox_area
    interpretation = interpret_extent(extent)
    return extent, interpretation

def interpret_extent(value):
    """
    Subjective interpretation of how well the region fills its bounding box.
    """
    if value == 0.0:
        return "none"
    elif value < 0.2:
        return "very sparse"
    elif value < 0.5:
        return "somewhat scattered"
    elif value < 0.8:
        return "partially filling"
    elif value < 0.95:
        return "nearly filling"
    else:
        return "almost fully filling"


def measure_solidity(mask):
    """
    Computes Solidity = area / convex_hull_area.
    """
    convex_hull = convex_hull_image(mask)
    region_area = mask.sum()
    convex_area = convex_hull.sum()
    solidity = vqa_round(region_area / convex_area) if convex_area > 0 else 0.0
    return solidity, interpret_solidity(solidity)


def interpret_solidity(value):
    """
    Subjective interpretation of solidity.
    """
    if value == 0.0:
        return "none"
    elif value < 0.5:
        return "highly irregular and scattered"
    elif value < 0.8:
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

