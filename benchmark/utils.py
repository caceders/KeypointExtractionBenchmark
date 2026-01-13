from typing import Tuple
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random


import os
import cv2
import numpy as np
from typing import Tuple

def _make_circle_img(size: int = 400, radius: int = 20) -> np.ndarray:
    """
    White background with a filled black circle in the center.
    BGR uint8, shape (size, size, 3).
    """
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(img, center, radius, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    return img

def _make_single_corner_img(size: int = 400, square_size: int = 200, top_left: tuple[int,int] = (100, 100)) -> np.ndarray:
    """
    White background with a 200x200 square whose intensity ramps from dark at the
    top-left corner to bright at the bottom-right corner. This produces a single
    dominant high-contrast corner at the square's top-left vertex.
    """
    img = np.full((size, size, 3), 255, dtype=np.uint8)

    y0, x0 = top_left
    y1, x1 = y0 + square_size, x0 + square_size

    # Create a diagonal gradient in [0, 255] inside the square:
    # 0 at (y0, x0) -> 255 at (y1-1, x1-1)
    gx = np.linspace(1.0, 1.0, square_size, dtype=np.float32)
    gy = gx
    XX, YY = np.meshgrid(gx, gy)
    grad = ((XX + YY) / 2.0) * 255.0  # 0 at top-left, 255 at bottom-right
    square_gray = grad.astype(np.uint8)

    # Place gradient square into the white background (3 channels)
    img[y0:y1, x0:x1] = cv2.merge([square_gray, square_gray, square_gray])

    return img

def load_HPSequences(path_to_HPSequences: str, prepend_synthetic: bool = True, shape: str  = "circle" 
                    ) -> Tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """
    Load the HPSequence dataset (PPM images and homographies),
    optionally prepending a synthetic sequence for detector testing.

    Returns:
        image_sequences: 2D list of sequences -> images
        homography_sequences: 2D list of sequences -> homography matrices
                              (reference image to each subsequent image).
    """
    image_sequences: list[list[np.ndarray]] = []
    homography_sequences: list[list[np.ndarray]] = []

    # Iterate over all subfolders.
    for name in os.listdir(path_to_HPSequences):
        subfolder = os.path.join(path_to_HPSequences, name)
        if not os.path.isdir(subfolder):
            continue

        images = []
        homographies = []

        # Iterate over the sorted file list so that 1 comes before 2 and so on.
        for filename in sorted(os.listdir(subfolder)):
            filepath = os.path.join(subfolder, filename)

            # Store .ppm files,
            if filename.lower().endswith(".ppm"):
                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to load image: {filepath}")
                images.append(image)

            # and homographies.
            elif filename.startswith("H_"):
                homography = np.loadtxt(filepath)
                homography = np.linalg.inv(homography)
                homographies.append(homography)

        image_sequences.append(images)
        homography_sequences.append(homographies)

    if prepend_synthetic:
        sizes = [5,10,20,40,60,100]
        synthetic_images = []
        if shape == "circle":
            for i in range(6):
                synthetic_images.append(_make_circle_img(size=400, radius=sizes[i]))
        elif shape == "square":
            for i in range(6):
                synthetic_images.append(_make_single_corner_img(size=400, square_size=sizes[i], top_left=(100, 100)))

        # Reference is the first image; homography maps reference -> second image.
        H_identity = np.eye(3, dtype=float)

        image_sequences.insert(0, synthetic_images)
        homography_sequences.insert(0, [H_identity,H_identity,H_identity,H_identity,H_identity])

    return image_sequences, homography_sequences



def calculate_overlap_one_circle_to_many(circle_diameter: float, other_circles_diameters : np.ndarray, distances):
    # circle_radius   = float(circleerence_feature.keypoint.size) / 2.0
    # other_circles_radii    = np.asarray(other_circlesated_features_size_transformed, dtype=float) / 2.0
    circle_radius = circle_diameter / 2.0
    other_circles_radii = other_circles_diameters / 2.0
    EPS = 1e-12  # small epsilon for numerical stability

    circle_area  = np.pi * (circle_radius ** 2)      # scalar
    other_circles_areas = np.pi * (other_circles_radii  ** 2)      # vector

    # Intersection area (vectorized)
    intersectional_area = np.zeros_like(distances, dtype=float)

    # Case 1: disjoint (no overlap)
    disjoint_mask  = distances >= circle_radius + other_circles_radii

    # Case 2: one circle fully contained in the other
    contained_mask = distances <= np.abs(circle_radius - other_circles_radii)
    if np.any(contained_mask):
        intersectional_area[contained_mask] = np.pi * (np.minimum(circle_radius, other_circles_radii[contained_mask]) ** 2)

    # Case 3: partial overlap (lens)
    partial_mask = (~disjoint_mask) & (~contained_mask)
    if np.any(partial_mask):
        distances_partial  = distances[partial_mask]
        other_circles_radii_partial = other_circles_radii[partial_mask]

        # Stable arccos arguments
        cos1 = (distances_partial**2 + circle_radius**2 - other_circles_radii_partial**2) / (2.0 * distances_partial * circle_radius + EPS)
        cos2 = (distances_partial**2 + other_circles_radii_partial**2 - circle_radius**2) / (2.0 * distances_partial * other_circles_radii_partial + EPS)
        cos1 = np.clip(cos1, -1.0, 1.0)
        cos2 = np.clip(cos2, -1.0, 1.0)

        #MATH for overlap of circles
        term1 = circle_radius**2 * np.arccos(cos1)
        term2 = other_circles_radii_partial**2      * np.arccos(cos2)
        sq = (-distances_partial + circle_radius + other_circles_radii_partial) * (distances_partial + circle_radius - other_circles_radii_partial) * (distances_partial - circle_radius + other_circles_radii_partial) * (distances_partial + circle_radius + other_circles_radii_partial)
        term3 = 0.5 * np.sqrt(np.clip(sq, 0.0, None))

        intersectional_area[partial_mask] = term1 + term2 - term3

    # Overlap fractions — require BOTH circles to meet the threshold
    overlap_circle_frac = intersectional_area / (circle_area  + EPS)   # coverage of the circleerence circle
    overlap_other_circles_frac = intersectional_area / (other_circles_areas + EPS)   # coverage of each other_circlesated circle
    overlap_min = np.minimum(overlap_circle_frac,overlap_other_circles_frac)

    return overlap_min

def compare_rankings_and_visualize_across_sets(
    match_sets,                   # List[MatchSet]
    properties,                   # List[MatchRankingProperty]
    visualize = False,
    ignore_negatives_in_same_sequence=True,
    sample_scatter=4000,          # subsample points for scatter (None = all)
    figure_dpi=130,
    scatter_set_index=0           # which set to use for the scatter matrix
):
    # --- Basic checks ---
    if not isinstance(match_sets, (list, tuple)) or len(match_sets) == 0:
        raise ValueError("Provide a non-empty list/array of MatchSet instances.")
    if not isinstance(properties, (list, tuple)) or len(properties) < 2:
        raise ValueError("Provide at least two MatchRankingProperty items.")

    prop_names = [p.name for p in properties]
    P = len(prop_names)

    # --- Compute a Spearman matrix for each set ---
    spearman_matrices = []
    valid_set_indices = []

    for set_idx, match_set in enumerate(match_sets):
        # Collect matches from this set
        matches = list(iter(match_set))
        if len(matches) == 0:
            continue

        # Build orders per property (respecting ignore_negatives_in_same_sequence and higher_is_better)
        orders = {}
        universes = {}
        for p in properties:
            # Filter indices according to ignore_negatives_in_same_sequence
            idxs = []
            for i, m in enumerate(matches):
                if ignore_negatives_in_same_sequence and (not m.is_correct) and m.is_in_same_sequece:
                    continue
                idxs.append(i)

            if len(idxs) == 0:
                idxs = []  # no items; handled later

            # Generate comparable scores for ranking
            scores = []
            for i in idxs:
                if p.name not in matches[i].match_properties:
                    raise KeyError(f"Match at index {i} in set {set_idx} is missing '{p.name}' in match_properties.")
                v = float(matches[i].match_properties[p.name])
                scores.append(v if p.higher_is_better else -v)

            # Sort by score descending; tie-break by original index for stability
            pairs = list(zip(idxs, scores))
            pairs.sort(key=lambda x: (x[1], -x[0]), reverse=True)
            orders[p.name] = [i for i, _ in pairs]
            universes[p.name] = idxs

        # Compute common universe across properties for this set
        common_universe = set(universes[prop_names[0]])
        for name in prop_names[1:]:
            common_universe &= set(universes[name])
        common_universe = sorted(list(common_universe))
        if len(common_universe) < 2:
            # Skip this set if not enough items to form correlations
            continue

        # Restrict each order to the common universe while preserving relative order
        restricted_orders = {}
        for name in prop_names:
            restrict = []
            present = set(common_universe)
            for idx in orders[name]:
                if idx in present:
                    restrict.append(idx)
            restricted_orders[name] = restrict

        # Build rank position maps
        rankpos = {}
        for name in prop_names:
            rp = {}
            for r, idx in enumerate(restricted_orders[name]):
                rp[idx] = r
            rankpos[name] = rp

        # Spearman matrix for this set
        S = np.zeros((P, P), dtype=float)
        for i in range(P):
            for j in range(P):
                if i == j:
                    S[i, j] = 1.0
                else:
                    order_i = restricted_orders[prop_names[i]]
                    order_j = restricted_orders[prop_names[j]]
                    # Both orders are permutations of the same common universe
                    if len(order_i) != len(order_j):
                        # Shouldn't happen given restriction, but guard anyway
                        S[i, j] = 0.0
                        continue
                    if len(order_i) < 2:
                        S[i, j] = 0.0
                        continue
                    # Spearman: Pearson of rank positions
                    ar = np.array([rankpos[prop_names[i]][idx] for idx in order_i], dtype=float)
                    br = np.array([rankpos[prop_names[j]][idx] for idx in order_i], dtype=float)  # align on order_i indices
                    a_mean = float(np.mean(ar))
                    b_mean = float(np.mean(br))
                    num = float(np.sum((ar - a_mean) * (br - b_mean)))
                    den = math.sqrt(float(np.sum((ar - a_mean)**2)) * float(np.sum((br - b_mean)**2)))
                    S[i, j] = (num / den) if den > 0 else 0.0
        spearman_matrices.append(S)
        valid_set_indices.append(set_idx)

    if len(spearman_matrices) == 0:
        raise ValueError("No valid sets with sufficient common universe across properties to compute correlations.")

    
    # --- Average Spearman across sets (and compute std for annotation) ---
    stack = np.stack(spearman_matrices, axis=0)  # [num_sets, P, P]
    spearman_mean = np.mean(stack, axis=0)
    spearman_std = np.std(stack, axis=0)
    if visualize:
        # --- Figure 1: Average Spearman heatmap ---
        fig_hm, ax_hm = plt.subplots(figsize=(1.1*P + 2, 1.1*P + 2), dpi=figure_dpi)
        im = ax_hm.imshow(spearman_mean, cmap='coolwarm', vmin=-1, vmax=1)
        ax_hm.set_xticks(range(P))
        ax_hm.set_xticklabels(prop_names, rotation=45, ha='right')
        ax_hm.set_yticks(range(P))
        ax_hm.set_yticklabels(prop_names)
        ax_hm.set_title("Average Spearman rank correlation across match sets")
        for i in range(P):
            for j in range(P):
                val = spearman_mean[i, j]
                ax_hm.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color='black' if abs(val) < 0.75 else 'white',
                        fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        plt.tight_layout()

        # --- Figure 2: Scatter matrix from one valid set, annotated with average Spearman (mean±std) ---
        # Choose a valid set index for scatter (fallback to first valid if the requested one isn't valid)
        if scatter_set_index not in valid_set_indices:
            scatter_set_index = valid_set_indices[0]

        # Recompute orders for the selected set (to avoid storing all intermediate per-set data)
        matches = list(iter(match_sets[scatter_set_index]))
        orders = {}
        universes = {}
        for p in properties:
            idxs = []
            for i, m in enumerate(matches):
                if ignore_negatives_in_same_sequence and (not m.is_correct) and m.is_in_same_sequece:
                    continue
                idxs.append(i)
            scores = []
            for i in idxs:
                if p.name not in matches[i].match_properties:
                    raise KeyError(f"Match at index {i} in scatter set {scatter_set_index} is missing '{p.name}'.")
                v = float(matches[i].match_properties[p.name])
                scores.append(v if p.higher_is_better else -v)
            pairs = list(zip(idxs, scores))
            pairs.sort(key=lambda x: (x[1], -x[0]), reverse=True)
            orders[p.name] = [i for i, _ in pairs]
            universes[p.name] = idxs

        common_universe_scatter = set(universes[prop_names[0]])
        for name in prop_names[1:]:
            common_universe_scatter &= set(universes[name])
        common_universe_scatter = sorted(list(common_universe_scatter))

        if len(common_universe_scatter) >= 2:
            restricted_orders_scatter = {}
            for name in prop_names:
                restrict = []
                present = set(common_universe_scatter)
                for idx in orders[name]:
                    if idx in present:
                        restrict.append(idx)
                restricted_orders_scatter[name] = restrict

            rankpos_scatter = {}
            for name in prop_names:
                rp = {}
                for r, idx in enumerate(restricted_orders_scatter[name]):
                    rp[idx] = r
                rankpos_scatter[name] = rp

            # Subsample for clarity
            n_items = len(common_universe_scatter)
            rng = random.Random(42)
            plot_indices = common_universe_scatter
            if (sample_scatter is not None) and (n_items > sample_scatter):
                plot_indices = rng.sample(common_universe_scatter, sample_scatter)

            # Create scatter matrix
            fig_sc, axes = plt.subplots(P, P, figsize=(3*P, 3*P), dpi=figure_dpi, squeeze=False)
            max_rank = n_items - 1
            for i in range(P):
                for j in range(P):
                    ax = axes[i, j]
                    if i == j:
                        ax.axis('off')
                        ax.text(0.5, 0.5, prop_names[i], ha='center', va='center',
                                fontsize=12, fontweight='bold')
                    elif i > j:
                        # Lower triangle: scatter ranks (from chosen set)
                        x = np.array([rankpos_scatter[prop_names[j]][idx] for idx in plot_indices], dtype=int)
                        y = np.array([rankpos_scatter[prop_names[i]][idx] for idx in plot_indices], dtype=int)
                        ax.scatter(x, y, s=8, alpha=0.5)
                        ax.plot([0, max_rank], [0, max_rank], color='gray', lw=1, linestyle='--', alpha=0.8)
                        ax.set_xlim(0, max_rank)
                        ax.set_ylim(0, max_rank)
                        ax.set_xlabel(f"Rank: {prop_names[j]}")
                        ax.set_ylabel(f"Rank: {prop_names[i]}")
                        # Title uses average Spearman for (i, j)
                        ax.set_title(f"ρ={spearman_mean[i, j]:.2f}")
                    else:
                        # Upper triangle: display mean±std of Spearman across sets
                        ax.axis('off')
                        mu = spearman_mean[i, j]
                        sd = spearman_std[i, j]
                        ax.text(0.5, 0.5, f"ρ={mu:.3f}±{sd:.3f}",
                                ha='center', va='center', fontsize=11, fontweight='bold',
                                color='black' if abs(mu) < 0.75 else 'darkgreen')
            plt.tight_layout()
        else:
            # If we can't form a scatter (too few common items), inform via a lightweight figure
            fig_sc, ax = plt.subplots(figsize=(5, 2), dpi=figure_dpi)
            ax.axis('off')
            ax.text(0.5, 0.5, "Scatter matrix skipped (insufficient common universe in selected set)",
                    ha='center', va='center', fontsize=11)

        plt.show()
    return spearman_mean
    # return {
    #     "spearman_mean": spearman_mean,
    #     "spearman_std": spearman_std,
    #     "num_valid_sets": len(spearman_matrices),
    #     "valid_set_indices": valid_set_indices
    # }