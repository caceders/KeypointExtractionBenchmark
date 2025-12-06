from typing import Tuple
import cv2
import numpy as np
import os

def load_HPSequences(path_to_HPSequences: str) -> Tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """
    Load the HPSequence dataset:
    by passing the path to the sequence.

    Parameters
    ----------
    path_to_HPSequences: str
        The path to the extracted HPsequences dataset.

    Returns
    -------
    Tuple[list[list[np.ndarray]], list[list[np.ndarray]]
        A 2d list of the image sequences with the respective images and a
        2d list of the homographical transformation matrixes between the
        reference image and the related images.
    """
    image_sequences: list[list[np.ndarray]] = []
    homography_sequences: list[list[np.ndarray]] = []

    # Itterate over all subfolders.
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
                homographies.append(homography)

        image_sequences.append(images)
        homography_sequences.append(homographies)

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