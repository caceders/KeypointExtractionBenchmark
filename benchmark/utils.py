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