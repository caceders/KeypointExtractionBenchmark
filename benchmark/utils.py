from typing import Tuple
import numpy as np
import os
import cv2

def load_HPSequences(path_to_HPSequences: str) -> Tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    image_sequences: list[list[np.ndarray]] = []
    homography_sequences: list[list[np.ndarray]] = []

    # iterate over subfolders
    for name in os.listdir(path_to_HPSequences):
        subfolder = os.path.join(path_to_HPSequences, name)
        if not os.path.isdir(subfolder):
            continue

        images = []
        homographies = []

        for fname in sorted(os.listdir(subfolder)):
            fpath = os.path.join(subfolder, fname)

            if fname.lower().endswith(".ppm"):
                img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to load image: {fpath}")
                images.append(img)

            elif fname.startswith("H_"):
                H = np.loadtxt(fpath)
                homographies.append(H)

        image_sequences.append(images)
        homography_sequences.append(homographies)

    return image_sequences, homography_sequences

