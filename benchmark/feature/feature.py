from typing import Tuple
import cv2
import numpy as np
from beartype import beartype
from config import *

class Feature:
    '''
    An image feature consisting of a keypoint and a description.

    Attributes
    ----------
    keypoint : cv2.KeyPoint
        The keypoint of the feature represented by an openCV keypoint object.
    description : np.ndarray
        Description vector of the feature.
    sequence_index : int
        The sequence index of the specific sequence this feature was found in.
    image_inedx : int
        The image inedx of the image in a sequence this feature was found in.
    '''
    
    # Beartype commented for performance reasons
    #@beartype
    def __init__(self, keypoint: cv2.KeyPoint, description: np.ndarray, sequence_index: int, image_index: int):

        self.keypoint: cv2.KeyPoint = keypoint
        self.description: np.ndarray = description
        self.sequence_index = sequence_index
        self.image_index = image_index
        self._image_valid_matches: dict[int, list[Feature]] = {}
        self._all_valid_matches: list[Feature] = []
    
    @beartype
    def store_valid_match_for_image(self, related_image_index: int, feature: "Feature"):

        if not related_image_index in self._image_valid_matches:
            self._image_valid_matches[related_image_index] = []

        self._image_valid_matches[related_image_index].append(feature)
        self._all_valid_matches.append(feature)
    
    @beartype
    def get_valid_matches_for_image(self, related_image_index: int) -> list["Feature"]:
    

        if not related_image_index in self._image_valid_matches:
            return []
        return self._image_valid_matches[related_image_index].copy()
    

    def get_all_valid_matches(self) -> list["Feature"]:
        return self._all_valid_matches.copy()
    
    @beartype
    def is_match_with_other_valid(self, other: "Feature"):
        return other in self._all_valid_matches

    @property
    def pt(self)->np.ndarray:
        return np.array([self.keypoint.pt[0], self.keypoint.pt[1]])

    @beartype
    def get_pt_after_homography_transform(self, H : np.ndarray) -> Tuple[float, float]:
        if H.shape != (3,3): raise TypeError("Homography must be a 3x3 np.ndrray")
        x, y = self.pt
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]
    
    @beartype
    def get_size_after_homography_transform(self, H : np.ndarray):
        x, y = self.keypoint.pt
        r = self.keypoint.size / 2

        # Four sample points around the keypoint
        pts = np.array([
            [x + r, y],
            [x - r, y],
            [x, y + r],
            [x, y - r]
        ], dtype=np.float32)

        # Stack with transform dimension
        pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])  # (4,3)
        center_h = np.array([x, y, 1.0], dtype=np.float32)

        # Apply homography
        pts_t_h = pts_h @ H.T
        center_t_h = H @ center_h
        pts_t = pts_t_h[:, :2] / pts_t_h[:, 2:3]
        center_t = center_t_h[:2] / center_t_h[2]

        # Measure new radius as the average stretch
        new_r = np.mean(np.linalg.norm(pts_t - center_t, axis=1))
        return 2 * new_r  # keypoint size