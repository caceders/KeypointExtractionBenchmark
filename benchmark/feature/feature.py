import numpy as np
from typing import Tuple
import cv2

class Feature:

    def __init__(self, keypoint: cv2.KeyPoint, descriptor: np.ndarray):

        self.kp: cv2.KeyPoint = keypoint
        self.desc: np.ndarray = descriptor
        self._image_valid_matches: dict["Feature", float] = {}
        self._all_valid_matches: list["Feature"] = []

    def store_valid_match_for_image(self, related_image_index: int, feature: "Feature", score: float) -> None:
        
        if not related_image_index in self._image_valid_matches:
            self._image_valid_matches[related_image_index] = {}

        self._image_valid_matches[related_image_index][feature] = score
        self._all_valid_matches.append(feature)
    
    def get_valid_matches_for_image(self, related_image_index: int) -> dict["Feature", float] | None:
        if not related_image_index in self._image_valid_matches:
            return None
        return self._image_valid_matches[related_image_index]
    
    def get_all_valid_matches(self) -> list["Feature"]:
        return self._all_valid_matches
    
    def is_match_with_other_valid(self, other: "Feature"):
        return other in self._all_valid_matches

    @property
    def pt(self):
        return self.kp.pt

    def get_pt_after_homography_transform(self, H) -> Tuple[float, float]:
        x, y = self.pt
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]
   
    