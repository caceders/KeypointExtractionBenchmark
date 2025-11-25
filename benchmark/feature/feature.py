import numpy as np
from typing import Tuple
import cv2

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
    def __init__(self, keypoint: cv2.KeyPoint, description: np.ndarray, sequence_index: int, image_index: int):

        if not isinstance(keypoint, cv2.KeyPoint): raise(TypeError("Keypoint must be of type cv2.KeyPoint"))
        if not isinstance(description, np.ndarray): raise(TypeError("Descriptor must be of type np.ndarray"))


        self.keypoint: cv2.KeyPoint = keypoint
        self.description: np.ndarray = description
        self.sequence_index = sequence_index
        self.image_index = image_index
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
    def pt(self)->np.ndarray[int]:
        return np.array([self.keypoint.pt[0], self.keypoint.pt[0]])

    def get_pt_after_homography_transform(self, H) -> Tuple[float, float]:
        x, y = self.pt
        v = H @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]
    
    def get_size_after_homography_transform(self, H):
        x, y = self.keypoint.pt
        r = self.keypoint.size / 2

        # Four sample points around the keypoint
        pts = np.array([
            [x + r, y],
            [x - r, y],
            [x, y + r],
            [x, y - r]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Transform the center
        center = np.array([[[x, y]]], dtype=np.float32)
        center_t = cv2.perspectiveTransform(center, H).reshape(2)

        # Transform the sample points
        pts_t = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

        # Measure new radius as the average stretch
        new_r = np.mean(np.linalg.norm(pts_t - center_t, axis=1))
        return 2 * new_r  # keypoint size