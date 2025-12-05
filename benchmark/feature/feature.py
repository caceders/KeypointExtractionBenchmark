from typing import Tuple
import cv2
import numpy as np

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
        if not isinstance(sequence_index, int): raise(TypeError("Sequence index need to be of type int"))
        if not isinstance(image_index, int): raise(TypeError("Image index need to be of type int"))

        self.keypoint: cv2.KeyPoint = keypoint
        self.description: np.ndarray = description
        self.sequence_index = sequence_index
        self.image_index = image_index
        self._image_valid_matches: dict[int, dict[Feature, float]] = {}
        self._all_valid_matches: dict[Feature, float] = {}

    def store_valid_match_for_image(self, related_image_index: int, feature: "Feature", score: int | float):

        if not isinstance(related_image_index, int): raise(TypeError("Related image index must be of type int"))
        if not isinstance(feature, Feature): raise(TypeError("Feature must be of type Feature"))
        if not isinstance(score, (int, float)): raise(TypeError("Score need to be of type int or float"))

        if not related_image_index in self._image_valid_matches:
            self._image_valid_matches[related_image_index] = {}

        self._image_valid_matches[related_image_index][feature] = score
        self._all_valid_matches[feature] = score
    
    def get_valid_matches_for_image(self, related_image_index: int) -> dict["Feature", float]:
        
        if not isinstance(related_image_index, int): raise(TypeError("Related image index must be of type int"))

        if not related_image_index in self._image_valid_matches:
            return {}
        return self._image_valid_matches[related_image_index].copy()
    
    def get_all_valid_matches(self) -> dict["Feature" , float]:
        return self._all_valid_matches.copy()
    
    def is_match_with_other_valid(self, other: "Feature"):
        if not isinstance(other, Feature): raise(TypeError("Other must be of type Feature"))
        return other in self._all_valid_matches

    @property
    def pt(self)->np.ndarray:
        return np.array([self.keypoint.pt[0], self.keypoint.pt[1]])

    def get_pt_after_homography_transform(self, H) -> Tuple[float, float]:
        if (not isinstance(H, np.ndarray)) or H.shape != (3,3): raise TypeError("Homography must be a 3x3 np.ndrray")
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