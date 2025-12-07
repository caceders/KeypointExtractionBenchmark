from typing import Callable, Tuple
import cv2
import numpy as np
from timeit import default_timer as timer
from beartype import beartype


MEASUREMENT_AREA_NORMALISATION_KERNEL_IMAGE_RATIO = 0.01

class FeatureExtractor:
    """
    Wrapper class for a feature extraction method.

    Attributes
    ----------
    detect_keypoints : Callable[[list[np.ndarray]], list[cv2.KeyPoint]]
        function for detecting keypoints. Needs to take in an image on the form of a numpy
        array and return a python list of openCV keypoints.
    describe_keypoints: Callable[[list[np.ndarray], list[cv2.KeyPoint]], list[np.ndarray]]
        function for detecting keypoints. Needs to take in python list of openCV keypoints
        and an image on the from of a numpy array and return the descriptor vectors on the form of a numpy array.
    """
    #@beartype
    def __init__(self,
                detect_keypoints : Callable[[np.ndarray], list[cv2.KeyPoint]],
                describe_keypoints: Callable[[np.ndarray, list[cv2.KeyPoint]], list[np.ndarray]],
                distance_type: int,
                ):
        
        if distance_type not in [cv2.NORM_HAMMING, cv2.NORM_L2]:
            raise TypeError("Distance type needs to be cv2.NORM_HAMMING or cv2.NORM_L2")

        self._detect_keypoints = detect_keypoints
        self._describe_keypoints = describe_keypoints
        self.distance_type = distance_type

    #@beartype
    @classmethod
    def from_opencv(cls,
                opencv_detect_keypoints : Callable[[np.ndarray], list[cv2.KeyPoint]],
                opencv_describe_keypoints: Callable[[np.ndarray, list[cv2.KeyPoint]], list[np.ndarray]],
                distance_type: int,
                ):
                
        def detect_keypoint_wrapper(image: np.ndarray) -> list[cv2.KeyPoint]:
            keypoints = opencv_detect_keypoints(image)
            return list(keypoints)


        def describe_keypoint_wrapper(image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
            _, descs = opencv_describe_keypoints(image, keypoints)
            return list(descs)
        

        return cls(
            detect_keypoints = detect_keypoint_wrapper,
            describe_keypoints = describe_keypoint_wrapper,
            distance_type = distance_type,
            )


    #@beartype
    def detect_keypoints(self, image: np.ndarray) -> list[cv2.KeyPoint]:
        return self._detect_keypoints(image)

    #@beartype
    def describe_keypoints(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        return self._describe_keypoints(image, keypoints)
        
    #@beartype
    def get_extraction_time_on_image(self, image: np.ndarray) -> float:
        start = timer()
        keypoints = self._detect_keypoints(image)
        self._describe_keypoints(image, keypoints)
        end = timer()
        time = end - start
        return time