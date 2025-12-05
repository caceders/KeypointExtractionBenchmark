from typing import Callable, Tuple
import cv2
import numpy as np
from timeit import default_timer as timer


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
    def __init__(self,
                detect_keypoints : Callable[[np.ndarray], list[cv2.KeyPoint]],
                describe_keypoints: Callable[[np.ndarray, list[cv2.KeyPoint]], list[np.ndarray]],
                distance_type: int,
                ):
        
        if not isinstance(detect_keypoints, Callable): raise(TypeError("detect keypoint callback must be Callable"))
        if not isinstance(describe_keypoints, Callable): raise(TypeError("describe keypoint callback must be Callable"))

        if distance_type not in [cv2.NORM_HAMMING, cv2.NORM_L2]:
            raise TypeError("Distance type needs to be cv2.NORM_HAMMING or cv2.NORM_L2")

        self._detect_keypoints = detect_keypoints
        self._describe_keypoints = describe_keypoints
        self.distance_type = distance_type


    @classmethod
    def from_opencv(cls,
                opencv_detect_keypoints : Callable[[np.ndarray], list[cv2.KeyPoint]],
                opencv_describe_keypoints: Callable[[np.ndarray, list[cv2.KeyPoint]], list[np.ndarray]],
                distance_type: int,
                ):
        
        if not isinstance(opencv_detect_keypoints, Callable): raise(TypeError("detect keypoint callback must be callable"))
        if not isinstance(opencv_describe_keypoints, Callable): raise(TypeError("describe keypoint callback must be callable"))
        
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


    
    def detect_keypoints(self, image: np.ndarray) -> list[cv2.KeyPoint]:
        if not isinstance(image, np.ndarray): raise TypeError("Image must be of type np.ndarray")
        return self._detect_keypoints(image)

    
    def describe_keypoints(self, image: np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        if not isinstance(image, np.ndarray): raise TypeError("Image must be of type np.ndarray")
        if not isinstance(keypoints, list) or (len(keypoints) != 0 and not all(isinstance(keypoint, cv2.KeyPoint) for keypoint in keypoints)): raise TypeError("keypoints must be a list of cv2.KeyPoints")
        return self._describe_keypoints(image, keypoints)
        

    def get_extraction_time_on_image(self, image: np.ndarray) -> float:
        if not isinstance(image, np.ndarray): raise TypeError("Image must be of type np.ndarray")
        start = timer()
        keypoints = self._detect_keypoints(image)
        self._describe_keypoints(image, keypoints)
        end = timer()
        time = end - start
        return time