import cv2
import numpy as np
from typing import Callable, Tuple

class FeatureExtractor:
    def __init__(self,
                detect_kps_func : Callable[[list[np.ndarray]], list[cv2.KeyPoint]],
                describe_kps_func: Callable[[list[np.ndarray], list[cv2.KeyPoint]], list[np.ndarray]]
                ):
        self._detect_kps_func = detect_kps_func
        self._describe_kps_func = describe_kps_func

    @classmethod
    def from_opencv(cls,
                opencv_detect_kps_func : Callable[[list[np.ndarray]], list[cv2.KeyPoint]],
                opencv_describe_kps_func: Callable[[list[np.ndarray], list[cv2.KeyPoint]], Tuple[list[cv2.KeyPoint], list[np.ndarray]]]
                ):
        
        
        def describe_kp_func_wrapper(img: list[np.ndarray], kps: list[cv2.KeyPoint]) -> list[np.ndarray]:
            _, descs = opencv_describe_kps_func(img, kps)
            return descs
        

        return cls(
            detect_kps_func = opencv_detect_kps_func,
            describe_kps_func = describe_kp_func_wrapper
            )
    
    def detect_kps(self, img: np.ndarray) -> list[cv2.KeyPoint]:
        return self._detect_kps_func(img)
    
    def describe_kps(self, img: np.ndarray, kps: list[cv2.KeyPoint]) -> list[np.ndarray]:
        return self._describe_kps_func(img, kps)