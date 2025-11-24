import cv2
import numpy as np
from typing import Callable, Tuple

BASE_REGION_SIZE = 8

class FeatureExtractor:
    def __init__(self,
                detect_kps_func : Callable[[list[np.ndarray]], list[cv2.KeyPoint]],
                describe_kps_func: Callable[[list[np.ndarray], list[cv2.KeyPoint]], list[np.ndarray]],
                use_normalisation: bool = False,
                detection_region_size: int | None = None,
                descrption_region_size: int | None = None
                ):
        self._detect_kps_func = detect_kps_func
        self._describe_kps_func = describe_kps_func
        
        if use_normalisation and ((not detection_region_size) or (not descrption_region_size)):
            raise TypeError("To use normalisation pass detection and description region size")
        
        self.use_normalisation = use_normalisation
        self.detection_region_size = detection_region_size
        self.description_region_size = descrption_region_size

    @classmethod
    def from_opencv(cls,
                opencv_detect_kps_func : Callable[[list[np.ndarray]], list[cv2.KeyPoint]],
                opencv_describe_kps_func: Callable[[list[np.ndarray], list[cv2.KeyPoint]], Tuple[list[cv2.KeyPoint], list[np.ndarray]]],
                use_normalisation: bool = False,
                detection_region_size: int | None = None,
                descrption_region_size: int | None = None
                ):
        
        
        def describe_kp_func_wrapper(img: list[np.ndarray], kps: list[cv2.KeyPoint]) -> list[np.ndarray]:
            _, descs = opencv_describe_kps_func(img, kps)
            return descs
        

        return cls(
            detect_kps_func = opencv_detect_kps_func,
            describe_kps_func = describe_kp_func_wrapper,
            use_normalisation = use_normalisation,
            detection_region_size = detection_region_size,
            descrption_region_size = descrption_region_size,
            )
    
    def detect_kps(self, img: np.ndarray) -> list[cv2.KeyPoint]:
        if self.use_normalisation:
            resize_factor = self.detection_region_size / BASE_REGION_SIZE
            img_copy = img.copy()

            new_shape_x, new_shape_y = img.shape[:2] # We only need height and width
            new_shape_x = int(round(float(new_shape_x) * resize_factor))
            new_shape_y = int(round(float(new_shape_y) * resize_factor))

            img_copy = cv2.resize(img, (new_shape_x, new_shape_y), interpolation=cv2.INTER_CUBIC)
        return self._detect_kps_func(img_copy)
    
    def describe_kps(self, img: np.ndarray, kps: list[cv2.KeyPoint]) -> list[np.ndarray]:
        if self.use_normalisation:
            image_resize_factor = self.description_region_size / BASE_REGION_SIZE
            img_copy = img.copy()

            new_shape_x, new_shape_y = img.shape[:2] # We only need height and width
            new_shape_x = int(round(float(new_shape_x) * image_resize_factor))
            new_shape_y = int(round(float(new_shape_y) * image_resize_factor))


            ## Copy keypoints and change center and size according to scale change.
            keypoint_resize_factor = self.description_region_size / self.detection_region_size
            kps_copy = []
            for kp in kps:
                kp_copy_x, kp_copy_y, kp_copy_size = (kp.pt[0] * keypoint_resize_factor, kp.pt[1] * keypoint_resize_factor, kp.size * keypoint_resize_factor)
                kp_copy = cv2.KeyPoint(kp_copy_x, kp_copy_y, kp_copy_size, kp.angle, kp.response, kp.octave, kp.class_id)
                kps_copy.append(kp_copy)

            img_copy = cv2.resize(img_copy, (new_shape_x, new_shape_y), interpolation=cv2.INTER_CUBIC)
        return self._describe_kps_func(img_copy, kps_copy)