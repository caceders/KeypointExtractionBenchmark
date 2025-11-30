from typing import Callable, Tuple
import cv2
import numpy as np

KERNEL_IMAGE_RATIO = 0.01

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
    use_normalisation: bool = False
        If true then measurement area normalisation is used. The image is resized to have a constant ratio between detector
        and descriptor kernel and image width. If used detection region size and description region size is required.
    detection_region_size: int | None = None
        The kernel size of the detector.
    description_region_size: int | None = None
        The kernel size of the descriptor
    """
    def __init__(self,
                detect_keypoints : Callable[[np.ndarray], list[cv2.KeyPoint]],
                describe_keypoints: Callable[[np.ndarray, list[cv2.KeyPoint]], list[np.ndarray]],
                use_normalisation: bool = False,
                detection_region_size: int | None = None,
                description_region_size: int | None = None
                ):
        self._detect_keypoints = detect_keypoints
        self._describe_keypoints = describe_keypoints
        self._use_normalisation = use_normalisation
        
        if use_normalisation:
            if ((not detection_region_size) or (not description_region_size)):
                raise TypeError("To use normalisation pass detection and description region size")
            else:
                self.detection_region_size = detection_region_size
                self.description_region_size = description_region_size
        

    @classmethod
    def from_opencv(cls,
                opencv_detect_keypoints : Callable[[np.ndarray], list[cv2.KeyPoint]],
                opencv_describe_keypoints: Callable[[np.ndarray, list[cv2.KeyPoint]], list[np.ndarray]],
                use_normalisation: bool = False,
                detection_region_size: int | None = None,
                descrption_region_size: int | None = None
                ):
        
        
        def describe_keypoint_wrapper(img: np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
            _, descs = opencv_describe_keypoints(img, keypoints)
            return list(descs)
        

        return cls(
            detect_keypoints = opencv_detect_keypoints,
            describe_keypoints = describe_keypoint_wrapper,
            use_normalisation = use_normalisation,
            detection_region_size = detection_region_size,
            description_region_size = descrption_region_size,
            )

    def get_detection_image_scale_factor(self, image_size)->float:
        if not self._use_normalisation:
            return 1
        return (self.detection_region_size/(image_size * KERNEL_IMAGE_RATIO))
    
    def get_description_image_scale_factor(self, image_size)->float:
        if not self._use_normalisation:
            return 1
        return (self.description_region_size/(image_size * KERNEL_IMAGE_RATIO))
    
    def detect_keypoints(self, img: np.ndarray) -> list[cv2.KeyPoint]:
        if self._use_normalisation:
            resize_factor = self.get_detection_image_scale_factor(len(img[0][0]))
            img_copy = img.copy()

            new_shape_x, new_shape_y = img.shape[:2] # We only need height and width
            new_shape_x = int(round(float(new_shape_x) * resize_factor))
            new_shape_y = int(round(float(new_shape_y) * resize_factor))

            img_copy = np.array(cv2.resize(img, (new_shape_x, new_shape_y), interpolation=cv2.INTER_CUBIC))
            return self._detect_keypoints(img_copy)
        else:
            return self._detect_keypoints(img)

    
    def describe_keypoints(self, img: np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        if self._use_normalisation:
            image_resize_factor = self.get_description_image_scale_factor(len(img[0][0]))
            img_copy = img.copy()

            new_shape_x, new_shape_y = img.shape[:2] # We only need height and width
            new_shape_x = int(round(float(new_shape_x) * image_resize_factor))
            new_shape_y = int(round(float(new_shape_y) * image_resize_factor))


            ## Copy keypoints and change center and size according to scale change.
            keypoints_copy = []
            for keypoint in keypoints:
                description_image_scale_factor = self.get_description_image_scale_factor(len(img[0][0]))
                detection_image_scale_factor = self.get_detection_image_scale_factor(len(img[0][0]))
                keypoint_resize_factor = description_image_scale_factor/detection_image_scale_factor
                keypoint_copy_x = keypoint.pt[0] * keypoint_resize_factor
                keypoint_copy_y = keypoint.pt[1] * keypoint_resize_factor
                keypoint_copy_size = keypoint.size * keypoint_resize_factor
                keypoint_copy = cv2.KeyPoint(keypoint_copy_x, keypoint_copy_y, keypoint_copy_size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)
                keypoints_copy.append(keypoint_copy)

            img_copy = np.array(cv2.resize(img_copy, (new_shape_x, new_shape_y), interpolation=cv2.INTER_CUBIC))
            return self._describe_keypoints(img_copy, keypoints_copy)
        else:
            return self._describe_keypoints(img, keypoints)
