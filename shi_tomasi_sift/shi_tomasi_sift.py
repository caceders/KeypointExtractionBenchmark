import cv2
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class shi_tomasi_sift():

    # NB: OpenCV operates with (y, x), while matplotlib and numpy operates with (x, y). Be sure to convert correctly!

    # region Public

    def __init__(self) -> None:
        pass


    def detect(self, img : NDArray) -> list[cv2.KeyPoint]:
        pass
    

    def compute(self, img : NDArray) -> Tuple[list[cv2.KeyPoint], list[NDArray]]:
        pass


    def detect_and_compute(self, img : NDArray) -> Tuple[list[cv2.KeyPoint], list[NDArray]]:
        pass
    

    # endregion


    # region Private


    # region Shi Tomasi


    def _calculate_Ix_and_Iy(self, I : NDArray) -> Tuple[NDArray,  NDArray]:
        pass


    def _calculate_Ixx_Ixy_and_Iyy(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        pass


    def _caclculate_structure_tensors(self, Ix : NDArray, Iy : NDArray) -> NDArray:
        Ixx, Ixy, Iyy = self._calculate_Ixx_Ixy_and_Iyy(Ix, Iy)
        pass


    def _calculate_response(self, Ix : NDArray, Iy : NDArray) -> NDArray:
        structure_tensors = self._caclculate_structure_tensors(Ix, Iy)
        pass


    def _threshold_response(self, response : NDArray) -> NDArray:
        pass


    def _perform_non_maxmum_supression(self, response : NDArray) -> NDArray:
        pass


    def _get_keypoint_positions(self, response) -> NDArray:
        pass        


    # endregion


    # region SIFT
    

    def _calculate_magnitude_and_angle(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray]:
        pass

    
    def _weight_area_with_gaussian_window(self, area: NDarray, gaussian_std: float) -> NDArray:
        pass


    def _calculate_histogram(self, array : NDArray, weights : NDArray, num_bins : int, start = None, stop = None, cyclic : bool = False) -> Tuple[NDArray, NDArray]:
        '''
        Returns ([bins], [values])
        '''
        pass


    def _get_angle_and_keypoints_from_histogram(self, bins : NDArray, values : NDArray) -> list[Tuple[float, Tuple[int, int]]]:
        '''
        Returns a list of (angle, (x, y))
        '''
        pass

    
    def _extract_area(self, matrix : NDArray, center : Tuple[int, int], size : int) -> NDArray:  
        pass

    
    def _rotate_matrix_around_center(self, matrix : NDArray, angle: float) -> NDArray:
        '''
        Returns the new "rotated" pixel positions, based on the indexes, around the center of the matrix
        '''

    
    
    # endregion

    # region Debug and tools

    def extract_area(self, matrix: NDArray, center: Tuple[int, int], size : int) -> NDArray:
        '''
        Returns a subarea in the matrix around the center with window size (size, size)
        '''
        pass

    def plot_image(self, ax : Axes, image: NDArray, center : Tuple[int, int] = None, size : int = None, plot_title: str = None) -> None:
        '''
        Plots an image or a subregion of an image
        '''
        pass

    def plot_magnitude_and_orientation(self, ax : Axes, magnitude: NDArray, orientation : NDArray, center: Tuple[int, int] = None, size: int = None, plot_title: str = None) -> None:
        '''
        Plots an image of the magniutude with arrows as orientation
        '''
        pass

    def plot_histogram(self, hist : NDArray, bins: NDArray):
        pass


    # Ideas:
    # - Extract area
    # - Plot window (center + size around)
    # - Plot magnitude w/arrows
    # - Plot histograms
    # - 

    # endregion

    # endregion