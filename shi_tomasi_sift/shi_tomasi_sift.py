import cv2
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

class ShiTomasiSift():

    # NB: OpenCV operates with (y, x), while matplotlib and numpy operates with (x, y). Be sure to convert correctly!

    # region Public

    def __init__(self) -> None:
        pass


    def detect(self, img : NDArray, quality_level = 0.01, max_corners = 1000) -> list[cv2.KeyPoint]:
        Ix, Iy = self._calculate_Ix_and_Iy(img)
        response = self._calculate_response(Ix, Iy)
        threshold = np.max(response) * quality_level
        tresholded = self._threshold_response(response, threshold)
        supressed = self._perform_non_maxmum_supression(tresholded)
        coords = self._get_keypoint_positions(supressed)

        keypoints = []

        for coord in coords:
            keypoint = cv2.KeyPoint(float(coord[1]), float(coord[0]), 3, response = float(supressed[coord[0]][coord[1]]))
            keypoints.append(keypoint)
        
        keypoints.sort(key = lambda keypoint : - keypoint.response)

        return keypoints[:max_corners]
    

    def compute(self, img : NDArray) -> Tuple[list[cv2.KeyPoint], list[NDArray]]:
        pass


    def detect_and_compute(self, img : NDArray) -> Tuple[list[cv2.KeyPoint], list[NDArray]]:
        pass
    

    # endregion


    # region Private


    # region Shi Tomasi


    def _calculate_Ix_and_Iy(self, I : NDArray) -> Tuple[NDArray,  NDArray]:
        sobel_x = np.array(
            [[-1, 0 , 1],
             [-2, 0, 2],
             [-1, 0, 1]]
             )
        
        sobel_y = np.array(
            [[-1, -2 , -1],
             [0, 0, 0],
             [1, 2, 1]])
        
        Ix = convolve2d(I, sobel_x)
        Iy = convolve2d(I, sobel_y)

        return (Ix, Iy)


    def _calculate_Ixx_Ixy_and_Iyy(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        
        return (Ixx, Ixy, Iyy)


    def _caclculate_structure_tensors(self, Ix : NDArray, Iy : NDArray, window: NDArray | None = None) -> Tuple[NDArray, NDArray, NDArray]:
        Ixx, Ixy, Iyy = self._calculate_Ixx_Ixy_and_Iyy(Ix, Iy)
        if window is None:
            window = np.ones((3,3)) * 1/9
        A = convolve2d(Ixx, window)
        B = convolve2d(Ixy, window)
        C = convolve2d(Iyy, window)
        
        return A, B, C


    def _calculate_response(self, Ix : NDArray, Iy : NDArray) -> NDArray:
        A, B, C = self._caclculate_structure_tensors(Ix, Iy)
        response = 0.5 * (A + C) - 0.5 * (np.abs(A - C) + 2 * np.abs(B))
        return response


    def _threshold_response(self, response : NDArray, threshold: float) -> NDArray:
        thresholded = response.copy()
        thresholded[thresholded < threshold] = 0
        return thresholded


    def _perform_non_maxmum_supression(self, response : NDArray, size : int = 3) -> NDArray:
        local_max = maximum_filter(response, size)
        mask = np.isclose(response, local_max)
        supressed = np.zeros_like(response)
        supressed[mask] = response[mask]

        return supressed

    def _get_keypoint_positions(self, response) -> NDArray:
        coords = np.column_stack(np.nonzero(response)) #(y, x)
        return coords



    # endregion


    # region SIFT
    

    def _calculate_magnitude_and_angle(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray]:
        magnitude = np.sqrt(Ix*Ix + Iy*Iy)
        angle = np.arctan2(Ix, Iy)

        return (magnitude, angle)

    
    def _weight_area_with_gaussian_window(self, area: NDArray, gaussian_std: float) -> NDArray:
        pass


    def _calculate_histogram(self,
                             array : NDArray,
                             weights : NDArray,
                             num_bins : int,
                             start = None,
                             stop = None,
                             cyclic : bool = False
                             ) -> Tuple[NDArray, NDArray]:
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

def extract_area(matrix: NDArray, center: Tuple[int, int], size : int) -> NDArray:
    '''
    Returns a subarea in the matrix around the center with window size (size, size). Elements in the window outside the
    border of the matrix is set to 0
    '''
    ## Calculate start and end indexes
    x_start, y_start = center[1] - size//2, center[0] - size//2
    x_end, y_end = x_start + size, y_start + size


    matrix_x_size, matrix_y_size = matrix.shape[1], matrix.shape[0]
    subarea = np.zeros((size, size), dtype=matrix.dtype)

    # Handle area outside matrix
    # x low
    if x_start >= 0:
        matrix_x_start = x_start
        subarea_x_start = 0
    else:
        matrix_x_start = 0
        subarea_x_start = -x_start

    # y low
    if y_start >= 0:
        matrix_y_start = y_start
        subarea_y_start = 0
    else:
        matrix_y_start = 0
        subarea_y_start = -y_start

    # x high
    if x_end <= matrix_x_size:
        matrix_x_end = x_end
        subarea_x_end = size
    else:
        matrix_x_end = matrix_x_size
        subarea_x_end = size - (x_end - matrix_x_size)

    # y high
    if y_end <= matrix_y_size:
        matrix_y_end = y_end
        subarea_y_end = size
    else:
        matrix_y_end = matrix_y_size
        subarea_y_end = size - (y_end - matrix_y_size)

    subarea[subarea_y_start : subarea_y_end, subarea_x_start : subarea_x_end] = matrix[matrix_y_start : matrix_y_end, matrix_x_start : matrix_x_end]

    return subarea


def plot_image(ax : Axes,
               image: NDArray,
               center : Tuple[int, int] = None,
               size : int = None,
               plot_title: str = None,
               show_values : bool = False,
               value_color : str = "r"
               ) -> None:
    '''
    Plots an image or a subregion of an image
    '''

    if center is None:
        area = image
    else:
        area = extract_area(image, center, size)

    vmin = np.min(image)
    vmax = np.max(image)

    if plot_title is not None:
        ax.set_title(plot_title)

    ax.imshow(area, cmap = "gray", vmin=vmin, vmax=vmax)

    if show_values:
        for y_index in range(area.shape[0]):
            for x_index in range(area.shape[1]):

                # Make value displayed as scientific notation
                value_text = f"{area[y_index][x_index]:.2e}"
                texts = value_text.split("e")
                texts[0] += ("\n")
                value_text = texts[0] + texts[1]

                ax.text(x_index, y_index, value_text, color = value_color, fontsize = "x-small", horizontalalignment = "center", verticalalignment='center')
    

def plot_magnitude_and_angle(ax : Axes,
                                   image: NDArray,
                                   magnitude: NDArray,
                                   angle : NDArray,
                                   center: Tuple[int, int] = None,
                                   size: int = None,
                                   plot_title: str = None
                                   ) -> None:
    '''
    Plots an image of with arrows as angle and respective magnitudes. Orientation goes from [-PI, PI] with respect to the x axis
    '''
    if center is None:
        image_area = image
        magnitude_area = magnitude
        angle_area = angle
    else:
        image_area = extract_area(image, center, size)
        magnitude_area = extract_area(magnitude, center, size)
        angle_area = extract_area(angle, center, size)
    
    plot_image(ax, image, center, size, plot_title)

    for y_index in range(magnitude_area.shape[0]):
        for x_index in range(magnitude_area.shape[1]):
            dx = np.sin(angle_area[y_index][x_index]) * magnitude_area[y_index][x_index] / np.max(magnitude_area)
            dy = np.cos(angle_area[y_index][x_index]) * magnitude_area[y_index][x_index] / np.max(magnitude_area)


            ax.annotate("", xytext=(x_index, y_index), xy=(x_index + dx, y_index + dy), arrowprops=dict(arrowstyle="->"))


def plot_histogram(hist : NDArray, bins: NDArray):
    pass

    # endregion

    # endregion