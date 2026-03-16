import cv2
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import maximum_filter
from scipy.signal.windows import gaussian
from scipy.ndimage import uniform_filter1d

class ShiTomasiSift():

    # TODO:
    # - Add window options to structure tensor window
    # - Changle all array indexing to (y, x) and every other reference to coordinate to (x, y)
    # - Changle all image variables to be the same (currently there are both I and img)
    # - Check everything has format
    #   - return value
    #   - expected input types
    # - Make (hist, bin) order be identical to numpy

    # NB: OpenCV operates with (y, x), while matplotlib and numpy operates with (x, y). Be sure to convert correctly!

    # region Public

    def __init__(self,
                 derivation_operator : str = "simple",
                 structure_tensor_window : NDArray | None = None,
                 response_type : str = "normal",
                 use_previous_max_when_calculating_threshold : bool = False,
                 first_max_value_when_previous_max_is_used : int = 80000,
                 quality_level : float = 0.001,
                 max_corners : int = 1000,
                 perform_non_maxima_supression : bool = True,
                 use_descriptor_window_buffer_as_orientation_calculation_window_size : bool = True,
                 orientation_calculation_window_radius : int = 8,
                 orientation_calculation_gaussian_weight_std : float = 4.5, ## 1.5 * keypoint size (3)
                 orientation_calculation_bin_count : int = 36,
                 create_new_keypoint_for_large_angle_histogram_values : bool = True,
                 large_angle_histogram_value_threshold : float = 0.8,
                 descriptor_window_radius : int = 8,
                 use_orientation_buffer: bool = True,
                 num_subwindows_per_axis : int = 4,
                 descriptor_subwindow_size : int = 4,
                 descriptor_gaussian_weight_std : float = 8, ## 16/2 = 1/2 window size # -1 for equal weighting
                 descriptor_bin_count : int = 8,
                 drop_keypoints_on_border : bool = False,
                 base_blur_sigma : float = 1.6,
                 num_octaves_in_scale_pyramid : int = 5,
                 scale_pyramid_scaling_factor: float = 1.2,
                 scale_pyramid_blur_sigma: float = 0.5, #-1 for no blur
                 calculate_orientation_for_keypoints: bool = False,
                 d_weight : float = 0.75,
                 use_circular_descriptor : bool = False,
                 n_rings : int = 16,
                 ) -> None:
        
        self.shi_tomasi = cv2.GFTTDetector.create(maxCorners=max_corners)
        
        self.derivation_operator = derivation_operator

        if self.derivation_operator not in ["simple", "sobel", "prewitt", "scharr"]:
            raise ValueError(f"Invalid derivation operator: {self.derivation_operator}. Valid is 'simple, 'sobel', 'prewitt' or 'scharr'")

        self.structure_tensor_window = structure_tensor_window
        if self.structure_tensor_window is None:
            self.structure_tensor_window = np.ones((3,3)) * 1/9

        self.response_type = response_type
        if self.response_type not in ["sftt", "normal"]:
            raise ValueError(f"Invalid response type: {self.response_type}. Valid is 'sftt' or 'normal'")

        self.use_previous_max_when_calculating_threshold = use_previous_max_when_calculating_threshold
        self.first_max_value_when_previous_max_is_used = first_max_value_when_previous_max_is_used
        self.prev_max_value = self.first_max_value_when_previous_max_is_used
        self.quality_level = quality_level
        self.max_corners = max_corners
        self.perform_non_maxima_supression = perform_non_maxima_supression

        # Check that subwindows can fit within descriptor window
        assert 2 * descriptor_window_radius + 1 > num_subwindows_per_axis * descriptor_subwindow_size

        

        self.orientation_calculation_window_size = orientation_calculation_window_radius * 2 + 1
        self.orientation_calculation_gaussian_weight_std = orientation_calculation_gaussian_weight_std
        self.orientation_calculation_bin_count = orientation_calculation_bin_count
        self.create_new_keypoint_for_large_angle_histogram_values = create_new_keypoint_for_large_angle_histogram_values
        self.large_angle_histogram_value_threshold = large_angle_histogram_value_threshold
        self.descriptor_window_size = descriptor_window_radius * 2 + 1

        # Keep a descriptor window buffer for orientation and rotation of keypoint. 
        self.descriptor_window_buffer_size = int(np.ceil(np.sqrt(2) * self.descriptor_window_size))
        if self.descriptor_window_buffer_size % 2 == 0:
            self.descriptor_window_buffer_size += 1 # Keep the window buffer odd (equal length on both sides)

        if not use_orientation_buffer:
            self.descriptor_window_buffer_size = self.descriptor_window_size
        
        if use_descriptor_window_buffer_as_orientation_calculation_window_size and use_orientation_buffer:
            self.orientation_calculation_window_size = self.descriptor_window_buffer_size

        self.descriptor_subwindow_size = descriptor_subwindow_size
        self.descriptor_gaussian_weight_std = descriptor_gaussian_weight_std
        self.descriptor_bin_count = descriptor_bin_count
        self.drop_keypoints_on_border = drop_keypoints_on_border
        self.calculate_orientation_for_keypoints = calculate_orientation_for_keypoints

        self.num_octaves_in_scale_pyramid = num_octaves_in_scale_pyramid
        self.scale_pyramid_scaling_factor = scale_pyramid_scaling_factor
        self.scale_pyramid_blur_sigma = scale_pyramid_blur_sigma
        self.d_weight = d_weight
        self.base_blur_sigma = base_blur_sigma
        self.use_circular_descriptor = use_circular_descriptor
        self.n_rings = n_rings

        # Build grid of original coordinates (x, y)
        xs, ys = np.meshgrid(np.arange(self.descriptor_window_size), np.arange(self.descriptor_window_size))
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)

        # Build grid of coordinates in buffer

        buffer_xs, buffer_ys = np.meshgrid(np.arange(self.descriptor_window_buffer_size), np.arange(self.descriptor_window_buffer_size))
        self.buffer_xs = buffer_xs.astype(np.float32)
        self.buffer_ys = buffer_ys.astype(np.float32)

        # Pre-compute circular ring weights (n_rings, N, N)
        self.circular_weights = self._calculate_circular_weights()

    def detect(self, img : NDArray,
               Ix : NDArray | None = None,
               Iy : NDArray | None = None
               ) -> list[cv2.KeyPoint]:

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints = []
        for octave in range(self.num_octaves_in_scale_pyramid):
            if octave != 0:
                img = downsample(img, self.scale_pyramid_scaling_factor, self.scale_pyramid_blur_sigma)

            #if Ix is None or Iy is None: Er vel ingen grunn til at man forventer å ha disse?
            # Jo fordi vi beregner Ix og Iy utenfor detect funksjonen i detect_and_compute, og passer
            # Ix og Iy inn i begge som argument. Tror egentlig vi bare kan regne det ut uansett da, fordi
            # beregningen av Ix og Iy er ikke en bottleneck akkurat nå.
            Ix, Iy = self._calculate_Ix_and_Iy(img)
            response = self._calculate_response(Ix, Iy)
            response = self._threshold_response(response)
            if self.perform_non_maxima_supression:
                response = self._perform_non_maximum_supression(response)
            coords = self._get_keypoint_positions(response)


            for coord in coords:
                keypoint = cv2.KeyPoint(float(coord[1]), float(coord[0]), 3 * octave * self.scale_pyramid_scaling_factor, response = float(response[coord[0], coord[1]]), octave=octave)
                keypoints.append(keypoint)
            
        keypoints.sort(key = lambda keypoint : - keypoint.response)

            # kps = self.shi_tomasi.detect(img)
            # for kp in kps:
            #     keypoint = cv2.KeyPoint(kp.pt[0], kp.pt[1], 3 * octave * self.scale_pyramid_scaling_factor, response = kp.response, octave=octave)
            #     keypoints.append(keypoint)
        # keypoints.sort(key = lambda keypoint : - keypoint.response)
            

        return keypoints[:self.max_corners]
    

    def compute(self, img : NDArray,
                keypoints : list[cv2.KeyPoint],
                Ix : NDArray | None = None,
                Iy : NDArray | None = None
                ) -> Tuple[list[cv2.KeyPoint], list[NDArray]]:
        
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        descriptors = []
        new_keypoints = []

        # Base blur
        if self.base_blur_sigma != -1:
            img = cv2.GaussianBlur(img, (0, 0), self.base_blur_sigma)

        for octave in range(self.num_octaves_in_scale_pyramid):
            if octave != 0:
                img = downsample(img, self.scale_pyramid_scaling_factor, self.scale_pyramid_blur_sigma)


            if self.drop_keypoints_on_border:
                keypoints = self._drop_keypoints_on_border(keypoints, img)
            
            Ix, Iy = self._calculate_Ix_and_Iy(img)
            magnitude, pixel_angles = self._calculate_magnitude_and_angle(Ix, Iy)

            for keypoint in keypoints:
                if keypoint.octave != octave:
                    continue
                if self.calculate_orientation_for_keypoints:
                    orientation_calculation_area_magnitude = extract_area(magnitude,
                                                                        (int(keypoint.pt[0]), int(keypoint.pt[1])),
                                                                        self.orientation_calculation_window_size,
                                                                        "edge")
                    orientation_calculation_area_angle = extract_area(pixel_angles,
                                                                    (int(keypoint.pt[0]), int(keypoint.pt[1])),
                                                                    self.orientation_calculation_window_size,
                                                                    "edge")
                    
                    orientation_calculation_weighted_magnitude = weight_area_with_gaussian_window(orientation_calculation_area_magnitude, self.orientation_calculation_gaussian_weight_std)
                    orientation_bins, orientation_histogram = self._calculate_histogram(orientation_calculation_area_angle,
                                                                                        orientation_calculation_weighted_magnitude,
                                                                                        self.orientation_calculation_bin_count,
                                                                                        0,
                                                                                        360)
                    # smoothed_histogram = self._smooth_histogram_circular(orientation_histogram)
                    # kp_angles = self._get_angles_from_histogram(smoothed_histogram, orientation_bins)
                    kp_angles = self._get_angles_from_histogram(orientation_histogram, orientation_bins)
                else:
                    if keypoint.angle == -1:
                        kp_angles = [0]
                    else:
                        kp_angles = [keypoint.angle]

                description_area_magnitude = extract_area(magnitude,
                                                        (int(keypoint.pt[0]), int(keypoint.pt[1])),
                                                            self.descriptor_window_buffer_size)
                
                description_area_angle = extract_area(pixel_angles,
                                                    (int(keypoint.pt[0]), int(keypoint.pt[1])),
                                                    self.descriptor_window_buffer_size)
                

                if (self.descriptor_gaussian_weight_std == -1):
                    description_weighted_magnitude = description_area_magnitude
                else:
                    description_weighted_magnitude = weight_area_with_gaussian_window(description_area_magnitude, self.descriptor_gaussian_weight_std)
                

                for kp_angle in kp_angles:

                    # Rotate angles and coordinates
                    rotated_description_area_angles = description_area_angle - kp_angle
                    rotated_description_area_angles %= 360
                    rotated_coordinates = self._rotate_coordinates_around_center(-kp_angle)

                    if self.use_circular_descriptor:
                        # Circular/ring descriptor: each ring is one "cell"
                        # Shape: (n_rings, descriptor_bin_count) — same 128 dims as 4×4×8 by default
                        descriptor = np.zeros((self.n_rings, self.descriptor_bin_count), dtype=np.float64)

                        for ring_index in range(self.n_rings):
                            total_weights = description_weighted_magnitude * self.circular_weights[ring_index]
                            _, values = self._calculate_histogram(rotated_description_area_angles,
                                                                total_weights,
                                                                self.descriptor_bin_count,
                                                                0,
                                                                360,
                                                                True)
                            descriptor[ring_index] = values

                    else:
                        # Original stacked subwindow descriptor
                        num_subwindows_along_axis = self.descriptor_window_size // self.descriptor_subwindow_size

                        subwindow_positions = self._calculate_descriptor_subwindow_center_positions()

                        descriptor = np.ndarray((num_subwindows_along_axis, num_subwindows_along_axis, self.descriptor_bin_count))

                        positional_weights = self._calculate_positional_weights_with_respect_to_subwindows(rotated_coordinates, subwindow_positions)

                        for y_index in range(positional_weights.shape[0]):
                            for x_index in range(positional_weights.shape[0]):
                                total_weights = description_weighted_magnitude * positional_weights[y_index, x_index]
                                _, values = self._calculate_histogram(rotated_description_area_angles,
                                                                    total_weights,
                                                                    self.descriptor_bin_count,
                                                                    0,
                                                                    360,
                                                                    True)
                                descriptor[y_index, x_index] = values

                    descriptor = descriptor.flatten()
                    descriptor = descriptor / np.linalg.norm(descriptor)
                    descriptor[descriptor > 0.2] = 0.2
                    descriptor = descriptor / np.linalg.norm(descriptor)
                    
                    new_keypoint = self._create_new_keypoint(keypoint, kp_angle)

                    new_keypoints.append(new_keypoint)
                    descriptors.append(descriptor)
        
        return (new_keypoints, descriptors)
                
            

    def detect_and_compute(self, img : NDArray) -> Tuple[list[cv2.KeyPoint], list[NDArray]]:
        Ix, Iy = self._calculate_Ix_and_Iy(img)


    # endregion

 
    # region Private


    # region Shi Tomasi


    def _calculate_Ix_and_Iy(self, I : NDArray) -> Tuple[NDArray,  NDArray]:
        '''
        Returns Ix and Iy
        '''
            
        if self.derivation_operator == "sobel":
            operator_x = np.array(
                [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
                )
            
            operator_y = np.array(
                [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]])
            
        elif self.derivation_operator == "scharr":
            operator_x = np.array(
                [[-3, 0, 3],
                [-10, 0, 10],
                [-3, 0, 3]]
                )
            
            operator_y = np.array(
                [[-3, -10, -3],
                [0, 0, 0],
                [3, 10, 3]])
        
        elif  self.derivation_operator == "prewitt":
            operator_x = np.array(
                [[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]]
                )
            
            operator_y = np.array(
                [[-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]])
            
        elif  self.derivation_operator == "simple":
            operator_x = np.array(
                [[0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0]]
                )
            
            operator_y = np.array(
                [[0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]])
            
        
        
        Ix = cv2.filter2D(I, cv2.CV_32F, -operator_x, borderType=cv2.BORDER_REPLICATE)
        Iy = cv2.filter2D(I, cv2.CV_32F, -operator_y, borderType=cv2.BORDER_REPLICATE)


        return (Ix, Iy)


    def _calculate_Ixx_Ixy_and_Iyy(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        
        return (Ixx, Ixy, Iyy)


    def _calculate_structure_tensors(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        Ixx, Ixy, Iyy = self._calculate_Ixx_Ixy_and_Iyy(Ix, Iy)
        A = cv2.filter2D(Ixx, -1, self.structure_tensor_window, borderType=cv2.BORDER_REFLECT)
        B = cv2.filter2D(Ixy, -1, self.structure_tensor_window, borderType=cv2.BORDER_REFLECT)
        C = cv2.filter2D(Iyy, -1, self.structure_tensor_window, borderType=cv2.BORDER_REFLECT)

        
        return A, B, C


    def _calculate_response(self, Ix : NDArray, Iy : NDArray) -> NDArray:
        A, B, C = self._calculate_structure_tensors(Ix, Iy)
        if self.response_type == "sftt":
            response = 0.5 * (A + C) - 0.5 * (np.abs(A - C) + 2 * np.abs(B))
        else:
            ## From copilot ##
            M = np.zeros(A.shape + (2, 2))
            M[..., 0, 0] = A
            M[..., 0, 1] = B
            M[..., 1, 0] = B
            M[..., 1, 1] = C
            response = np.linalg.eigvalsh(M)[..., 0]
            #################
        return response


    def _threshold_response(self, response : NDArray) -> NDArray:
        if self.use_previous_max_when_calculating_threshold:
            threshold = self.quality_level * self.prev_max_value
        else:
            threshold = self.quality_level * np.max(response)

        thresholded = response.copy()
        thresholded[thresholded < threshold] = 0
        self.prev_max_value = np.max(response)
        return thresholded


    def _perform_non_maximum_supression(self, response : NDArray, size : int = 3) -> NDArray:
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
    
    def _drop_keypoints_on_border(self, keypoints: list[cv2.KeyPoint], img: NDArray) -> list[cv2.KeyPoint]:
        # Drop keypoints on the border where calculation of keypoint orientation or descriptor would be outside of the image.
        index_to_drop = []
        for index, keypoint in enumerate(keypoints):
            if ((keypoint.pt[0] < self.descriptor_window_buffer_size // 2) or
                (keypoint.pt[1] < self.descriptor_window_buffer_size // 2) or 
                (keypoint.pt[0] < self.orientation_calculation_window_size // 2) or
                (keypoint.pt[1] < self.orientation_calculation_window_size // 2) or
                (keypoint.pt[0] > ((img.shape[0]) - (self.descriptor_window_buffer_size // 2))) or
                (keypoint.pt[1] > ((img.shape[1]) -(self.descriptor_window_buffer_size // 2))) or 
                (keypoint.pt[0] > ((img.shape[0]) - (self.orientation_calculation_window_size // 2))) or
                (keypoint.pt[1] > ((img.shape[1]) -(self.orientation_calculation_window_size // 2)))):
                    index_to_drop.append(index)
        index_to_drop.reverse()
        for index in index_to_drop:
            keypoints.pop(index)
        
        return keypoints


    def _calculate_magnitude_and_angle(self, Ix : NDArray, Iy : NDArray) -> Tuple[NDArray, NDArray]:
        magnitude = np.sqrt(Ix*Ix + Iy*Iy)
        angle = (np.arctan2(Iy, Ix) + np.pi) * 360 / (2 * np.pi) # Change from [-pi, pi] -> [0, 360)
        angle[angle == 360] = 0

        return (magnitude, angle)


    def _calculate_histogram(
        self,
        array: NDArray,
        weights: NDArray,
        num_bins: int,
        start: float,
        stop: float,
        interpolated: bool = False,
        cyclic: bool = True
    ) -> Tuple[NDArray, NDArray]:

        # Flatten views
        x = np.ravel(array)
        w = np.ravel(weights)

        width = (stop - start) / num_bins
        bin_centers = start + (np.arange(num_bins) + 0.5) * width

        if cyclic:
            # Map to [start, stop)
            rng = (stop - start)
            x = ((x - start) % rng) + start
        else:
            # Drop out-of-range
            mask = (x >= start) & (x < stop)
            x = x[mask]
            w = w[mask]

        if not interpolated:
            hist, _ = np.histogram(x, bins=num_bins, range=(start, stop), weights=w)
            return bin_centers, hist

        # ---------- Edge-based interpolation (original behavior) ----------
        v = (x - start) / width                  # identical to original
        low  = np.floor(v).astype(np.int64)
        high = np.ceil(v).astype(np.int64)

        # original behavior: only HIGH wraps, LOW does NOT
        high = np.where(high >= num_bins, 0, high)

        # weights exactly as before
        low_weight  = 1.0 - np.abs(v - low)
        high_weight = 1.0 - low_weight

        # LOW contributions (only those in valid range)
        hist = np.zeros(num_bins, dtype=np.float64)

        mask_low = (low >= 0) & (low < num_bins)
        if np.any(mask_low):
            hist += np.bincount(
                low[mask_low],
                weights=w[mask_low] * low_weight[mask_low],
                minlength=num_bins
            )
        
        mask_high = (high >= 0) & (high < num_bins)
        if np.any(mask_high):
            hist += np.bincount(
                high[mask_high],
                weights=w[mask_high] * high_weight[mask_high],
                minlength=num_bins
            )

        return bin_centers, hist

    def _smooth_histogram_circular(self, hist, passes=6):
        kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
        for _ in range(passes):
            # Wrap-pad, convolve, unpad
            padded = np.concatenate([hist[-2:], hist, hist[:2]])
            hist = np.convolve(padded, kernel, mode='valid')
        return hist

    def _get_angles_from_histogram(self, values : NDArray, bins : NDArray) -> list[float]:
        '''
        Returns a list of angles, one for each keypoint
        '''

        bin_size = np.abs(bins[1] - bins[0])

        max_value = np.max(values)
        max_index = np.argmax(values)

        angles = []

        
        if self.create_new_keypoint_for_large_angle_histogram_values:
            bin_indexes_to_compute = np.where(values > (max_value * self.large_angle_histogram_value_threshold))[0]
        else:
            bin_indexes_to_compute = np.array([max_index])
            

        for index in bin_indexes_to_compute:
                # SIFT paper use parabola fitting, approximated by normal interpolation
                last_index = index + 1
                if last_index >= len(bins):
                    last_index = 0 # Wrap around

                prev_bin_value = bins[index] - bin_size
                bin_value = bins[index]
                next_bin_value = bins[index] + bin_size

                # angle = ((values[index-1] * prev_bin_value + values[index] * bin_value + values[last_index] * next_bin_value)/
                #          (values[index-1] + values[index] + values[last_index]))

                h_prev = values[index - 1]
                h_curr = values[index]
                h_next = values[last_index]
                denom = (h_prev - 2*h_curr + h_next)
                offset = 0.5 * (h_prev - h_next) / (denom + 1e-9) if abs(denom) > 1e-9 else 0.0
                angle = (bins[index] + offset * bin_size)
                
                angle %= 360
                
                angles.append(angle)
        
        return angles

    
    def _rotate_coordinates_around_center(self, angle: float) -> NDArray:
        '''
        Returns the new "rotated" pixel positions, based on the indexes, around the center of the matrix
        '''

        # Compute center
        cx = self.descriptor_window_buffer_size // 2
        cy = self.descriptor_window_buffer_size // 2

        # Shift to center
        x_shifted = self.buffer_xs - cx
        y_shifted = self.buffer_ys - cy

        cos_a = np.cos((angle / 360)*2*np.pi)
        sin_a = np.sin((angle / 360)*2*np.pi)

        # Rotation formula
        x_rot = x_shifted * cos_a - y_shifted * sin_a
        y_rot = x_shifted * sin_a + y_shifted * cos_a

        # Shift back
        x_rot += cx
        y_rot += cy

        # Stack into (H, W, 2)
        rotated_coordinates = np.dstack([x_rot, y_rot])

        return rotated_coordinates
    

    def _calculate_descriptor_subwindow_center_positions(self) -> NDArray:
        '''
        Returns an array with the center position of the window at specified index (Of rotation buffer window)
        '''
        num_subwindows_along_axis = self.descriptor_window_size // self.descriptor_subwindow_size

        offset_due_to_rotation_buffer = (self.descriptor_window_buffer_size - self.descriptor_window_size)//2

        distances = np.array([[(self.descriptor_subwindow_size/2 + i*self.descriptor_subwindow_size + offset_due_to_rotation_buffer,
                                self.descriptor_subwindow_size/2 + j*self.descriptor_subwindow_size + offset_due_to_rotation_buffer)
                                for i in range(num_subwindows_along_axis)]
                                for j in range(num_subwindows_along_axis)])

        return distances
    
    
    def _calculate_positional_weights_with_respect_to_subwindows(
        self,
        rotated_coordinates: NDArray,                 # (H, W, 2) with order (x, y) packed in the last dimension in your code
        subwindow_center_positions: NDArray           # (K, K, 2)
    ) -> NDArray:
        """
        Vectorized version:
        Returns weights with shape (K, K, H, W), identical to the original function,
        but computed via broadcasting (no Python loops).
        """
        H, W, _ = rotated_coordinates.shape
        K = subwindow_center_positions.shape[0]
        assert subwindow_center_positions.shape[1] == K, "Expected square (K, K, 2) centers"

        # Ensure float32 (saves memory and is plenty precise here)
        coords = rotated_coordinates.astype(np.float32, copy=False)  # (H, W, 2)
        centers = subwindow_center_positions.astype(np.float32, copy=False)  # (K, K, 2)

        # Flatten centers to (K*K, 2)
        centers_flat = centers.reshape(-1, 2)  # (M, 2) where M = K*K

        # Broadcast pixel coords against centers:
        #   coords[..., None, :] has shape (H, W, 1, 2)
        #   centers_flat[None, None, ...] has shape (1, 1, M, 2)
        # Result dists: (H, W, M, 2)
        d = np.abs(coords[..., None, :] - centers_flat[None, None, :, :])

        # Convert distances to per-axis weights and clip below 0
        inv_size = 1.0 / float(self.descriptor_subwindow_size)
        w = 1.0 - (d*self.d_weight) * inv_size                             # (H, W, M, 2)
        np.maximum(w, 0.0, out=w)                          # clip negatives to 0 in-place

        # Combine x and y contributions (axis=-1 is the (2,) axis)
        # NOTE: Your original code multiplies the two axes, regardless of naming.
        wxy = w[..., 0] * w[..., 1]                        # (H, W, M)

        # Reshape to (H, W, K, K) then transpose to (K, K, H, W) to match your original output
        weights = wxy.reshape(H, W, K, K).transpose(2, 3, 0, 1)  # (K, K, H, W)

        return weights
    
    def _create_new_keypoint(self, keypoint: cv2.KeyPoint, kp_angle) -> cv2.KeyPoint:

        new_keypoint = cv2.KeyPoint(keypoint.pt[0] * self.scale_pyramid_scaling_factor ** keypoint.octave,
                                    keypoint.pt[1] * self.scale_pyramid_scaling_factor ** keypoint.octave,
                                    keypoint.size,
                                    kp_angle,
                                    keypoint.response)
        
        return new_keypoint

    
    def _calculate_circular_weights(self) -> NDArray:
        """
        Returns a (16, N, N) weight matrix where N = descriptor_window_buffer_size.
        Each of the 16 slices is a smooth radial Gaussian centred at that ring's
        midpoint radius — like concentric ripples, overlapping smoothly rather
        than hard-cut rings. Max radius is N/2 (reaches the window edge).
        """
        N          = self.descriptor_window_buffer_size
        max_dist   = self.descriptor_window_size / 2.0
        n_rings    = self.n_rings
        ring_width = max_dist / n_rings

        # Pixel distances from the window centre
        centre = (N - 1) / 2.0
        gy, gx = np.mgrid[0:N, 0:N]
        dist = np.sqrt((gx - centre) ** 2 + (gy - centre) ** 2).astype(np.float32)  # (N, N)

        # Ring midpoint radii and sigma (controls ripple width/overlap)
        r_mid = (np.arange(n_rings) + 0.5) * ring_width   # (16,)
        sigma = ring_width * 0.6

        # Gaussian centred at each r_mid, normalised by circumference
        d_to_ring = dist[None, :, :] - r_mid[:, None, None]           # (16, N, N)
        gaussian  = np.exp(-0.5 * (d_to_ring / sigma) ** 2)           # (16, N, N)
        circ_norm = 1.0 / (2.0 * np.pi * r_mid)                       # (16,)
        weights   = gaussian * circ_norm[:, None, None]                # (16, N, N)

        return weights.astype(np.float32)
        
    # endregion

    # region Debug and tools
def weight_area_with_gaussian_window(area: NDArray, gaussian_std: float) -> NDArray:
        size = area.shape[0]
        gaussian_1d_window = gaussian(size, gaussian_std)
        gaussian_weight = np.outer(gaussian_1d_window, gaussian_1d_window)
        weighted_area = area * gaussian_weight
        return weighted_area


def extract_area(matrix: NDArray, center: Tuple[int, int], size : int, border_handling : str = "zero") -> NDArray:
    '''
    Returns a subarea in the matrix around the center (x, y) with window size (size, size). Elements in the window outside the
    border of the matrix is set to 0
    '''
    ## Calculate start and end indexes
    x_start, y_start = center[0] - size//2, center[1] - size//2
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

    subarea = np.zeros((size, size), dtype=matrix.dtype)

    if border_handling == "edge":
        # x low
        subarea[subarea_y_start : subarea_y_end, 0:subarea_x_start] = matrix[matrix_y_start : matrix_y_end,matrix_x_start:matrix_x_start+1]
        # y low
        subarea[0:subarea_y_start,subarea_x_start : subarea_x_end] = matrix[matrix_y_start:matrix_y_start+1,matrix_x_start : matrix_x_end]
        # x high
        subarea[subarea_y_start : subarea_y_end, subarea_x_end:size] = matrix[matrix_y_start : matrix_y_end,matrix_x_end-1:matrix_x_end]
        # y high
        subarea[subarea_y_end:size,subarea_x_start : subarea_x_end] = matrix[matrix_y_end-1:matrix_y_end, matrix_x_start : matrix_x_end]

        # corner x low y low
        subarea[0:subarea_y_start, 0:subarea_x_start] = matrix[matrix_y_start, matrix_x_start]
        # corner x high y low
        subarea[0:subarea_y_start, subarea_x_end:size] = matrix[matrix_y_start, matrix_x_end - 1]
        # corner x low y high
        subarea[subarea_y_end:size, 0:subarea_x_start] = matrix[matrix_y_end - 1, matrix_x_start]
        # corner x high y high
        subarea[subarea_y_end:size, subarea_x_end:size] = matrix[matrix_y_end - 1, matrix_x_end - 1]

    subarea[subarea_y_start : subarea_y_end, subarea_x_start : subarea_x_end] = matrix[matrix_y_start : matrix_y_end, matrix_x_start : matrix_x_end]

    return subarea

def downsample(img: NDArray, scale_factor: float, sigma: float) -> NDArray:
    if sigma != -1:
        img = cv2.GaussianBlur(img, (0,0), sigma, borderType=cv2.BORDER_REFLECT_101)
    downsampled_img = cv2.resize(img, None, fx= 1/scale_factor, fy= 1/scale_factor)


    return downsampled_img



def plot_image(ax : Axes,
               image: NDArray,
               center : Tuple[int, int] = None,
               size : int = None,
               plot_title: str = None,
               show_values : bool = False,
               value_color : str = "r",
               border_handling : str = "zero"
               ) -> None:
    '''
    Plots an image or a subregion of an image
    '''

    if ax is None:
        ax = plt.figure().gca()

    if center is None:
        area = image
    else:
        area = extract_area(image, center, size, border_handling)

    vmin = np.min(image)
    vmax = np.max(image)

    if plot_title is not None:
        ax.set_title(plot_title)

    ax.imshow(area, cmap = "gray", vmin=vmin, vmax=vmax)

    if show_values:
        for y_index in range(area.shape[0]):
            for x_index in range(area.shape[1]):

                # Make value displayed as scientific notation
                value_text = f"{area[y_index, x_index]:.2e}"
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
                                   plot_title: str = None,
                                   show_values: bool = False,
                                   value_color: str = "r",
                                   arrow_color: str = "blue",
                                   border_handling : str = "zero"
                                   ) -> None:
    '''
    Plots an image of with arrows as angle and respective magnitudes. Orientation goes from [-PI, PI] with respect to the x axis
    '''
    if ax is None:
        ax = plt.figure().gca()

    if center is None:
        magnitude_area = magnitude
        angle_area = angle
    else:
        magnitude_area = extract_area(magnitude, center, size, border_handling)
        angle_area = extract_area(angle, center, size, border_handling)
    
    plot_image(ax, image, center, size, plot_title, show_values = show_values, value_color=value_color, border_handling = border_handling)

    for y_index in range(magnitude_area.shape[0]):
        for x_index in range(magnitude_area.shape[1]):
            dx = np.cos((angle_area[y_index, x_index]/360) * 2 * np.pi) * magnitude_area[y_index, x_index] / np.max(magnitude_area)
            dy = np.sin((angle_area[y_index, x_index]/360) * 2 * np.pi) * magnitude_area[y_index, x_index] / np.max(magnitude_area)


            ax.annotate("", xytext=(x_index, y_index), xy=(x_index + dx, y_index + dy), arrowprops=dict(arrowstyle="->", color = arrow_color))


def plot_area_and_arrow_with_angle(ax : Axes,
                                   image: NDArray,
                                   angle : float,
                                   center: Tuple[int, int] = None,
                                   size: int = None,
                                   plot_title: str = None,
                                   show_values: bool = False,
                                   value_color: str = "r",
                                   arrow_color: str = "blue",
                                   border_handling : str = "zero"
                                   ) -> None:
    if ax is None:
        ax = plt.figure().gca()
    plot_image(ax, image, center, size, plot_title, show_values, value_color, border_handling)
    image_size = size
    if image_size is None:
        image_size = image.shape[0]
    
    arrow_size = image_size // 4

    x_start = y_start = (image_size//2)
    dx = np.cos((angle/360)*2*np.pi) * arrow_size
    dy = np.sin((angle/360)*2*np.pi) * arrow_size
    ax.annotate("", xytext=(x_start, y_start), xy=(x_start + dx, y_start + dy), arrowprops=dict(arrowstyle="->", color = arrow_color))


def plot_histogram(ax : Axes, hist : NDArray, bins: NDArray, plot_title: str = None):
    if ax is None:
        ax = plt.figure().gca()
    width = (bins[1] - bins[0])/2
    ax.bar(bins, hist, width)

    if plot_title is not None:
        ax.set_title(plot_title)


def plot_arrows_rotated_coordinates(ax : Axes,
                                   original: NDArray,
                                   rotated_coordinated: NDArray,
                                   plot_title: str = None
                                   ) -> None:
    
    if ax is None:
        ax = plt.figure().gca()


    plot_image(ax, original, plot_title = plot_title)

    max_coordinate = original.shape[0] - 1

    for y_index in range(original.shape[0]):
        for x_index in range(original.shape[1]):
            ax.annotate("", xytext=(x_index, y_index), xy=(rotated_coordinated[y_index, x_index, 0],rotated_coordinated[y_index, x_index, 1]), arrowprops=dict(arrowstyle="->", color = "r"))
    
    
def plot_subwindow_positions(ax : Axes,
                             image: NDArray,
                             subwindow_positions: NDArray,
                             plot_title: str = None,
                            ) -> None:
    
    if ax is None:
        ax = plt.figure().gca()
    plot_image(ax, image, plot_title = plot_title)

    subwindow_radius = (subwindow_positions[..., 0, 0] - subwindow_positions[..., 1, 0]) / 2

    for y_index in range(subwindow_positions.shape[0]):
        for x_index in range(subwindow_positions.shape[1]):
            subwindow_position = subwindow_positions[y_index, x_index]
            ax.text(subwindow_position[0], subwindow_position[1], f"({subwindow_position[0]}, {subwindow_position[1]})", color = "r", horizontalalignment = "center", verticalalignment = "center")
           
           ## From copilot
            cx, cy = subwindow_position  # center
            s = subwindow_radius

            # Four corners
            x0, x1 = cx - s, cx + s
            y0, y1 = cy - s, cy + s

            # Draw square
            ax.plot([x0, x1], [y0, y0], color="r")  # bottom
            ax.plot([x1, x1], [y0, y1], color="r")  # right
            ax.plot([x1, x0], [y1, y1], color="r")  # top
            ax.plot([x0, x0], [y1, y0], color="r")  # left


    # endregion

    # endregion