import numpy as np
from numpy.typing import NDArray
import cv2

# Constants from OpenCV sift.simd.hpp (branch 4.x)
SIFT_ORI_HIST_BINS  = 36
SIFT_ORI_SIG_FCTR   = 1.5
SIFT_ORI_RADIUS     = 4.5   # = 3 * SIFT_ORI_SIG_FCTR
SIFT_ORI_PEAK_RATIO = 0.8

def _calculate_orientation_of_keypoint(keypoint_or_center: cv2.KeyPoint | tuple,
                                        magnitude: NDArray, pixel_angles: NDArray, orientation_calculation_window_size,
                                        orientation_calculation_gaussian_weight_std,
                                        orientation_calculation_bin_count) -> list:
    """
    Reimplementation of OpenCV calcOrientationHist().
    Source: opencv/modules/features2d/src/sift.simd.hpp (branch 4.x)
    Uses fixed orientation_calculation_window_size instead of scale-adaptive radius.
    """
    if type(keypoint_or_center) == cv2.KeyPoint:
        x, y = keypoint_or_center.pt[0], keypoint_or_center.pt[1]
    else:
        x, y = keypoint_or_center[0], keypoint_or_center[1]

    cx, cy = int(x), int(y)
    radius = orientation_calculation_window_size // 2
    sigma  = orientation_calculation_gaussian_weight_std
    n      = orientation_calculation_bin_count

    expf_scale = -1.0 / (2.0 * sigma * sigma)

    h, w = magnitude.shape
    hist = np.zeros(n, dtype=np.float64)

    for i in range(-radius, radius + 1):
        py = cy + i
        if py < 0 or py >= h:
            continue
        for j in range(-radius, radius + 1):
            px = cx + j
            if px < 0 or px >= w:
                continue

            mag    = magnitude[py, px]
            angle  = pixel_angles[py, px] % 360.0
            weight = np.exp((i*i + j*j) * expf_scale)

            bin_idx = int(round(n / 360.0 * angle)) % n
            hist[bin_idx] += weight * mag

    # Single-pass smoothing — matches OpenCV calcOrientationHist()
    # kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    # padded = np.concatenate([hist[-2:], hist, hist[:2]])
    # hist   = np.convolve(padded, kernel, mode='valid')

    max_val = np.max(hist)
    if max_val == 0:
        return []

    threshold = 0.8 * max_val
    angles = []

    for j in range(n):
        l = (j - 1) % n
        r = (j + 1) % n
        if hist[j] > hist[l] and hist[j] > hist[r] and hist[j] >= threshold:
            denom  = hist[l] - 2.0 * hist[j] + hist[r]
            offset = 0.5 * (hist[l] - hist[r]) / (denom + 1e-9) if abs(denom) > 1e-9 else 0.0
            refined_bin = j + offset + 0.5
            angles.append((refined_bin * 360.0 / n) % 360.0)

    return angles