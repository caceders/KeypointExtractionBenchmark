import numpy as np
import cv2
import random

def random_rotation_matrix(angle_deg, cx, cy):
    """Return 3x3 homography for rotation around image center."""
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Translate center to origin -> rotate -> translate back
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,   1 ]], dtype=np.float32)

    R = np.array([[cos_a, -sin_a, 0],
                  [sin_a,  cos_a, 0],
                  [0,      0,     1]], dtype=np.float32)

    T2 = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0,  1]], dtype=np.float32)

    return T2 @ R @ T1


def random_scale_matrix(scale, cx, cy):
    """Return 3x3 homography for scaling around center."""
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0,  1 ]], dtype=np.float32)

    S = np.array([[scale, 0,     0],
                  [0,     scale, 0],
                  [0,     0,     1]], dtype=np.float32)

    T2 = np.array([[1, 0, cx],
                   [0, 1, cy],
                   [0, 0,  1]], dtype=np.float32)

    return T2 @ S @ T1


def apply_gaussian_blur(image, sigma):
    if sigma <= 0:
        return image
    ksize = int(max(3, 2 * int(3*sigma) + 1))
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_motion_blur(image, ksize, angle_deg):
    if ksize <= 1:
        return image

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0

    # Rotate kernel to chosen angle
    M = cv2.getRotationMatrix2D((ksize/2, ksize/2), angle_deg, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= np.sum(kernel)

    return cv2.filter2D(image, -1, kernel)


def apply_image_noise(image_sequences, homography_sequences,
                        rot_range=(0, 0),
                        #rot_range=(-20, 20),
                        scale_range=(1, 1),
                        # scale_range=(0.8, 1.2),
                        gauss_sigma_range=(0, 0),
                        # gauss_sigma_range=(0, 1.2),
                        motion_length_range=(0, 0)):
                        # motion_length_range=(0, 15)):
    """
    Applies random scale, rotation, gaussian blur, and motion blur
    to all related images in each sequence, updating homographies accordingly.
    """

    new_image_sequences = []
    new_homography_sequences = []

    for seq_images, seq_homs in zip(image_sequences, homography_sequences):
        ref_img = seq_images[0]
        h_ref, w_ref = ref_img.shape[:2]

        new_images = [ref_img.copy()]
        new_homs = []

        for img, H in zip(seq_images[1:], seq_homs):

            h, w = img.shape[:2]
            cx, cy = w / 2, h / 2

            # --- Random parameters ---
            angle = random.uniform(*rot_range)
            scale = random.uniform(*scale_range)
            sigma = random.uniform(*gauss_sigma_range)
            motion_len = int(random.uniform(*motion_length_range))
            motion_angle = random.uniform(0, 180)

            # --- Construct T = rotation · scale ---
            S = random_scale_matrix(scale, cx, cy)
            R = random_rotation_matrix(angle, cx, cy)
            T = R @ S

            # --- Apply geometric transform ---
            transformed = cv2.warpPerspective(img, T, (w, h))

            # --- Apply blur ---
            transformed = apply_gaussian_blur(transformed, sigma)
            transformed = apply_motion_blur(transformed, motion_len, motion_angle)

            new_images.append(transformed)

            # --- Update homography ---
            # H_old is in the convention:  related → reference
            # T is applied to the related image
            # New mapping: related_transformed → reference
            H_new = H @ np.linalg.inv(T)
            new_homs.append(H_new)

        new_image_sequences.append(new_images)
        new_homography_sequences.append(new_homs)
    image_sequences = new_image_sequences
    homography_sequences = new_homography_sequences