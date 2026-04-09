import numpy as np
import cv2
import random
import math
import json


def generate_noise(
    image_sequences,
    rot_range=(0, 0),
    scale_range=(1, 1),
    gauss_sigma_range=(0, 0),
    motion_length_range=(0, 0)
):
    noise_sequences = []

    sum_rotation = []
    sum_scale = []
    sum_gaussian_sigma = []
    sum_motion_len = []

    for seq_images in image_sequences:
        seq_noise = []

        for _ in seq_images[1:]:
            angle = random.uniform(*rot_range)
            scale = math.exp(random.uniform(math.log(scale_range[0]),math.log(scale_range[1])))
            sigma = random.uniform(*gauss_sigma_range)
            motion_len = int(random.uniform(*motion_length_range))
            motion_angle = random.uniform(0, 180)

            seq_noise.append({
                "angle": angle,
                "scale": scale,
                "gauss_sigma": sigma,
                "motion_len": motion_len,
                "motion_angle": motion_angle,
            })

            sum_rotation.append(abs(angle))
            sum_scale.append(abs(math.log(scale)))
            sum_gaussian_sigma.append(sigma)
            sum_motion_len.append(motion_len)

        noise_sequences.append(seq_noise)

    meta = {
        "scale_sampling": "log-uniform",
        "num_sequences": len(noise_sequences),
        "num_frames": sum(len(s) for s in noise_sequences),
        "mean_abs_rotation_deg": float(np.mean(sum_rotation)),
        "mean_log_scale_dev": float(np.mean(sum_scale)),
        "mean_gauss_sigma": float(np.mean(sum_gaussian_sigma)),
        "mean_motion_len": float(np.mean(sum_motion_len)),
    }
    print(meta)

    return {
        "meta": meta,
        "noise": noise_sequences
    }


def apply_noise(image_sequences, homography_sequences, noise):
    noise_sequences = noise["noise"]

    new_image_sequences = []
    new_homography_sequences = []

    for seq_images, seq_homs, seq_noise in zip(
        image_sequences, homography_sequences, noise_sequences
    ):
        ref_img = seq_images[0]
        h, w = ref_img.shape[:2]
        cx, cy = w / 2, h / 2

        new_images = [ref_img.copy()]
        new_homs = []

        for img, H, p in zip(seq_images[1:], seq_homs, seq_noise):
            S = random_scale_matrix(p["scale"], cx, cy)
            R = random_rotation_matrix(p["angle"], cx, cy)
            T = R @ S

            transformed = cv2.warpPerspective(img, T, (w, h))
            transformed = apply_gaussian_blur(transformed, p["gauss_sigma"])
            transformed = apply_motion_blur(
                transformed,
                p["motion_len"],
                p["motion_angle"]
            )

            new_images.append(transformed)
            new_homs.append(H @ np.linalg.inv(T))

        new_image_sequences.append(new_images)
        new_homography_sequences.append(new_homs)

    return new_image_sequences, new_homography_sequences


def save_noise(noise, path):
    with open(path, "w") as f:
        json.dump(noise, f, indent=2)


def load_noise(path):
    with open(path, "r") as f:
        return json.load(f)


def random_rotation_matrix(angle_deg, cx, cy):
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)

    return np.array([
        [ c, -s, cx - c * cx + s * cy],
        [ s,  c, cy - s * cx - c * cy],
        [ 0,  0, 1]
    ], dtype=np.float32)


def random_scale_matrix(scale, cx, cy):
    return np.array([
        [scale, 0, cx * (1 - scale)],
        [0, scale, cy * (1 - scale)],
        [0, 0, 1]
    ], dtype=np.float32)


def apply_gaussian_blur(image, sigma):
    if sigma <= 0:
        return image
    ksize = int(max(3, 2 * int(3 * sigma) + 1))
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_motion_blur(image, ksize, angle_deg):
    if ksize <= 1:
        return image

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= np.sum(kernel)

    return cv2.filter2D(image, -1, kernel)