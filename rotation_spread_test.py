import shi_tomasi_sift as sts
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from benchmark.utils import load_HPSequences

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
N_ROTATIONS = 1000  # per keypoint
# ---------------------------------------------------------------

dataset_image_sequences, _ = load_HPSequences(r"hpatches-sequences-release")

shi_tomasi_sift = sts.ShiTomasiSift(calculate_orientation_for_keypoints=True,
                            response_type="sftt",
                            orientation_calculation_gaussian_weight_std=8,
                            num_octaves_in_scale_pyramid=1,
                            descriptor_gaussian_weight_std=50,
                            derivation_operator="simple",
                            use_circular_descriptor=False,
                            d_weight=0.7,
                            base_blur_sigma=-1,
                            max_corners=250
                            )

# Per-keypoint summary stats (one entry per keypoint)
kp_mean_pairwise = []
kp_std_pairwise = []
kp_mean_to_mean = []
kp_std_to_mean = []

# All individual distances pooled across all keypoints
all_distances_to_mean = []
all_abs_rotations = []

x_pos = shi_tomasi_sift.descriptor_window_buffer_size // 2
y_pos = shi_tomasi_sift.descriptor_window_buffer_size // 2

for image_set in tqdm(dataset_image_sequences, leave=False):
    for img in image_set:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints = shi_tomasi_sift.detect(img)
        if len(keypoints) == 0:
            continue

        # Use only the strongest keypoint per image
        keypoint = keypoints[0]

        buffered_area = sts.extract_area(img,
                                         (int(keypoint.pt[0]), int(keypoint.pt[1])),
                                         shi_tomasi_sift.descriptor_window_buffer_size,
                                         "edge")

        descriptors = []
        abs_rotations = []

        for _ in range(N_ROTATIONS):
            rotation = int(np.random.rand() * 360)
            abs_rotation = min(rotation, 360 - rotation)
            rotated_area = imutils.rotate(buffered_area, rotation)

            dx, dy = shi_tomasi_sift._calculate_Ix_and_Iy(rotated_area)
            sub_dx = sts.extract_area(dx, (x_pos, y_pos), shi_tomasi_sift.orientation_calculation_window_size)
            sub_dy = sts.extract_area(dy, (x_pos, y_pos), shi_tomasi_sift.orientation_calculation_window_size)

            magnitudes, pixel_angles = shi_tomasi_sift._calculate_magnitude_and_angle(sub_dx, sub_dy)
            orientation_angles = shi_tomasi_sift._calculate_orientation_of_keypoint((x_pos, y_pos), magnitudes, pixel_angles)

            if len(orientation_angles) == 0:
                continue

            dominant_angle = orientation_angles[0]

            buffered_dx, buffered_dy = shi_tomasi_sift._calculate_Ix_and_Iy(rotated_area)
            buffered_magnitudes, buffered_pixel_angles = shi_tomasi_sift._calculate_magnitude_and_angle(buffered_dx, buffered_dy)

            _, descriptor = shi_tomasi_sift._calculate_descriptor(
                (x_pos, y_pos),
                dominant_angle,
                buffered_magnitudes,
                buffered_pixel_angles
            )

            descriptors.append(descriptor)
            abs_rotations.append(abs_rotation)

        if len(descriptors) < 2:
            continue

        descriptors = np.array(descriptors)
        abs_rotations = np.array(abs_rotations)

        # Pairwise distances
        n = len(descriptors)
        pairwise = np.linalg.norm(descriptors[:, None] - descriptors[None, :], axis=-1)
        upper = pairwise[np.triu_indices(n, k=1)]

        # Distance to mean
        mean_descriptor = np.mean(descriptors, axis=0)
        distances_to_mean = np.linalg.norm(descriptors - mean_descriptor, axis=1)

        kp_mean_pairwise.append(np.mean(upper))
        kp_std_pairwise.append(np.std(upper))
        kp_mean_to_mean.append(np.mean(distances_to_mean))
        kp_std_to_mean.append(np.std(distances_to_mean))

        all_distances_to_mean.extend(distances_to_mean.tolist())
        all_abs_rotations.extend(abs_rotations.tolist())

kp_mean_pairwise = np.array(kp_mean_pairwise)
kp_std_pairwise = np.array(kp_std_pairwise)
kp_mean_to_mean = np.array(kp_mean_to_mean)
kp_std_to_mean = np.array(kp_std_to_mean)
all_distances_to_mean = np.array(all_distances_to_mean)
all_abs_rotations = np.array(all_abs_rotations)

print(f"Keypoints processed: {len(kp_mean_pairwise)}")
print(f"\nPairwise distance (per-keypoint mean) — mean: {np.mean(kp_mean_pairwise):.4f}, std: {np.std(kp_mean_pairwise):.4f}")
print(f"Distance to mean   (per-keypoint mean) — mean: {np.mean(kp_mean_to_mean):.4f}, std: {np.std(kp_mean_to_mean):.4f}")

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

plt.figure()
plt.hist(kp_mean_pairwise, bins=50)
plt.xlabel("Mean pairwise L2 distance")
plt.ylabel("Count")
plt.title("Per-keypoint mean pairwise descriptor distance")

plt.figure()
plt.hist(kp_mean_to_mean, bins=50)
plt.xlabel("Mean L2 distance to mean descriptor")
plt.ylabel("Count")
plt.title("Per-keypoint mean distance to mean descriptor")

plt.figure()
plt.hist(all_distances_to_mean, bins=100)
plt.xlabel("L2 distance to mean descriptor")
plt.ylabel("Count")
plt.title("All distances to mean descriptor (pooled)")

plt.figure()
sorted_d = np.sort(all_distances_to_mean)
plt.plot(sorted_d, np.arange(1, len(sorted_d)+1) / len(sorted_d))
plt.xlabel("L2 distance to mean descriptor")
plt.ylabel("Fraction")
plt.title("CDF of distance to mean descriptor (pooled)")

plt.figure()
plt.scatter(all_abs_rotations, all_distances_to_mean, alpha=0.05, s=5)
r = np.corrcoef(all_abs_rotations, all_distances_to_mean)[0, 1]
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("L2 distance to mean descriptor")
plt.title(f"Descriptor drift vs absolute rotation (r = {r:.3f})")

# Mean drift per rotation bin
bin_edges = np.arange(0, 181, 5)
bin_centers = bin_edges[:-1] + 2.5
bin_indices = np.clip(np.digitize(all_abs_rotations, bin_edges) - 1, 0, len(bin_edges) - 2)
bin_means = [np.mean(all_distances_to_mean[bin_indices == k]) if np.any(bin_indices == k) else np.nan
             for k in range(len(bin_edges) - 1)]
bin_stds  = [np.std(all_distances_to_mean[bin_indices == k])  if np.any(bin_indices == k) else np.nan
             for k in range(len(bin_edges) - 1)]
bin_means = np.array(bin_means)
bin_stds  = np.array(bin_stds)

plt.figure()
plt.bar(bin_centers, bin_means, width=4, label='Mean')
plt.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none', color='black', capsize=2, label='±1 std')
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("L2 distance to mean descriptor")
plt.legend()
plt.title("Mean ± std descriptor drift per 5° rotation bin")

plt.show()