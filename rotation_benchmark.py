import shi_tomasi_sift as sts
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from benchmark.utils import load_HPSequences
import csv
import opencvSIFTOrientation

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
LOAD_FROM_CSV = True
CSV_PATH = "orientation_benchmark_results_sides_banned_parabola_smoothing.csv"
ROTATION_BIN_SIZE = 1  # degrees
# ---------------------------------------------------------------

def rotate_highquality(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

if LOAD_FROM_CSV:
    average_angle_amounts = []
    amounts_difference_in_num_angles = []
    rotations = []
    closest_angle_errors = []
    closest_descriptor_errors = []
    intrinsic_descriptor_errors = []
    orientation_induced_errors = []
    pair_rotations = []

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    sections = []
    current = []
    for row in rows:
        if row == []:
            if current:
                sections.append(current)
                current = []
        else:
            current.append(row)
    if current:
        sections.append(current)

    for row in sections[0][1:]:
        average_angle_amounts.append(float(row[0]))
    for row in sections[1][1:]:
        amounts_difference_in_num_angles.append(float(row[0]))
    for row in sections[2][1:]:
        rotations.append(int(row[0]))
    for row in sections[3][1:]:
        closest_angle_errors.append(float(row[0]))
        closest_descriptor_errors.append(float(row[1]))
        intrinsic_descriptor_errors.append(float(row[2]))
        orientation_induced_errors.append(float(row[3]))
        pair_rotations.append(int(row[4]))

else:
    dataset_image_sequences, _ = load_HPSequences(r"hpatches-sequences-release")

    shi_tomasi_sift = sts.ShiTomasiSift(calculate_orientation_for_keypoints=True,
                                response_type="sftt",
                                orientation_calculation_gaussian_weight_std=50,
                                create_new_keypoint_for_large_angle_histogram_values=True,
                                num_octaves_in_scale_pyramid=1,
                                descriptor_gaussian_weight_std=50,
                                derivation_operator="simple",
                                use_circular_descriptor=False,
                                orientation_calculation_bin_count=36,
                                d_weight=0.7,
                                base_blur_sigma=-1,
                                max_corners=250,
                                enable_histogram_smoothing=True,
                                )

    average_angle_amounts = []
    amounts_difference_in_num_angles = []
    closest_angle_errors = []
    closest_descriptor_errors = []
    intrinsic_descriptor_errors = []
    orientation_induced_errors = []
    rotations = []
    pair_rotations = []

    for image_set in tqdm(dataset_image_sequences, leave=False):
        for img in image_set:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            our_kps = shi_tomasi_sift.detect(img)

            for keypoint in our_kps:
                non_rotated_buffered_area = sts.extract_area(img,
                                            (int(keypoint.pt[0]), int(keypoint.pt[1])),
                                            shi_tomasi_sift.descriptor_window_buffer_size,
                                            "edge")
                rotation = int(np.random.rand() * 360)
                abs_rotation = min(rotation, 360 - rotation)
                rotated_buffered_area = rotate_highquality(non_rotated_buffered_area, rotation)

                x_pos, y_pos = shi_tomasi_sift.descriptor_window_buffer_size//2, shi_tomasi_sift.descriptor_window_buffer_size//2

                non_rotated_buffered_dx, non_rotated_buffered_dy = shi_tomasi_sift._calculate_Ix_and_Iy(non_rotated_buffered_area)
                non_rotated_dx = sts.extract_area(non_rotated_buffered_dx, (x_pos, y_pos), shi_tomasi_sift.orientation_calculation_window_size)
                non_rotated_dy = sts.extract_area(non_rotated_buffered_dy, (x_pos, y_pos), shi_tomasi_sift.orientation_calculation_window_size)
                non_rotated_magnitudes, non_rotated_pixel_angles = shi_tomasi_sift._calculate_magnitude_and_angle(non_rotated_dx, non_rotated_dy)
                # non_rotated_orientation_angles = opencvSIFTOrientation._calculate_orientation_of_keypoint((x_pos, y_pos), non_rotated_magnitudes, non_rotated_pixel_angles, shi_tomasi_sift.orientation_calculation_window_size, shi_tomasi_sift.orientation_calculation_gaussian_weight_std, shi_tomasi_sift.orientation_calculation_bin_count)
                non_rotated_orientation_angles = shi_tomasi_sift._calculate_orientation_of_keypoint((x_pos, y_pos), non_rotated_magnitudes, non_rotated_pixel_angles)

                rotated_buffered_dx, rotated_buffered_dy = shi_tomasi_sift._calculate_Ix_and_Iy(rotated_buffered_area)
                rotated_dx = sts.extract_area(rotated_buffered_dx, (x_pos, y_pos), shi_tomasi_sift.orientation_calculation_window_size)
                rotated_dy = sts.extract_area(rotated_buffered_dy, (x_pos, y_pos), shi_tomasi_sift.orientation_calculation_window_size)
                rotated_magnitudes, rotated_pixel_angles = shi_tomasi_sift._calculate_magnitude_and_angle(rotated_dx, rotated_dy)
                # rotated_orientation_angles = opencvSIFTOrientation._calculate_orientation_of_keypoint((x_pos, y_pos), rotated_magnitudes, rotated_pixel_angles, shi_tomasi_sift.orientation_calculation_window_size, shi_tomasi_sift.orientation_calculation_gaussian_weight_std, shi_tomasi_sift.orientation_calculation_bin_count)
                rotated_orientation_angles = shi_tomasi_sift._calculate_orientation_of_keypoint((x_pos, y_pos), rotated_magnitudes, rotated_pixel_angles)

                corrected_rotated_orientation_angles = [a + rotation for a in rotated_orientation_angles]

                average_angle_amounts.append((len(non_rotated_orientation_angles) + len(corrected_rotated_orientation_angles)) / 2)
                amounts_difference_in_num_angles.append(abs(len(non_rotated_orientation_angles) - len(corrected_rotated_orientation_angles)))
                rotations.append(abs_rotation)

                if len(non_rotated_orientation_angles) > 0 and len(corrected_rotated_orientation_angles) > 0:
                    pairs = []
                    for nr_idx, nr_angle in enumerate(non_rotated_orientation_angles):
                        for r_idx, corr_angle in enumerate(corrected_rotated_orientation_angles):
                            x = (corr_angle - nr_angle) % 360
                            diff = min(x, 360 - x)
                            pairs.append((diff, nr_idx, r_idx))

                    pairs.sort(key=lambda p: p[0])

                    used_nr = set()
                    used_r = set()
                    for diff, nr_idx, r_idx in pairs:
                        if nr_idx in used_nr or r_idx in used_r:
                            continue
                        used_nr.add(nr_idx)
                        used_r.add(r_idx)

                        closest_angle_errors.append(diff)
                        pair_rotations.append(abs_rotation)

                        _, normal_descriptor = shi_tomasi_sift._calculate_descriptor(
                            (x_pos, y_pos),
                            non_rotated_orientation_angles[nr_idx],
                            non_rotated_magnitudes,
                            non_rotated_pixel_angles
                        )
                        _, rotated_descriptor = shi_tomasi_sift._calculate_descriptor(
                            (x_pos, y_pos),
                            rotated_orientation_angles[r_idx],
                            rotated_magnitudes,
                            rotated_pixel_angles
                        )

                        gt_rotated_angle = (non_rotated_orientation_angles[nr_idx] - rotation) % 360
                        _, rotated_descriptor_gt = shi_tomasi_sift._calculate_descriptor(
                            (x_pos, y_pos),
                            gt_rotated_angle,
                            rotated_magnitudes,
                            rotated_pixel_angles
                        )

                        closest_descriptor_errors.append(np.linalg.norm(normal_descriptor - rotated_descriptor))
                        intrinsic_descriptor_errors.append(np.linalg.norm(normal_descriptor - rotated_descriptor_gt))
                        orientation_induced_errors.append(np.linalg.norm(rotated_descriptor - rotated_descriptor_gt))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["average_angle_amount"])
        writer.writerows([[v] for v in average_angle_amounts])

        writer.writerow([])
        writer.writerow(["amount_difference_in_num_angles"])
        writer.writerows([[v] for v in amounts_difference_in_num_angles])

        writer.writerow([])
        writer.writerow(["abs_rotation"])
        writer.writerows([[v] for v in rotations])

        writer.writerow([])
        writer.writerow(["closest_angle_error", "closest_descriptor_error", "intrinsic_descriptor_error", "orientation_induced_error", "abs_rotation"])
        writer.writerows(zip(closest_angle_errors, closest_descriptor_errors, intrinsic_descriptor_errors, orientation_induced_errors, pair_rotations))


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
closest_angle_errors = np.array(closest_angle_errors)
closest_descriptor_errors = np.array(closest_descriptor_errors)
intrinsic_descriptor_errors = np.array(intrinsic_descriptor_errors)
orientation_induced_errors = np.array(orientation_induced_errors)
pair_rotations = np.array(pair_rotations)

bin_edges = np.arange(0, 181, ROTATION_BIN_SIZE)
bin_centers = bin_edges[:-1] + ROTATION_BIN_SIZE / 2
bin_indices = np.clip(np.digitize(pair_rotations, bin_edges) - 1, 0, len(bin_edges) - 2)

def per_bin_stats(values, bin_indices, n_bins):
    means, stds, medians, p25, p75 = [], [], [], [], []
    for k in range(n_bins):
        v = values[bin_indices == k]
        if len(v) > 0:
            means.append(np.mean(v))
            stds.append(np.std(v))
            medians.append(np.median(v))
            p25.append(np.percentile(v, 25))
            p75.append(np.percentile(v, 75))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            medians.append(np.nan)
            p25.append(np.nan)
            p75.append(np.nan)
    return (np.array(means), np.array(stds),
            np.array(medians), np.array(p25), np.array(p75))

n_bins = len(bin_edges) - 1

# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

plt.figure()
values, counts = np.unique(average_angle_amounts, return_counts=True)
plt.bar(values, counts)
plt.xlabel("Average number of angles")
plt.ylabel("Count")
plt.title("Average angle amounts")

plt.figure()
values, counts = np.unique(amounts_difference_in_num_angles, return_counts=True)
plt.bar(values, counts)
plt.xlabel("Difference in number of angles")
plt.ylabel("Count")
plt.title("Difference in number of angles between rotated and non-rotated")

plt.figure()
plt.hist(closest_angle_errors, bins=list(range(181)))
plt.xlabel("Angle error (degrees)")
plt.ylabel("Count")
plt.title("Closest angle errors")

plt.figure()
plt.hist(closest_descriptor_errors, bins=[i/100 for i in range(0, 200)])
plt.xlabel("Descriptor error (L2)")
plt.ylabel("Count")
plt.title("Closest descriptor errors")

plt.figure()
sorted_errors = np.sort(closest_angle_errors)
plt.plot(sorted_errors, np.arange(1, len(sorted_errors)+1) / len(sorted_errors))
plt.xlabel("Angle error (degrees)")
plt.ylabel("Fraction of matches")
plt.axvline(10, color='r', linestyle='--', label='10°')
plt.axvline(20, color='orange', linestyle='--', label='20°')
plt.legend()
plt.title("CDF of closest angle errors")

# Core decomposition plot
plt.figure()
for arr, label in [
    (closest_descriptor_errors,   "Total error"),
    (intrinsic_descriptor_errors, "Intrinsic (perfect orientation)"),
    (orientation_induced_errors,  "Orientation-induced"),
]:
    s = np.sort(arr)
    plt.plot(s, np.arange(1, len(s)+1) / len(s), label=label)
plt.xlabel("L2 descriptor error")
plt.ylabel("Fraction of matches")
plt.legend()
plt.title("CDF: decomposition of descriptor error")

plt.figure()
plt.hexbin(closest_angle_errors, closest_descriptor_errors, gridsize=40, cmap='viridis')
plt.colorbar(label='count')
r = np.corrcoef(closest_angle_errors, closest_descriptor_errors)[0, 1]
plt.xlabel("Angle error (degrees)")
plt.ylabel("Descriptor error (L2)")
plt.title(f"Angle error vs total descriptor error (r = {r:.3f})")

plt.figure()
plt.hexbin(closest_angle_errors, orientation_induced_errors, gridsize=40, cmap='viridis')
plt.colorbar(label='count')
r = np.corrcoef(closest_angle_errors, orientation_induced_errors)[0, 1]
plt.xlabel("Angle error (degrees)")
plt.ylabel("Orientation-induced descriptor error (L2)")
plt.title(f"Angle error vs orientation-induced descriptor error (r = {r:.3f})")

plt.figure()
plt.hist(rotations, bins=np.arange(0, 181, ROTATION_BIN_SIZE))
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Count")
plt.title(f"Distribution of absolute rotations ({ROTATION_BIN_SIZE}° bins)")

plt.figure()
plt.hexbin(pair_rotations, closest_angle_errors, gridsize=int(180/ROTATION_BIN_SIZE), cmap='viridis')
plt.colorbar(label='count')
r = np.corrcoef(pair_rotations, closest_angle_errors)[0, 1]
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Angle error (degrees)")
plt.title(f"Angle error vs absolute rotation (r = {r:.3f})")

plt.figure()
plt.hexbin(pair_rotations, intrinsic_descriptor_errors, gridsize=int(180/ROTATION_BIN_SIZE), cmap='viridis')
plt.colorbar(label='count')
r = np.corrcoef(pair_rotations, intrinsic_descriptor_errors)[0, 1]
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Intrinsic descriptor error (L2)")
plt.title(f"Intrinsic descriptor error vs absolute rotation (r = {r:.3f})")

means, stds, medians, p25, p75 = per_bin_stats(closest_angle_errors, bin_indices, n_bins)

plt.figure()
plt.bar(bin_centers, means, width=ROTATION_BIN_SIZE * 0.8, label='Mean')
plt.errorbar(bin_centers, means, yerr=stds, fmt='none', color='black', capsize=2, label='±1 std')
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Angle error (degrees)")
plt.legend()
plt.title(f"Mean ± std angle error per {ROTATION_BIN_SIZE}° rotation bin")

plt.figure()
plt.bar(bin_centers, medians, width=ROTATION_BIN_SIZE * 0.8, label='Median')
plt.errorbar(bin_centers, medians, yerr=[medians - p25, p75 - medians],
             fmt='none', color='black', capsize=2, label='IQR')
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Angle error (degrees)")
plt.legend()
plt.title(f"Median + IQR angle error per {ROTATION_BIN_SIZE}° rotation bin")

means_d, stds_d, medians_d, p25_d, p75_d = per_bin_stats(closest_descriptor_errors, bin_indices, n_bins)
means_i, stds_i, medians_i, p25_i, p75_i = per_bin_stats(intrinsic_descriptor_errors, bin_indices, n_bins)
means_o, stds_o, medians_o, p25_o, p75_o = per_bin_stats(orientation_induced_errors, bin_indices, n_bins)

plt.figure()
plt.bar(bin_centers, means_d, width=ROTATION_BIN_SIZE * 0.8, label='Mean')
plt.errorbar(bin_centers, means_d, yerr=stds_d, fmt='none', color='black', capsize=2, label='±1 std')
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Descriptor error (L2)")
plt.legend()
plt.title(f"Mean ± std total descriptor error per {ROTATION_BIN_SIZE}° rotation bin")

plt.figure()
plt.bar(bin_centers, medians_d, width=ROTATION_BIN_SIZE * 0.8, label='Median')
plt.errorbar(bin_centers, medians_d, yerr=[medians_d - p25_d, p75_d - medians_d],
             fmt='none', color='black', capsize=2, label='IQR')
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Descriptor error (L2)")
plt.legend()
plt.title(f"Median + IQR total descriptor error per {ROTATION_BIN_SIZE}° rotation bin")

# Stacked bar: intrinsic vs orientation-induced per rotation bin
plt.figure()
plt.bar(bin_centers, means_i, width=ROTATION_BIN_SIZE * 0.8, label='Intrinsic', alpha=0.8)
plt.bar(bin_centers, means_o, width=ROTATION_BIN_SIZE * 0.8, bottom=means_i, label='Orientation-induced', alpha=0.8)
plt.xlabel("Absolute rotation (degrees)")
plt.ylabel("Mean descriptor error (L2)")
plt.legend()
plt.title(f"Stacked mean descriptor error by source per {ROTATION_BIN_SIZE}° bin")

ranges = [(0, 22), (22, 45), (45, 67), (67, 90), (90, 112), (112, 135), (135, 157), (157, 180)]

plt.figure()
for lo, hi in ranges:
    mask = (pair_rotations >= lo) & (pair_rotations < hi)
    if np.sum(mask) > 0:
        s = np.sort(closest_angle_errors[mask])
        plt.plot(s, np.arange(1, len(s)+1) / len(s), label=f"{lo}–{hi}°")
plt.xlabel("Angle error (degrees)")
plt.ylabel("Fraction of matches")
plt.legend(fontsize='small')
plt.title("CDF of angle error by absolute rotation range")

plt.figure()
for lo, hi in ranges:
    mask = (pair_rotations >= lo) & (pair_rotations < hi)
    if np.sum(mask) > 0:
        s = np.sort(closest_descriptor_errors[mask])
        plt.plot(s, np.arange(1, len(s)+1) / len(s), label=f"{lo}–{hi}°")
plt.xlabel("Descriptor error (L2)")
plt.ylabel("Fraction of matches")
plt.legend(fontsize='small')
plt.title("CDF of total descriptor error by absolute rotation range")

plt.figure()
for lo, hi in ranges:
    mask = (pair_rotations >= lo) & (pair_rotations < hi)
    if np.sum(mask) > 0:
        s = np.sort(intrinsic_descriptor_errors[mask])
        plt.plot(s, np.arange(1, len(s)+1) / len(s), label=f"{lo}–{hi}°")
plt.xlabel("Descriptor error (L2)")
plt.ylabel("Fraction of matches")
plt.legend(fontsize='small')
plt.title("CDF of intrinsic descriptor error by absolute rotation range")

plt.figure()
for lo, hi in ranges:
    mask = (pair_rotations >= lo) & (pair_rotations < hi)
    if np.sum(mask) > 0:
        s = np.sort(orientation_induced_errors[mask])
        plt.plot(s, np.arange(1, len(s)+1) / len(s), label=f"{lo}–{hi}°")
plt.xlabel("Descriptor error (L2)")
plt.ylabel("Fraction of matches")
plt.legend(fontsize='small')
plt.title("CDF of orientation-induced descriptor error by absolute rotation range")

plt.show()