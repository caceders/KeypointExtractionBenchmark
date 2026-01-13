from tqdm import tqdm
from benchmark.utils import calculate_overlap_one_circle_to_many
from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.matching import MatchSet, greedy_maximum_bipartite_matching, Match
from typing import Tuple, Callable
import random
import numpy as np
import warnings
from beartype import beartype
from config import *
import cv2

# Beartype commented out for performance
#@beartype
def speed_test(feature_extractor: FeatureExtractor, dataset_image_sequences: list[list[np.ndarray]]):
     
    time_per_image = []
    for _, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Calculating speed")):
        for _, image in enumerate(image_sequence):
            time = feature_extractor.get_extraction_time_on_image(image)
            time_per_image.append(time)

    time = np.average(time_per_image)
    speed = 1/time
    return speed


#@beartype
def find_all_features_for_dataset(feature_extractor: FeatureExtractor, dataset_image_sequences: list[list[np.ndarray]], image_feature_set: ImageFeatureSet, max_features: int):  

    for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Finding all features")):
        for image_index, image in enumerate(image_sequence):

            keypoints = feature_extractor.detect_keypoints(image)
            if (len(keypoints) == 0):
                break

            keypoints, descriptions = feature_extractor.describe_keypoints(image, keypoints)

            # For debug ################################
            
            #out_image = cv2.drawKeypoints(image, [max(list(keypoints), key=lambda kp: kp.response)], None, color=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            out_image = cv2.drawKeypoints(image, keypoints, None, color=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Keypoints", out_image)
            cv2.waitKey(0)   # 200 ms = 0.2 s
            cv2.destroyAllWindows()
            ############################################
            
            # prebinding locals for performance increase
            Feature_ = Feature
            seq = sequence_index
            img = image_index
            zip_ = zip

            features = [Feature_(kp, desc, seq, img) for kp, desc in zip_(keypoints, descriptions)]
            
            if max_features < len(features):
                # Pick the top max_features elements
                scores = np.array([f.keypoint.response for f in features])
                idx = np.argpartition(scores, -max_features)[-max_features:]
                features = [features[i] for i in idx]
            
            image_feature_set[sequence_index][image_index] = features


#@beartype
def calculate_valid_matches(image_feature_set: ImageFeatureSet, dataset_homography_sequence: list[list[np.ndarray]], FEATURE_OVERLAP_THRESHOLD: float):
    
    set_numbers_of_possible_correct_matches= []
    set_repeatabilities = []
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):

        numbers_of_possible_correct_matches = []
        repeatabilities = []

        reference_features = image_feature_sequence.reference_image_features
        
        for related_image_index, related_images_features in enumerate(image_feature_sequence.related_images_features):

            homography = dataset_homography_sequence[sequence_index][related_image_index]

            if len(related_images_features) == 0:
                continue

            # transform position 
            related_features_position_transformed = np.array([feature.get_pt_after_homography_transform(homography) for feature in related_images_features])

            # transform sizes
            related_features_size_transformed = np.array([feature.get_size_after_homography_transform(homography) for feature in related_images_features])
            overlap_matrix = []
            for reference_feature in reference_features:
                
                # Check distances
                distances = np.linalg.norm(related_features_position_transformed - reference_feature.pt, axis=1)
                
                overlaps = calculate_overlap_one_circle_to_many(reference_feature.keypoint.size, related_features_size_transformed, distances)
                overlap_matrix.append(overlaps)

                # Final mask: ONLY overlap criterion
                mask = (overlaps >= FEATURE_OVERLAP_THRESHOLD)

                valid_feature_indexes = np.nonzero(mask)[0]
                if valid_feature_indexes.size == 0:
                    continue

                # Store valid features for that check mask
                for index in valid_feature_indexes:
                    related_feature = related_images_features[index]
                    distance = distances[index]
                    reference_feature.store_valid_match_for_image(related_image_index, related_feature)
                    related_feature.store_valid_match_for_image(0, reference_feature)

            # Run matching
            overlap_matrix_np = np.array(overlap_matrix)

            matches = greedy_maximum_bipartite_matching(reference_features, related_images_features, overlap_matrix_np, True, False)

            number_of_possible_correct_matches = sum(1 for match in matches
                                                        if (valid_matches:=match.reference_feature.get_valid_matches_for_image(related_image_index)) is not None and 
                                                        match.related_feature in valid_matches)

            numbers_of_possible_correct_matches.append(number_of_possible_correct_matches)

            number_of_reference_features = len(reference_features)

            repeatability = (
                number_of_possible_correct_matches / number_of_reference_features
                if number_of_reference_features > 0 else 0.0
            )
            repeatabilities.append(repeatability)

        set_numbers_of_possible_correct_matches.append(numbers_of_possible_correct_matches)
        set_repeatabilities.append(repeatabilities)

    return set_numbers_of_possible_correct_matches, set_repeatabilities


#@beartype
def calculate_matching_evaluation(feature_extractor : FeatureExtractor, image_feature_set : ImageFeatureSet, matching_approach : Callable, dataset_image_sequences: list[list[np.ndarray]], dataset_homography_sequence: list[list[np.ndarray]]) -> list[MatchSet]:
    matching_match_sets: list[MatchSet] = []
    for seq, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
        matching_match_set = MatchSet()
        matching_match_sets.append(matching_match_set)
        reference_features = image_feature_sequence.reference_image_features

        for rel_idx, related_image_features in enumerate(image_feature_sequence.related_images_features):
            matches : list[Match] = matching_approach(reference_features, related_image_features, feature_extractor.distance_type)
            matching_match_set.add_match(matches)

            if seq == 0:

                ## For debug ################################
                def transformed_keypoint_size(kp, H):
                    """Compute the transformed keypoint center and new size using global NUM_SAMPLE_POINTS_SCALE_CHANGE_ESTIMATION."""
                    x, y = kp.pt
                    r = kp.size / 2

                    # Sample points around the keypoint
                    angles = np.linspace(0, 2*np.pi, NUM_SAMPLE_POINTS_SCALE_CHANGE_ESTIMATION, endpoint=False)
                    pts = np.array([[x + r * np.cos(a), y + r * np.sin(a)] for a in angles], dtype=np.float32)

                    # Homogeneous coordinates
                    pts_h = np.hstack([pts, np.ones((NUM_SAMPLE_POINTS_SCALE_CHANGE_ESTIMATION, 1), dtype=np.float32)])
                    center_h = np.array([x, y, 1.0], dtype=np.float32)

                    # Apply homography
                    pts_t_h = pts_h @ H.T
                    center_t_h = H @ center_h
                    pts_t = pts_t_h[:, :2] / pts_t_h[:, 2:3]
                    center_t = center_t_h[:2] / center_t_h[2]

                    new_r = np.mean(np.linalg.norm(pts_t - center_t, axis=1))
                    return 2 * new_r, tuple(center_t.astype(int))

                # -------------------
                # Images and homography
                img_ref = dataset_image_sequences[seq][0]
                img_rel = dataset_image_sequences[seq][rel_idx + 1]
                H = dataset_homography_sequence[seq][rel_idx]

                h1, w1 = img_ref.shape[:2]
                h2, w2 = img_rel.shape[:2]

                # Sort matches
                sorted_matches = sorted(matches, key=lambda m: m.match_properties["distance"])

                # Window names
                main_window = "Matches Viewer"
                overlay_window = "Highlighted Match Overlay"
                cv2.namedWindow(main_window)
                cv2.namedWindow(overlay_window)

                # Current match index
                current_idx = 0  # 0 = first match

                def redraw_main(idx):
                    """Draw main viewer with matches up to current index."""
                    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img_ref.dtype)
                    out[:h1, :w1] = img_ref
                    out[:h2, w1:w1 + w2] = img_rel

                    n = idx + 1
                    visible = sorted_matches[:n]

                    for i, match in enumerate(visible):
                        kp1 = match.reference_feature.keypoint
                        kp2 = match.related_feature.keypoint

                        p1 = tuple(map(int, kp1.pt))
                        p2 = (int(kp2.pt[0] + w1), int(kp2.pt[1]))

                        r1 = max(2, int(kp1.size / 2))
                        r2 = max(2, int(kp2.size / 2))

                        if i == n - 1:
                            color = (0, 255, 255)  # highlighted
                            thickness = 2
                        else:
                            color = (0, 255, 0) if match.is_correct else (0, 0, 255)
                            thickness = 1

                        cv2.circle(out, p1, r1, color, thickness)
                        cv2.circle(out, p2, r2, color, thickness)
                        cv2.line(out, p1, p2, color, thickness)

                    cv2.imshow(main_window, out)

                def redraw_overlay(idx):
                    """Draw overlay image with only the highlighted match transformed."""
                    match = sorted_matches[idx]
                    kp1 = match.reference_feature.keypoint
                    kp2 = match.related_feature.keypoint

                    overlay = img_ref.copy()  # overlay only on reference image

                    # Draw reference keypoint
                    r1 = max(2, int(kp1.size / 2))
                    cv2.circle(overlay, tuple(map(int, kp1.pt)), r1, (0, 255, 0), 2)

                    # Draw transformed related keypoint
                    new_size, new_center = transformed_keypoint_size(kp2, H)
                    cv2.circle(overlay, new_center, max(2, int(new_size / 2)), (255, 0, 0), 2)

                    cv2.imshow(overlay_window, overlay)

                def update(idx):
                    """Update both windows for current index."""
                    redraw_main(idx)
                    redraw_overlay(idx)
                    cv2.setTrackbarPos("Matches", main_window, idx)

                # Trackbar callback
                def on_trackbar(val):
                    global current_idx
                    current_idx = val
                    update(current_idx)

                # Create trackbar
                cv2.createTrackbar("Matches", main_window, current_idx, len(sorted_matches)-1, on_trackbar)

                # Initial draw
                update(current_idx)

                # -------------------
                # Main loop
                while True:
                    key = cv2.waitKey(30) & 0xFF

                    if key == 27:  # ESC
                        break
                    elif key == 81 or key == ord('a'):  # LEFT arrow or 'a'
                        current_idx = max(0, current_idx - 1)
                        update(current_idx)
                    elif key == 83 or key == ord('d'):  # RIGHT arrow or 'd'
                        current_idx = min(len(sorted_matches) - 1, current_idx + 1)
                        update(current_idx)

                cv2.destroyAllWindows()

                #############################################

    return matching_match_sets


#@beartype
def calculate_verification_evaluation(feature_extractor : FeatureExtractor, image_feature_set: ImageFeatureSet, correct_to_random_ratio: int, matching_approach : Callable) -> list[MatchSet]:
    verification_match_sets: list [MatchSet] = []

    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating verification results")):
        verification_match_set = MatchSet()
        verification_match_sets.append(verification_match_set)

        choice_pool = [(choice_sequence_index, choice_image_index)  #(sequence index, image index)
            for choice_sequence_index, choice_sequence in enumerate(image_feature_set) 
            if choice_sequence_index != sequence_index 
            for choice_image_index in range(len(choice_sequence))] 
        
        for reference_feature in image_feature_sequence.reference_image_features:
            
            # Find related images with equivalent feature
            related_images_to_use = [i for i in range(len(image_feature_sequence.related_images_features)) if reference_feature.get_valid_matches_for_image(i)]
            num_random_images = len(related_images_to_use) * correct_to_random_ratio

            # Match for all relevant related images
            for related_image_features in image_feature_sequence.related_images_features:

                matches = matching_approach([reference_feature], related_image_features, feature_extractor.distance_type)
                verification_match_set.add_match(matches)

            chosen_random_images = random.sample(choice_pool, num_random_images)

            # Match for all random images
            for random_sequence_index, random_image_index in chosen_random_images:
                random_image_features = image_feature_set[random_sequence_index][random_image_index]
                match = matching_approach([reference_feature], random_image_features, feature_extractor.distance_type)
                verification_match_set.add_match(match)
    
    return verification_match_sets


#@beartype
def calculate_retrieval_evaluation(feature_extractor : FeatureExtractor, image_feature_set : ImageFeatureSet, correct_to_random_ratio : int, max_num_retrieval_features : int, matching_approach : Callable) -> list[MatchSet]:
    retrieval_match_sets : list[MatchSet] = []
    all_features = [feature 
                    for image_feature_sequence in image_feature_set
                    for image_features in image_feature_sequence
                    for feature in image_features]  
    num_features = len(all_features)          

    for image_feature_sequence in tqdm(image_feature_set, leave = False, desc = "Calculating retrieval results"):

        retrieval_match_set = MatchSet()
        retrieval_match_sets.append(retrieval_match_set)
        
        for reference_feature in image_feature_sequence.reference_image_features:
            # Choose max_num_retrieval_features
            correct_features = list(reference_feature.get_all_valid_matches())
            if len(correct_features) > max_num_retrieval_features:
                correct_features = random.sample(correct_features, max_num_retrieval_features)
            num_random_features = len(correct_features) * correct_to_random_ratio

            invalid_set = set(reference_feature.get_all_valid_matches())
            invalid_set.add(reference_feature)
                        
            num_valid_features = num_features - len(invalid_set)

            if num_random_features > num_valid_features:
                warnings.warn(f"Not enough features to fully calculate retrieval.", UserWarning)
                num_random_features = num_valid_features

            rng = np.random.default_rng() 
            random_features_idxs = set()
             
            while len(random_features_idxs) < num_random_features:
                candidate_idx = rng.integers(0, num_features)
                if (candidate_idx not in random_features_idxs):
                    candidate_feature = all_features[candidate_idx]
                    if candidate_feature not in invalid_set:
                        random_features_idxs.add(candidate_idx)

            random_features = [all_features[i] for i in random_features_idxs]
            
            # #THIS MIGHT BE FASTER FOR MORE RETRIEVAL FEATURES
            # valid_indices = [i for i in range(num_features) if all_features[i] not in invalid_set]
            # random_features = [all_features[i] for i in random.sample(valid_indices, num_random_features)]


            features_to_chose_from = correct_features + list(random_features)
            
            # Match
            match = matching_approach([reference_feature], features_to_chose_from, feature_extractor.distance_type)
            retrieval_match_set.add_match(match)
    return retrieval_match_sets