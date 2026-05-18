from tqdm import tqdm
from scipy.spatial.distance import cdist
from benchmark.utils import calculate_overlap_one_circle_to_many, downsample, visualize_matches_with_scale_change, non_maximal_supression
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
def find_all_features_for_dataset(feature_extractor: FeatureExtractor, dataset_image_sequences: list[list[np.ndarray]], image_feature_set: ImageFeatureSet, max_features: int, keypoint_size_scaling: int, FORCE_CONSTANT_ANGLE: bool, DOWNSAMPLE_LEVEL: int, DOWNSAMPLE_FACTOR: float, INTRINSIC_SIGMA: float, INITIAL_SIGMA: float, APPLY_PROGRESSIVE_BLUR: bool, DOWNSAMPLE_INTERPOLATION_TYPE):  

    for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Finding all features")):
        for image_index, image in enumerate(image_sequence):
            
            # for i in range(DOWNSAMPLE_ITERATIONS):
            #     image = downsample(image,DOWNSAMPLE_FACTOR,1.2, DOWNSAMPLE_INTERPOLATION_TYPE)

            image = downsample(image, DOWNSAMPLE_LEVEL, DOWNSAMPLE_FACTOR, INTRINSIC_SIGMA, INITIAL_SIGMA, APPLY_PROGRESSIVE_BLUR, DOWNSAMPLE_INTERPOLATION_TYPE)

            keypoints = feature_extractor.detect_keypoints(image)
            num_keypoints = len(keypoints)
            if (num_keypoints == 0):
                continue

            min_required_keypoints = round(max_features*1.1)

            if APPLY_NMS:
                keypoints = non_maximal_supression(keypoints, NMS_RADIUS, min_required_keypoints)
            else:
                if min_required_keypoints < len(keypoints):
                    scores = np.array([keypoint.response for keypoint in keypoints])
                    idx = np.argpartition(scores, -min_required_keypoints)[-min_required_keypoints:]
                    keypoints = [keypoints[i] for i in idx]

            for keypoint in keypoints:
                keypoint.size = keypoint.size * keypoint_size_scaling
                if (FORCE_CONSTANT_ANGLE):
                    keypoint.angle = 0
            
            # if len(keypoints) > 250:
            #     keypoints = keypoints[250:500]

            keypoints, descriptions = feature_extractor.describe_keypoints(image, keypoints)

            for keypoint in keypoints:
                keypoint.pt = (keypoint.pt[0] * DOWNSAMPLE_FACTOR ** DOWNSAMPLE_LEVEL, keypoint.pt[1] * DOWNSAMPLE_FACTOR ** DOWNSAMPLE_LEVEL)

            if num_keypoints > min_required_keypoints and len(keypoints) < max_features and not APPLY_NMS:
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

            # if (num_keypoints < max_features):
            #     print("seq ", sequence_index, " ", image_index, " ", max_features-num_keypoints)
            

            # For debug ################################
            
            # best_keypoint = max(list(keypoints), key=lambda kp: kp.response)
            # biggest_keypoint = max(list(keypoints), key=lambda kp: kp.size)
            # print(best_keypoint.size)
            # print(best_keypoint.response)
            # print(biggest_keypoint.size)
            # print(biggest_keypoint.response)
            # out_image = cv2.drawKeypoints(image, [max(list(keypoints), key=lambda kp: kp.size), max(list(keypoints), key=lambda kp: kp.response)], None, color=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # #out_image = cv2.drawKeypoints(image, keypoints, None, color=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imshow("Keypoints", out_image)
            # cv2.waitKey(0)   # 200 ms = 0.2 s
            # cv2.destroyAllWindows()
            # ############################################
            
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
def calculate_valid_matches(image_feature_set: ImageFeatureSet, dataset_homography_sequence: list[list[np.ndarray]]):

    set_numbers_of_possible_correct_matches = []
    set_repeatabilities = []

    _angles = np.linspace(0, 2 * np.pi, NUM_SAMPLE_POINTS_SCALE_CHANGE_ESTIMATION, endpoint=False)
    _cos_a = np.cos(_angles)
    _sin_a = np.sin(_angles)
    EPS = 1e-12

    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):

        numbers_of_possible_correct_matches = []
        repeatabilities = []
        reference_features = image_feature_sequence.reference_image_features

        ref_pts = np.array([f.keypoint.pt for f in reference_features], dtype=np.float64)  # (n_ref, 2)
        if not USE_DISTANCE:
            ref_radii = np.array([f.keypoint.size / 2 for f in reference_features], dtype=np.float64)  # (n_ref,)

        for related_image_index, related_images_features in enumerate(image_feature_sequence.related_images_features):

            homography = dataset_homography_sequence[sequence_index][related_image_index]

            if len(related_images_features) == 0:
                numbers_of_possible_correct_matches.append(0)
                repeatabilities.append(0.0)
                continue

            n_rel = len(related_images_features)

            # Batch position transform for all related features
            rel_pts = np.array([f.keypoint.pt for f in related_images_features], dtype=np.float64)  # (n_rel, 2)
            rel_pts_h = np.column_stack([rel_pts, np.ones(n_rel)])                                   # (n_rel, 3)
            rel_t_h = rel_pts_h @ homography.T                                                       # (n_rel, 3)
            related_pos_t = rel_t_h[:, :2] / rel_t_h[:, 2:3]                                        # (n_rel, 2)

            # Full distance matrix in one call
            distance_matrix = cdist(ref_pts, related_pos_t)  # (n_ref, n_rel)

            if USE_DISTANCE:
                closeness_matrix_np = distance_matrix
                valid_mask = distance_matrix <= DISTANCE_THRESHOLD
            else:
                # Batch size transform: sample circle points around every related keypoint
                rel_radii_orig = np.array([f.keypoint.size / 2 for f in related_images_features], dtype=np.float64)
                sx = rel_pts[:, 0:1] + rel_radii_orig[:, None] * _cos_a  # (n_rel, n_sample)
                sy = rel_pts[:, 1:2] + rel_radii_orig[:, None] * _sin_a
                sample_h = np.stack([sx, sy, np.ones_like(sx)], axis=2)  # (n_rel, n_sample, 3)
                sample_t_h = sample_h @ homography.T                      # (n_rel, n_sample, 3)
                sample_t = sample_t_h[:, :, :2] / sample_t_h[:, :, 2:3]  # (n_rel, n_sample, 2)
                diffs = sample_t - related_pos_t[:, None, :]              # (n_rel, n_sample, 2)
                rel_radii_t = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (n_rel,)

                # Vectorized overlap matrix
                r1 = ref_radii[:, None]    # (n_ref, 1)
                r2 = rel_radii_t[None, :]  # (1, n_rel)
                d = distance_matrix

                intersection = np.zeros_like(d)
                disjoint  = d >= r1 + r2
                contained = (~disjoint) & (d <= np.abs(r1 - r2))
                partial   = (~disjoint) & (~contained)

                if contained.any():
                    intersection[contained] = np.pi * np.minimum(
                        np.broadcast_to(r1, d.shape)[contained],
                        np.broadcast_to(r2, d.shape)[contained],
                    ) ** 2

                if partial.any():
                    r1b = np.broadcast_to(r1, d.shape)
                    r2b = np.broadcast_to(r2, d.shape)
                    dp, r1p, r2p = d[partial], r1b[partial], r2b[partial]
                    cos1 = np.clip((dp**2 + r1p**2 - r2p**2) / (2*dp*r1p + EPS), -1, 1)
                    cos2 = np.clip((dp**2 + r2p**2 - r1p**2) / (2*dp*r2p + EPS), -1, 1)
                    sq   = np.clip((-dp+r1p+r2p)*(dp+r1p-r2p)*(dp-r1p+r2p)*(dp+r1p+r2p), 0, None)
                    intersection[partial] = r1p**2 * np.arccos(cos1) + r2p**2 * np.arccos(cos2) - 0.5*np.sqrt(sq)

                frac_ref = intersection / (np.pi * ref_radii[:, None]**2 + EPS)
                frac_rel = intersection / (np.pi * rel_radii_t[None, :]**2 + EPS)
                closeness_matrix_np = np.minimum(frac_ref, frac_rel)
                valid_mask = closeness_matrix_np >= FEATURE_OVERLAP_THRESHOLD

            # Store valid match pairs
            valid_ref_idxs, valid_rel_idxs = np.where(valid_mask)
            for ref_idx, rel_idx in zip(valid_ref_idxs.tolist(), valid_rel_idxs.tolist()):
                ref_f = reference_features[ref_idx]
                rel_f = related_images_features[rel_idx]
                ref_f.store_valid_match_for_image(related_image_index, rel_f)
                rel_f.store_valid_match_for_image(0, ref_f)

            # Sparse greedy over valid pairs only — equivalent to full-matrix greedy because
            # valid pairs (dist ≤ threshold or overlap ≥ threshold) always rank ahead of
            # invalid pairs in the greedy ordering, so the assignment among valid pairs
            # is unaffected by including invalid pairs in the matrix.
            if valid_ref_idxs.size == 0:
                number_of_possible_correct_matches = 0
            else:
                if USE_DISTANCE:
                    order = np.argsort(distance_matrix[valid_ref_idxs, valid_rel_idxs])
                else:
                    order = np.argsort(-closeness_matrix_np[valid_ref_idxs, valid_rel_idxs])
                sorted_refs = valid_ref_idxs[order].tolist()
                sorted_rels = valid_rel_idxs[order].tolist()
                matched_refs: set = set()
                matched_rels: set = set()
                number_of_possible_correct_matches = 0
                for r, q in zip(sorted_refs, sorted_rels):
                    if r not in matched_refs and q not in matched_rels:
                        matched_refs.add(r)
                        matched_rels.add(q)
                        number_of_possible_correct_matches += 1

            numbers_of_possible_correct_matches.append(number_of_possible_correct_matches)
            repeatability = (
                number_of_possible_correct_matches / len(reference_features) if reference_features else 0.0
            )
            repeatabilities.append(repeatability)

        set_numbers_of_possible_correct_matches.append(numbers_of_possible_correct_matches)
        set_repeatabilities.append(repeatabilities)

    return set_numbers_of_possible_correct_matches, set_repeatabilities


#@beartype
def calculate_matching_evaluation(feature_extractor : FeatureExtractor, image_feature_set : ImageFeatureSet, matching_approach : Callable, use_mnn : bool, dataset_image_sequences: list[list[np.ndarray]], dataset_homography_sequence: list[list[np.ndarray]], visualize: bool, seqs_to_visualize: int, DOWNSAMPLE_LEVEL: int, DOWNSAMPLE_FACTOR: float, INTRINSIC_SIGMA: float, INITIAL_SIGMA: float, APPLY_PROGRESSIVE_BLUR: bool, DOWNSAMPLE_INTERPOLATION_TYPE) -> list[MatchSet]:
    matching_match_sets: list[MatchSet] = []
    for seq_num, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
        matching_match_set = MatchSet()
        matching_match_sets.append(matching_match_set)
        reference_features = image_feature_sequence.reference_image_features

        for rel_idx, related_image_features in enumerate(image_feature_sequence.related_images_features):
            matches : list[Match] = matching_approach(reference_features, related_image_features, feature_extractor.distance_type, use_mnn)
            matching_match_set.add_match(matches)

            if visualize and seq_num in seqs_to_visualize:
                ref_image = cv2.resize(downsample(dataset_image_sequences[seq_num][0], DOWNSAMPLE_LEVEL, DOWNSAMPLE_FACTOR, INTRINSIC_SIGMA, INITIAL_SIGMA, APPLY_PROGRESSIVE_BLUR, DOWNSAMPLE_INTERPOLATION_TYPE), None, fx= DOWNSAMPLE_FACTOR**DOWNSAMPLE_LEVEL, fy= DOWNSAMPLE_FACTOR**DOWNSAMPLE_LEVEL, interpolation= INTER_NEAREST)
                rel_image = cv2.resize(downsample(dataset_image_sequences[seq_num][rel_idx+1], DOWNSAMPLE_LEVEL, DOWNSAMPLE_FACTOR, INTRINSIC_SIGMA, INITIAL_SIGMA, APPLY_PROGRESSIVE_BLUR, DOWNSAMPLE_INTERPOLATION_TYPE), None, fx= DOWNSAMPLE_FACTOR**DOWNSAMPLE_LEVEL, fy= DOWNSAMPLE_FACTOR**DOWNSAMPLE_LEVEL, interpolation= INTER_NEAREST)
                homography = dataset_homography_sequence[seq_num][rel_idx]
                visualize_matches_with_scale_change(NUM_SAMPLE_POINTS_SCALE_CHANGE_ESTIMATION, ref_image, rel_image, homography, matches)
                

    return matching_match_sets


#@beartype
def calculate_verification_evaluation(feature_extractor : FeatureExtractor, image_feature_set: ImageFeatureSet, correct_to_random_ratio: int, matching_approach : Callable, use_mnn : bool) -> list[MatchSet]:
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

                matches = matching_approach([reference_feature], related_image_features, feature_extractor.distance_type, use_mnn)
                verification_match_set.add_match(matches)

            chosen_random_images = random.sample(choice_pool, num_random_images)

            # Match for all random images
            for random_sequence_index, random_image_index in chosen_random_images:
                random_image_features = image_feature_set[random_sequence_index][random_image_index]
                match = matching_approach([reference_feature], random_image_features, feature_extractor.distance_type, use_mnn)
                verification_match_set.add_match(match)
    
    return verification_match_sets


#@beartype
def calculate_retrieval_evaluation(feature_extractor : FeatureExtractor, image_feature_set : ImageFeatureSet, correct_to_random_ratio : int, max_num_retrieval_features : int, matching_approach : Callable , use_mnn : bool) -> list[MatchSet]:
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
            match = matching_approach([reference_feature], features_to_chose_from, feature_extractor.distance_type, use_mnn)
            retrieval_match_set.add_match(match)
    return retrieval_match_sets