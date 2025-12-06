from tqdm import tqdm
from benchmark.utils import calculate_overlap_one_circle_to_many
from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.matching import MatchSet, greedy_maximum_bipartite_matching
from typing import Tuple, Callable
import random
import numpy as np
import warnings
from beartype import beartype


@beartype
def speed_test(feature_extractor: FeatureExtractor, dataset_image_sequences: list[list[np.ndarray]]):
     
    time_per_image = []
    for _, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Calculating speed")):
        for _, image in enumerate(image_sequence):
            time = feature_extractor.get_extraction_time_on_image(image)
            time_per_image.append(time)

    time = np.average(time_per_image)
    speed = 1/time
    return speed


@beartype
def find_all_features_for_dataset(feature_extractor: FeatureExtractor, dataset_image_sequences: list[list[np.ndarray]], image_feature_set: ImageFeatureSet, max_features: int):  

    for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Finding all features")):
        for image_index, image in enumerate(image_sequence):

            keypoints = feature_extractor.detect_keypoints(image)
            descriptions = feature_extractor.describe_keypoints(image, keypoints)
            
            features = [Feature(keypoint, description, sequence_index, image_index)
                        for _, (keypoint, description)
                        in enumerate(zip(keypoints, descriptions))]
            
            if max_features < len(features):
                features = random.sample(features, max_features)
            
            image_feature_set[sequence_index][image_index] = features


@beartype
def calculate_valid_matches(image_feature_set: ImageFeatureSet, dataset_homography_sequence: list[list[np.ndarray]], threshold: float, use_overlap: bool = True):
    
    set_numbers_of_possible_correct_matches= []
    set_repeatabilities = []
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):

        numbers_of_possible_correct_matches = []
        repeatabilities = []

        number_of_reference_features = len(image_feature_sequence.reference_image)
        reference_features = image_feature_sequence.reference_image

        for related_image_index, related_image_features in enumerate(image_feature_sequence.related_images):

            homography = dataset_homography_sequence[sequence_index][related_image_index]

            if len(related_image_features) == 0:
                continue

            # transform position 
            related_features_position_transformed = np.array([feature.get_pt_after_homography_transform(homography) for feature in related_image_features])

            # transform sizes
            related_features_size_transformed = np.array([feature.get_size_after_homography_transform(homography) for feature in related_image_features])
            overlap_matrix = []
            for reference_feature in reference_features:
                
                # Check distances
                distances = np.linalg.norm(related_features_position_transformed - reference_feature.pt, axis=1)
                
                # Create check mask
                if use_overlap:
                    ref_radius   = float(reference_feature.keypoint.size) / 2.0
                    rel_radii    = related_features_size_transformed / 2.0
                    EPS = 1e-12  # small epsilon for numerical stability

                    ref_area  = np.pi * (ref_radius ** 2)      # scalar
                    rel_areas = np.pi * (rel_radii  ** 2)      # vector

                    # Intersection area (vectorized)
                    intersectional_area = np.zeros_like(distances, dtype=float)
        
                    # Case 1: disjoint (no overlap)
                    disjoint_mask  = distances >= ref_radius + rel_radii

                    # Case 2: one circle fully contained in the other
                    contained_mask = distances <= np.abs(ref_radius - rel_radii)
                    if np.any(contained_mask):
                        intersectional_area[contained_mask] = np.pi * (np.minimum(ref_radius, rel_radii[contained_mask]) ** 2)

                    # Case 3: partial overlap (lens)
                    partial_mask = (~disjoint_mask) & (~contained_mask)
                    if np.any(partial_mask):
                        distances_partial  = distances[partial_mask]
                        rel_radii_partial = rel_radii[partial_mask]

                        # Stable arccos arguments
                        cos1 = (distances_partial**2 + ref_radius**2 - rel_radii_partial**2) / (2.0 * distances_partial * ref_radius + EPS)
                        cos2 = (distances_partial**2 + rel_radii_partial**2 - ref_radius**2) / (2.0 * distances_partial * rel_radii_partial      + EPS)
                        cos1 = np.clip(cos1, -1.0, 1.0)
                        cos2 = np.clip(cos2, -1.0, 1.0)

                        #MATH for overlap of circles
                        term1 = ref_radius**2 * np.arccos(cos1)
                        term2 = rel_radii_partial**2      * np.arccos(cos2)
                        sq = (-distances_partial + ref_radius + rel_radii_partial) * (distances_partial + ref_radius - rel_radii_partial) * (distances_partial - ref_radius + rel_radii_partial) * (distances_partial + ref_radius + rel_radii_partial)
                        term3 = 0.5 * np.sqrt(np.clip(sq, 0.0, None))

                        intersectional_area[partial_mask] = term1 + term2 - term3

                    # Overlap fractions — require BOTH circles to meet the threshold
                    overlap_ref_frac = intersectional_area / (ref_area  + EPS)   # coverage of the reference circle
                    overlap_rel_frac = intersectional_area / (rel_areas + EPS)   # coverage of each related circle
                    overlap_min = np.minimum(overlap_ref_frac,overlap_rel_frac)
                    overlap_matrix.append(overlap_min)

                    # Final mask: ONLY overlap criterion
                    mask = (overlap_ref_frac >= threshold) & (overlap_rel_frac >= threshold)

                else:
                    mask = (
                        (distances <= threshold)
                    )

                valid_feature_indexes = np.nonzero(mask)[0]
                if valid_feature_indexes.size == 0:
                    continue

                # Store valid features for that check mask
                for index in valid_feature_indexes:
                    related_feature = related_image_features[index]
                    distance = distances[index]
                    reference_feature.store_valid_match_for_image(related_image_index, related_feature, distance)
                    related_feature.store_valid_match_for_image(0, reference_feature, distance)

            # Run matching
            overlap_matrix_np = np.array(overlap_matrix)

            matches = greedy_maximum_bipartite_matching(reference_features, related_image_features, overlap_matrix_np, True, False)

            number_of_possible_correct_matches = sum(1 for match in matches
                                                        if (valid_matches:=match.reference_feature.get_valid_matches_for_image(related_image_index)) is not None and 
                                                        match.related_feature in valid_matches)

            numbers_of_possible_correct_matches.append(number_of_possible_correct_matches)

            repeatability = (
                number_of_possible_correct_matches / number_of_reference_features
                if number_of_reference_features > 0 else 0.0
            )
            repeatabilities.append(repeatability)

        set_numbers_of_possible_correct_matches.append(numbers_of_possible_correct_matches)
        set_repeatabilities.append(repeatabilities)

    return set_numbers_of_possible_correct_matches, set_repeatabilities


@beartype
def calculate_matching_evaluation(feature_extractor : FeatureExtractor, image_feature_set : ImageFeatureSet, matching_approach : Callable) -> list[MatchSet]:
    matching_match_sets: list[MatchSet] = []
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
        matching_match_set = MatchSet()
        matching_match_sets.append(matching_match_set)
        for related_image_index, related_image in enumerate(image_feature_sequence.related_images):

            # Reference and related features.
            reference_features = image_feature_sequence.reference_image.copy()
            related_features = image_feature_sequence.related_image(related_image_index).copy()

            matches = matching_approach(reference_features, related_features, feature_extractor.distance_type)
            matching_match_set.add_match(matches)
    return matching_match_sets


@beartype
def calculate_verification_evaluation(feature_extractor : FeatureExtractor, image_feature_set: ImageFeatureSet, correct_to_random_ratio: int, matching_approach : Callable) -> list[MatchSet]:
    verification_match_sets: list [MatchSet] = []
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating verification results")):
        verification_match_set = MatchSet()
        verification_match_sets.append(verification_match_set)
        reference_features = image_feature_sequence.reference_image.copy()
        for reference_feature in reference_features:
            
            # Find related images with equivalent feature
            related_images_to_use = []
            for image_index in range(len(image_feature_sequence.related_images)):
                if isinstance(reference_feature.get_valid_matches_for_image(image_index), dict):
                    related_images_to_use.append(image_index)
            
            num_random_images = len(related_images_to_use) * correct_to_random_ratio

            # Match for all relevant related images
            for related_image_index in range(len(related_images_to_use)):

                # Reference and related features.
                related_features = image_feature_sequence.related_image(related_image_index).copy()

                matches = matching_approach([reference_feature], related_features, feature_extractor.distance_type)
                verification_match_set.add_match(matches)
            
            # Pick random images
            choice_pool = [(choice_sequence_index, choice_image_index)  #(sequence index, image index)
                        for choice_sequence_index, choice_sequence in enumerate(image_feature_set) 
                        if choice_sequence_index != sequence_index 
                        for choice_image_index in range(len(choice_sequence))] 
            
            chosen_random_images = random.sample(choice_pool, num_random_images)
            
            # Match for all random images
            for random_sequence_index, random_image_index in chosen_random_images:
                random_image_features = image_feature_set[random_sequence_index][random_image_index].copy()
                match = matching_approach([reference_feature], random_image_features, feature_extractor.distance_type)
                verification_match_set.add_match(match)
    
    return verification_match_sets


@beartype
def calculate_retrieval_evaluation(feature_extractor : FeatureExtractor, image_feature_set : ImageFeatureSet, correct_to_random_ratio : int, matching_approach : Callable) -> list[MatchSet]:
    retrieval_match_sets : list[MatchSet] = []
    all_features = [feature 
                    for choice_image_feature_sequence in image_feature_set
                    for choice_image_features in choice_image_feature_sequence
                    for feature in choice_image_features.copy()]            

    all_features_array = np.array(all_features, dtype=object)
    for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating retrieval results")):

        retrieval_match_set = MatchSet()
        retrieval_match_sets.append(retrieval_match_set)
        reference_features = image_feature_sequence.reference_image.copy()
        
        for reference_feature in reference_features:

            # Choose maximum 5 correct features
            correct_features = list(reference_feature.get_all_valid_matches().keys())
            if len(correct_features) > 5:
                correct_features = random.sample(correct_features, 5)

            invalid_set = set(reference_feature.get_all_valid_matches())
            invalid_set.add(reference_feature)
            invalid_mask = np.array([feature not in invalid_set for feature in all_features_array])
            valid_features = all_features_array[invalid_mask]
            num_random_features = len(correct_features) * correct_to_random_ratio
            
            # Pick random features
            if num_random_features > len(valid_features):
                warnings.warn(f"Not enough features to fully calculate retrieval.", UserWarning)
                num_random_features = len(valid_features)
                
            random_features = np.random.choice(valid_features, size=num_random_features, replace=False)

            features_to_chose_from = correct_features + list(random_features)
            
            # Match
            match = matching_approach([reference_feature], features_to_chose_from, feature_extractor.distance_type)
            retrieval_match_set.add_match(match)
    return retrieval_match_sets