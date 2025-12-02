from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.matching import MatchSet, MatchRankingProperty, greedy_maximum_bipartite_matching_homographic_distance, greedy_maximum_bipartite_matching_descriptor_distance
from benchmark.utils import load_HPSequences
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import random
import traceback
import warnings

################################################ CONFIGURATIONS #######################################################
MAX_FEATURES = 30
RELATIVE_SCALE_DIFFERENCE_THRESHOLD = 100
ANGLE_THRESHOLD = 180
DISTANCE_THRESHOLD = 10
VERIFICATION_CORRECT_TO_RANDOM_RATIO = 5
RETRIEVAL_CORRECT_TO_RANDOM_RATIO = 400
USE_MEASUREMENT_AREA_NORMALISATION = False
#######################################################################################################################

if __name__ == "__main__":

    ####################################### SETUP TESTBENCH HERE #############################################################

    DEBUG = "all" # all/matching/verification/retrieval
    SKIP = ["speedtest", "retrieval"]

    ## Setup feature extractors.
    AGAST = cv2.AgastFeatureDetector_create()
    AKAZE = cv2.AKAZE_create()
    BRISK = cv2.BRISK_create()
    FAST = cv2.FastFeatureDetector_create()
    GFTT = cv2.GFTTDetector_create()
    KAZE = cv2.KAZE_create()
    MSER = cv2.MSER_create()
    ORB = cv2.ORB_create()
    SIFT = cv2.SIFT_create()
    SIMPLEBLOB = cv2.SimpleBlobDetector_create()

    features2d = {
        "AGAST" : AGAST,
        "AKAZE" : AKAZE,
        "BRISK" : BRISK,
        "FAST" : FAST,
        "GFTT" : GFTT,
        "KAZE" : KAZE,
        "MSER" : MSER,
        "ORB" : ORB,
        "SIFT" : SIFT,
        "SIMPLEBLOB" : SIMPLEBLOB
    }

    test_combinations: dict[str, FeatureExtractor] = {} # {Printable name of feature extraction method: feature extractor wrapper}
    
    scales = [0.05, 0.1, 0.5, 1, 1.5, 2, 3]

    for scale in scales:
        test_combinations["ORB " + str(scale)] = FeatureExtractor.from_opencv(ORB.detect, ORB.compute, cv2.NORM_HAMMING)
        test_combinations["SIFT " + str(scale)] = FeatureExtractor.from_opencv(SIFT.detect, SIFT.compute, cv2.NORM_L2)

    ## Setup matching approach
    distance_match_rank_property = MatchRankingProperty("distance", False)
    average_response_match_rank_property = MatchRankingProperty("average_response", True)
    average_ratio_match_rank_property = MatchRankingProperty("average_ratio", False)
    match_properties = [distance_match_rank_property, average_response_match_rank_property, average_ratio_match_rank_property]

    matching_approach = greedy_maximum_bipartite_matching_descriptor_distance

    #############################################################################################################################
    all_results = []

    warnings.filterwarnings("once", category=UserWarning)



    for feature_extractor_key_index, feature_extractor_key in enumerate(tqdm(test_combinations.keys(), desc = "Running tests")):
        feature_extractor: FeatureExtractor = test_combinations[feature_extractor_key]

        ## Load dataset.    
        dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release")

        scale = scales[feature_extractor_key_index//2]

        ## scale dataset.
        for sequence in dataset_image_sequences:
            for image in sequence:
                y, x = image.shape[:2]  # height, width
                new_x = int(round(x * scale))
                new_y = int(round(y * scale))
                image_resized = cv2.resize(image, (new_x, new_y), interpolation=cv2.INTER_CUBIC)


        num_sequences = len(dataset_image_sequences)
        num_related_images = len(dataset_image_sequences[0]) - 1
        image_feature_set = ImageFeatureSet(num_sequences, num_related_images)

        ## Speed test
        for feature_extractor_key in test_combinations.keys():

            feature_extractor: FeatureExtractor = test_combinations[feature_extractor_key]

            ## Speed test
            speed = 0
            if "speedtest" not in SKIP:
                time_per_image = []
                for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Calculating speed")):
                    for image_index, image in enumerate(image_sequence):
                        time = feature_extractor.get_extraction_time_on_image(image)
                        time_per_image.append(time)

                speed = np.average(time_per_image)



            ## Find features in all images.
            for sequence_index, image_sequence in enumerate(tqdm(dataset_image_sequences, leave=False, desc="Finding all features")):
                for image_index, image in enumerate(image_sequence):

                    keypoints = feature_extractor.detect_keypoints(image)
                    descriptions = feature_extractor.describe_keypoints(image, keypoints)
                    
                    features = [Feature(keypoint, description, sequence_index, image_index)
                                for _, (keypoint, description)
                                in enumerate(zip(keypoints, descriptions))]
                    
                    if MAX_FEATURES < len(features):
                        features = random.sample(features, MAX_FEATURES)
                    
                    image_feature_set[sequence_index][image_index].extend(features)



            ## Calculate valid matches.
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):

                reference_features = image_feature_sequence.reference_image
                related_images = image_feature_sequence.related_images

                num_related = len(related_images)

                for related_image_index in range(num_related):
                    related_features = related_images[related_image_index]
                    homography = dataset_homography_sequence[sequence_index][related_image_index]

                    if len(related_features) == 0:
                        continue

                    related_features_positions = np.array([feature.pt for feature in related_features], dtype=float)
                    related_features_size = np.array([feature.keypoint.size for feature in related_features], dtype=float)
                    related_features_angles = np.array([feature.keypoint.angle for feature in related_features], dtype=float)

                    # transform position
                    related_features_position_stacked = np.hstack([related_features_positions, np.ones((related_features_positions.shape[0], 1))])
                    related_features_position_stacked_T = (homography @ related_features_position_stacked.T).T
                    related_features_position_stacked_T /= related_features_position_stacked_T[:, 2:3]
                    related_features_position_transformed = related_features_position_stacked_T[:, :2]

                    # transform sizes
                    related_features_size_transformed = [related_feature.get_size_after_homography_transform(homography) for related_feature in related_features]

                    # Angles by transform unit circle angle vectors (approximate with linear part of homography)
                    related_features_angle = np.deg2rad(related_features_angles)
                    related_features_angle_stacked = np.stack([np.cos(related_features_angle), np.sin(related_features_angle)], axis=1)
                    v_h = (homography[:2,:2] @ related_features_angle_stacked.T).T
                    related_features_angle_transformed = np.rad2deg(np.arctan2(v_h[:, 1], v_h[:, 0]))

                    image_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
                    distance_threshold = DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(image_size)

                    for reference_feature in reference_features:
                        
                        # Check distances
                        reference_feature_position = reference_feature.pt 
                        distances = np.linalg.norm(related_features_position_transformed - reference_feature_position, axis=1)

                        # Check scales differences
                        reference_feature_size = reference_feature.keypoint.size
                        biggest = np.maximum(reference_feature_size, related_features_size_transformed)
                        smallest = np.minimum(reference_feature_size, related_features_size_transformed)
                        relative_scale_difference = np.abs(1 - biggest / smallest)

                        # Check angles differences
                        reference_feature_angle = reference_feature.keypoint.angle
                        angle_difference = np.abs((reference_feature_angle - related_features_angle_transformed + 180) % 360 - 180)

                        # Create check mask
                        mask = (
                            (distances <= distance_threshold) &
                            (relative_scale_difference <= RELATIVE_SCALE_DIFFERENCE_THRESHOLD) &
                            (angle_difference <= ANGLE_THRESHOLD)
                        )

                        valid_feature_indexes = np.nonzero(mask)[0]
                        if valid_feature_indexes.size == 0:
                            continue

                        # Store valid features for that check mask
                        for index in valid_feature_indexes:
                            related_feature = related_features[index]
                            distance = distances[index]
                            reference_feature.store_valid_match_for_image(related_image_index, related_feature, distance)
                            related_feature.store_valid_match_for_image(0, reference_feature, distance)



            ## Calculate repeatability and number of possible matches.
            set_numbers_of_possible_correct_matches= []
            set_repeatabilities = []
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

                numbers_of_possible_correct_matches = []
                repeatabilities = []

                reference_features = image_feature_sequence.reference_image
                number_of_reference_features = len(reference_features)

                for related_image_index, related_image_features in enumerate(image_feature_sequence.related_images):
                    
                    homography = dataset_homography_sequence[sequence_index][related_image_index]

                    # Run matching
                    matches = greedy_maximum_bipartite_matching_homographic_distance(
                        reference_features.copy(),
                        related_image_features.copy(),
                        homography
                    )

                    # Count how many matches are valid
                    number_of_possible_correct_matches = 0

                    for match in matches:
                        reference_feature = match.feature1
                        related_feature = match.feature2

                        features_for_valid_match = reference_feature.get_valid_matches_for_image(related_image_index)

                        if features_for_valid_match is not None and related_feature in features_for_valid_match:
                            number_of_possible_correct_matches += 1

                    numbers_of_possible_correct_matches.append(number_of_possible_correct_matches)

                    repeatability = (
                        number_of_possible_correct_matches / number_of_reference_features
                        if number_of_reference_features > 0 else 0.0
                    )
                    repeatabilities.append(repeatability)

                set_numbers_of_possible_correct_matches.append(numbers_of_possible_correct_matches)
                set_repeatabilities.append(repeatabilities)



            ## Calculate matching results.
            matching_match_sets: list[MatchSet] = []
            if DEBUG == "all" or DEBUG == "matching":
                for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
                    matching_match_set = MatchSet()
                    matching_match_sets.append(matching_match_set)
                    for related_image_index, related_image in enumerate(image_feature_sequence.related_images):

                        # Reference and related features.
                        reference_features = image_feature_sequence.reference_image.copy()
                        related_features = image_feature_sequence.related_image(related_image_index).copy()

                        matches = matching_approach(reference_features, related_features, feature_extractor.distance_type)
                        matching_match_set.add_match(matches)




            ## Calculate verification results.
            verification_match_sets: list [MatchSet] = []
            if DEBUG == "all" or DEBUG == "verification":
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
                        
                        num_random_images = len(related_images_to_use) * VERIFICATION_CORRECT_TO_RANDOM_RATIO

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
                            matches = matching_approach([reference_feature], random_image_features, feature_extractor.distance_type)
                            verification_match_set.add_match(matches)


            ## Calculate retrieval results.
            retrieval_match_sets : list[MatchSet] = []
            all_features = [feature 
                            for choice_image_feature_sequence in image_feature_set
                            for choice_image_features in choice_image_feature_sequence
                            for feature in choice_image_features.copy()]            

            all_features_array = np.array(all_features, dtype=object)
            if "retrieval" not in SKIP and DEBUG == "all" or DEBUG == "retrieval" :
                for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating retrieval results")):

                    retrieval_match_set = MatchSet()
                    retrieval_match_sets.append(retrieval_match_set)
                    reference_features = image_feature_sequence.reference_image.copy()
                    
                    for reference_feature in reference_features:

                        # Choose maximum 5 correct features
                        correct_features = reference_feature.get_all_valid_matches()
                        if len(correct_features) > 5:
                            correct_features = random.sample(correct_features, 5)

                        invalid_set = set(reference_feature.get_all_valid_matches())
                        invalid_set.add(reference_feature)
                        invalid_mask = np.array([feature not in invalid_set for feature in all_features_array])
                        valid_features = all_features_array[invalid_mask]
                        num_random_features = len(correct_features) * RETRIEVAL_CORRECT_TO_RANDOM_RATIO
                        
                        # Pick random features
                        if num_random_features > len(valid_features):
                            warnings.warn(f"Not enough features to fully calculate retrieval.", UserWarning)
                            num_random_features = len(valid_features)
                            
                        random_features = np.random.choice(valid_features, size=num_random_features, replace=False)

                        features_to_chose_from = correct_features + list(random_features)
                        
                        # Match
                        match = matching_approach([reference_feature], features_to_chose_from, feature_extractor.distance_type)
                        retrieval_match_set.add_match(match)

                        

            ## Store results
            set_numbers_of_possible_correct_matches = np.array(set_numbers_of_possible_correct_matches)
            set_numbers_of_possible_correct_matches.flatten()

            set_repeatabilities = np.array(set_repeatabilities)
            set_repeatabilities.flatten()

            total_possible_correct_matches = sum(
                num_correct_matches
                for num_correct_sequence_matches in set_numbers_of_possible_correct_matches
                for num_correct_matches in num_correct_sequence_matches
            )

            total_correct_matches = sum(
                1 if match.is_correct else 0
                for match_set in matching_match_sets
                for match in match_set
            )

            results = {
                #"combination": feature_extractor_key,
                #"speed": speed,
                #"cm_total: mean" : np.mean(set_numbers_of_possible_correct_matches),
                #"cm_total: std" : np.std(set_numbers_of_possible_correct_matches),
                "rep_total: mean" : np.mean(set_repeatabilities),
                "rep_total: std" : np.std(set_repeatabilities),
                "total num matches" : sum(len(match_set) for match_set in matching_match_sets),
                "num possible correct matches" : total_possible_correct_matches,
                "total correct matches" : total_correct_matches
            }

            for match_rank_property in match_properties:
                mAP = np.average([match_set.get_average_precision_score(match_rank_property) for match_set in matching_match_sets])
                results[f"Matching {match_rank_property.name} mAP"] =  mAP

            # Results from verification
            for match_ranking_property in match_properties:
                mAP = np.average([match_set.get_average_precision_score(match_ranking_property) for match_set in verification_match_sets])
                results[f"Verification {match_ranking_property.name} mAP"] = mAP

            # Results from retrieval
            for match_ranking_property in match_properties:
                mAP = np.average([match_set.get_average_precision_score(match_ranking_property) for match_set in retrieval_match_sets])
                results[f"Retrieval {match_ranking_property.name} mAP"] = mAP

            all_results.append(results)

        
            
    ################################################ STORE RESULTS ##############################################################
    for metric, result in results.items():
        print(metric, result)
    df = pd.DataFrame(all_results)
    df.to_csv("output.csv", index = False)
