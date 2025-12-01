from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.matching import MatchSet, MatchRankingProperty,homographic_optimal_matching, greedy_maximum_bipartite_matching
from benchmark.utils import load_HPSequences
from tqdm import tqdm
import cv2
import math
import numpy as np
import random
import warnings
import pandas as pd
import traceback

################################################ CONFIGURATIONS #######################################################
MAX_FEATURES = 500
RELATIVE_SCALE_DIFFERENCE_THRESHOLD = 100
ANGLE_THRESHOLD = 180
DISTANCE_THRESHOLD = 10
VERIFICATION_CORRECT_TO_RANDOM_RATIO = 5
RETRIEVAL_CORRECT_TO_RANDOM_RATIO = 1000
USE_MEASUREMENT_AREA_NORMALISATION = False
#######################################################################################################################

if __name__ == "__main__":

    ## Load dataset.    
    dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release") 

    ####################################### SETUP TESTBENCH HERE #############################################################

    DEBUG = "all" # all/matching/verification/retrieval

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
        # "AGAST" : AGAST,
        # "AKAZE" : AKAZE,
        # "BRISK" : BRISK,
        # "FAST" : FAST,
        # "GFTT" : GFTT,
        # "KAZE" : KAZE,
        # "MSER" : MSER,
        "ORB" : ORB,
        # "SIFT" : SIFT,
        # "SIMPLEBLOB" : SIMPLEBLOB
    }

    test_combinations: dict[str, FeatureExtractor] = {} # {Printable name of feature extraction method: feature extractor wrapper}

    for detector_key in features2d.keys():
        for descriptor_key in features2d.keys():
            distance_type = ""
            if descriptor_key in ["ORB", "AKAZE"]: 
                distance_type = cv2.NORM_HAMMING
            else: 
                distance_type = cv2.NORM_L2
            test_combinations[detector_key + "+" + descriptor_key] = FeatureExtractor.from_opencv(features2d[detector_key].detect, features2d[descriptor_key].compute, distance_type, use_normalisation=USE_MEASUREMENT_AREA_NORMALISATION)

    ## Setup matching approach
    distance_match_rank_property = MatchRankingProperty("distance", False)
    average_response_match_rank_property = MatchRankingProperty("average_response", True)
    average_ratio_match_rank_property = MatchRankingProperty("average_ratio", False)
    match_properties = [distance_match_rank_property, average_response_match_rank_property, average_ratio_match_rank_property]

    matching_approach = greedy_maximum_bipartite_matching

    #############################################################################################################################
    all_results = []

    warnings.filterwarnings("once", category=UserWarning)


    num_sequences = len(dataset_image_sequences)
    num_related_images = len(dataset_image_sequences[0]) - 1
    image_feature_set = ImageFeatureSet(num_sequences, num_related_images)

    for feature_extractor_key in test_combinations.keys():
        try:
            feature_extractor: FeatureExtractor = test_combinations[feature_extractor_key]



            ## Speed test
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

                    features.sort(key= lambda x: x.keypoint.response)
                    
                    if MAX_FEATURES < len(features):
                        features = random.sample(features, MAX_FEATURES)
                    
                    image_feature_set[sequence_index][image_index].extend(features)



            ## Calculate valid matches.
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating and ranking all valid matches")):
                for reference_feature in tqdm(image_feature_sequence.reference_image, leave=False):
                    for related_image_index, related_image in enumerate(image_feature_sequence.related_images):
                        img_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
                        for related_feature in related_image:

                            homography = dataset_homography_sequence[sequence_index][related_image_index]
                            transformed_related_feature_pt = related_feature.get_pt_after_homography_transform(homography)
                            distance = math.dist(reference_feature.pt, transformed_related_feature_pt)

                            distance_threshold = (DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size))

                            if distance > distance_threshold:
                                continue

                            reference_feature_size = reference_feature.keypoint.size
                            related_feature_size = related_feature.get_size_after_homography_transform(homography)
                            biggest_keypoint = max(reference_feature_size, related_feature_size)
                            smallest_keypoint = min(reference_feature_size, related_feature_size)
                            relative_scale_differnce = abs(1 - biggest_keypoint/smallest_keypoint)
                            if relative_scale_differnce > RELATIVE_SCALE_DIFFERENCE_THRESHOLD:
                                continue
                            
                            reference_feature_angle = reference_feature.keypoint.angle
                            related_feature_angle = related_feature.get_angle_after_homography(homography)
                            # Calculate the circular angle distance
                            angle_difference = abs((reference_feature_angle - related_feature_angle + 180) % 360 - 180)
                            if angle_difference > ANGLE_THRESHOLD:
                                continue

                            reference_feature.store_valid_match_for_image(related_image_index, related_feature, distance)
                            related_feature.store_valid_match_for_image(0, reference_feature, distance)



            ## Calculate repeatability and number of possible matches.
            set_nums_possible_correct_matches= []
            set_repeatabilities = []
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating optimal matching results")):

                nums_possible_correct_matches = []
                repeatabilities = []
                
                for related_image_index, related_image in enumerate(image_feature_sequence.related_images):

                    img_size = len(dataset_image_sequences[sequence_index][related_image_index+1])
                    homography = dataset_homography_sequence[sequence_index][related_image_index]
                    
                    reference_features = image_feature_sequence.reference_image.copy()
                    related_features = image_feature_sequence.related_image(related_image_index).copy()

                    matches = homographic_optimal_matching(reference_features, related_features, homography) 

                    # Check which matches were correct
                    num_possible_correct_matches = 0
                    for match in matches:
                            
                        feature1 = match.feature1
                        feature2 = match.feature2
                        
                        homography = dataset_homography_sequence[sequence_index][related_image_index]
                        transformed_feature_2_pt = feature2.get_pt_after_homography_transform(homography)
                        distance = math.dist(feature1.pt, transformed_feature_2_pt)

                        distance_threshold = (DISTANCE_THRESHOLD * feature_extractor.get_description_image_scale_factor(img_size))

                        if distance > distance_threshold:
                            continue

                        feature1_size = feature1.keypoint.size
                        feature2_size = feature2.get_size_after_homography_transform(homography)
                        biggest_keypoint = max(feature1_size, feature2_size)
                        smallest_keypoint = min(feature1_size, feature2_size)
                        relative_scale_differnce = abs(1 - biggest_keypoint/smallest_keypoint)
                        if relative_scale_differnce > RELATIVE_SCALE_DIFFERENCE_THRESHOLD:
                            continue
                        
                        feature1_angle = feature1.keypoint.angle
                        feature2_angle = feature2.get_angle_after_homography(homography)
                        # Calculate the circular angle distance
                        angle_difference = abs((feature1_angle - feature2_angle + 180) % 360 - 180)
                        if angle_difference > ANGLE_THRESHOLD:
                            continue

                        num_possible_correct_matches += 1

                    repeatability = num_possible_correct_matches/len(reference_features)

                    nums_possible_correct_matches.append(num_possible_correct_matches)
                    repeatabilities.append(repeatability)

                set_nums_possible_correct_matches.append(nums_possible_correct_matches)
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
                        choice_pool = [] #(sequence index, image index)
                        for choice_sequence_index in range(len(image_feature_set)):
                            if choice_sequence_index == sequence_index: # Do not take images from this sequence
                                continue
                            choice_pool.extend([(sequence_index, choice_image_index)
                                                for choice_image_index
                                                in range(len(image_feature_set[choice_sequence_index]))])
                        
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
            if DEBUG == "all" or DEBUG == "retrieval":
                for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating retrieval results")):

                    retrieval_match_set = MatchSet()
                    retrieval_match_sets.append(retrieval_match_set)
                    reference_features = image_feature_sequence.reference_image.copy()
                    
                    for reference_feature in reference_features:
                        
                        correct_features = reference_feature.get_all_valid_matches()[:5]
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
            set_nums_possible_correct_matches = np.array(set_nums_possible_correct_matches)
            set_nums_possible_correct_matches.flatten()

            set_repeatabilities = np.array(set_repeatabilities)
            set_repeatabilities.flatten()

            total_possible_correct_matches = sum(
                num_correct_matches
                for num_correct_sequence_matches in set_nums_possible_correct_matches
                for num_correct_matches in num_correct_sequence_matches
            )


            results = {
                "combination": feature_extractor_key,
                "speed": speed,
                "cm_total: mean" : np.mean(set_nums_possible_correct_matches),
                "cm_total: std" : np.std(set_nums_possible_correct_matches),
                "rep_total: mean" : np.mean(set_repeatabilities),
                "rep_total: std" : np.std(set_repeatabilities),
                "total num matches" : sum(len(match_set) for match_set in matching_match_sets),
                "num possible correct matches" : total_possible_correct_matches,
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
            
        except Exception as e:
            error_message = traceback.format_exc()
            with open("failed_combinations.txt", "a") as f:
                f.write(f"{feature_extractor_key}\n")
                f.write(f"{error_message}\n")
                f.write("\n")
            
    ################################################ STORE RESULTS ##############################################################

    df = pd.DataFrame(all_results)
    df.to_csv("output.csv", index = False)