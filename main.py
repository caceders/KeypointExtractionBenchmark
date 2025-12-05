from benchmark.feature import Feature
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.matching import MatchSet, MatchRankingProperty, greedy_maximum_bipartite_matching_homographic_distance, greedy_maximum_bipartite_matching_descriptor_distance, greedy_maximum_bipartite_matching
from benchmark.utils import load_HPSequences
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import random
import traceback
import warnings

################################################ CONFIGURATIONS #######################################################
MAX_FEATURES = 100
USE_OVERLAP = True
FEATURE_OVERLAP_THRESHOLD = 0.3
ALTERNATIVE_DISTANCE_THRESHOLD = 10
VERIFICATION_CORRECT_TO_RANDOM_RATIO = 5
RETRIEVAL_CORRECT_TO_RANDOM_RATIO = 100
MAX_NUM_RETRIEVAL_FEATURES = 5
USE_MEASUREMENT_AREA_NORMALISATION = False
#######################################################################################################################

if __name__ == "__main__":

    ####################################### SETUP TESTBENCH HERE #############################################################

    SKIP = ["speedtest"]

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
    
    scales = [1]
    #scales = [0.25,0.5,1,2]
    #sigmas = [1,2,4,7,10,15,25,35,50]
    # sigmas = [1,2,4,7,10,15,25,35]
    #sigmas = [1,1.6,2,3,5]
    sigmas = [1]
    # for sigma in sigmas:
    #     SIFT = cv2.SIFT_create(nfeatures = MAX_FEATURES, sigma = sigma, contrastThreshold = 0.02)
    #     test_combinations["SIFT" + str(sigma)] = FeatureExtractor.from_opencv(SIFT.detect, SIFT.compute, cv2.NORM_L2, USE_MEASUREMENT_AREA_NORMALISATION, 9, 9)

    for scale in scales:
        #test_combinations["SIFT " + str(scale)] = FeatureExtractor.from_opencv(SIFT.detect, SIFT.compute, cv2.NORM_L2, USE_MEASUREMENT_AREA_NORMALISATION, 9, 9)
        #test_combinations["SIFT2 " + str(scale)] = FeatureExtractor.from_opencv(SIFT2.detect, SIFT2.compute, cv2.NORM_L2, USE_MEASUREMENT_AREA_NORMALISATION, 9, 9)
        test_combinations["ORB " + str(scale)] = FeatureExtractor.from_opencv(ORB.detect, ORB.compute, cv2.NORM_HAMMING, USE_MEASUREMENT_AREA_NORMALISATION, 31, 31)

    ## Setup matching approach
    distance_match_rank_property = MatchRankingProperty("distance", False)
    average_response_match_rank_property = MatchRankingProperty("average_response", True)
    distinctiveness_match_rank_property = MatchRankingProperty("distinctiveness", True)
    match_properties = [distance_match_rank_property, average_response_match_rank_property, distinctiveness_match_rank_property]

    matching_approach = greedy_maximum_bipartite_matching_descriptor_distance

    #############################################################################################################################
    all_results = []

    warnings.filterwarnings("once", category=UserWarning)

    for feature_extractor_key_index, feature_extractor_key in enumerate(tqdm(test_combinations.keys(), desc = "Running tests")):
        feature_extractor: FeatureExtractor = test_combinations[feature_extractor_key]
        #try:
        ## Load dataset.    
        dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release")

        scale = scales[feature_extractor_key_index//len(test_combinations)]

        ## scale dataset.
        for sequence_index, sequence in enumerate(dataset_image_sequences):
            for image_index, image in enumerate(sequence):
                y, x = image.shape[:2]  # height, width
                new_x = int(round(x * scale))
                new_y = int(round(y * scale))
                dataset_image_sequences[sequence_index][image_index] = cv2.resize(image, (new_x, new_y), interpolation=cv2.INTER_CUBIC)


        num_sequences = len(dataset_image_sequences)
        num_related_images = len(dataset_image_sequences[0]) - 1
        image_feature_set = ImageFeatureSet(num_sequences, num_related_images)

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
                
                image_feature_set[sequence_index][image_index] = features


        ## Calculate valid matches, number of possible correct matches and repeatability
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
                    if USE_OVERLAP:
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
                        mask = (overlap_ref_frac >= FEATURE_OVERLAP_THRESHOLD) & (overlap_rel_frac >= FEATURE_OVERLAP_THRESHOLD)

                    else:
                        mask = (
                            (distances <= ALTERNATIVE_DISTANCE_THRESHOLD)
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
                #matches = greedy_maximum_bipartite_matching(reference_features, related_image_features, overlap_matrix_np)

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



        ## Calculate matching results.
        matching_match_sets: list[MatchSet] = []
        if "matching" not in SKIP:
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave=False, desc="Calculating matching results")):
                matching_match_set = MatchSet()
                matching_match_sets.append(matching_match_set)
                for related_image_index, related_image in enumerate(image_feature_sequence.related_images):

                    # Reference and related features.
                    reference_features = image_feature_sequence.reference_image.copy()
                    related_image_features = image_feature_sequence.related_image(related_image_index).copy()

                    matches = matching_approach(reference_features, related_image_features, feature_extractor.distance_type)
                    matching_match_set.add_match(matches)




        ## Calculate verification results.
        verification_match_sets: list [MatchSet] = []
        if "verification" not in SKIP:
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating verification results")):
                verification_match_set = MatchSet()
                verification_match_sets.append(verification_match_set)
                reference_features = image_feature_sequence.reference_image
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
                        related_image_features = image_feature_sequence.related_image(related_image_index)

                        match = matching_approach([reference_feature], related_image_features, feature_extractor.distance_type)
                        verification_match_set.add_match(match)
                    
                    # Pick random images
                    choice_pool = [(choice_sequence_index, choice_image_index)  #(sequence index, image index)
                                for choice_sequence_index, choice_sequence in enumerate(image_feature_set) 
                                if choice_sequence_index != sequence_index 
                                for choice_image_index in range(len(choice_sequence))] 
                    
                    chosen_random_images = random.sample(choice_pool, num_random_images)
                    
                    # Match for all random images
                    for random_sequence_index, random_image_index in chosen_random_images:
                        random_image_features = image_feature_set[random_sequence_index][random_image_index]
                        match = matching_approach([reference_feature], random_image_features, feature_extractor.distance_type)
                        verification_match_set.add_match(match)


        ## Calculate retrieval results.
        retrieval_match_sets : list[MatchSet] = []
        if "retrieval" not in SKIP:
            all_features = [feature 
                            for choice_image_feature_sequence in image_feature_set
                            for choice_image_features in choice_image_feature_sequence
                            for feature in choice_image_features]            

            all_features_array = np.array(all_features, dtype=object)
            for sequence_index, image_feature_sequence in enumerate(tqdm(image_feature_set, leave = False, desc = "Calculating retrieval results")):

                retrieval_match_set = MatchSet()
                retrieval_match_sets.append(retrieval_match_set)
                reference_features = image_feature_sequence.reference_image
                
                for reference_feature in reference_features:

                    # Choose MAX_NUM_RETRIEVAL_FEATURES correct features
                    correct_features = reference_feature.get_all_valid_matches()
                    if len(correct_features) > MAX_NUM_RETRIEVAL_FEATURES:
                        correct_features = random.sample(correct_features, MAX_NUM_RETRIEVAL_FEATURES)

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

        sizes = []
        responses = []
        for sequence in image_feature_set:
            for image in sequence:
                for feature in image:
                    sizes.append(feature.keypoint.size)
                    responses.append(feature.keypoint.response*sigmas[feature_extractor_key_index]**2)
        
        sizes = np.array(sizes)
        responses = np.array(responses)

        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        min_size = np.min(sizes)
        max_size = np.max(sizes)
        unique_sizes_count = len(set(sizes))
        avg_response = np.mean(responses)


        results = {
            "combination": feature_extractor_key,
            # "speed": speed,
            #"cm_total: mean" : np.mean(set_numbers_of_possible_correct_matches),
            #"cm_total: std" : np.std(set_numbers_of_possible_correct_matches),
            "rep_total: mean" : np.mean(set_repeatabilities),
            #"rep_total: std" : np.std(set_repeatabilities),
            "total features" : len(image_feature_set.get_features()),
            "total num matches" : sum(len(match_set) for match_set in matching_match_sets),
            "num possible correct matches" : total_possible_correct_matches,
            "total correct matches" : total_correct_matches,
            "size: mean": avg_size,
            #"size: std": std_size,
            #"size: min": min_size,
            "size: max": max_size,
            "size: unique count": unique_sizes_count,
            "response: avg": avg_response
        }

        for match_rank_property in match_properties:
            mAP = np.average([match_set.get_average_precision_score(match_rank_property) for match_set in matching_match_sets])
            results[f"Matching {match_rank_property.name} mAP"] =  mAP

        # Results from verification
        for match_ranking_property in match_properties:
            mAP = np.average([match_set.get_average_precision_score(match_ranking_property) for match_set in verification_match_sets])
            results[f"Verification {match_ranking_property.name} mAP"] = mAP

        # Results from retrieval
        if "retrieval" not in SKIP:
            for match_ranking_property in match_properties:
                mAP = np.average([match_set.get_average_precision_score(match_ranking_property) for match_set in retrieval_match_sets])
                results[f"Retrieval {match_ranking_property.name} mAP"] = mAP

        all_results.append(results)
                
        for key, value in results.items():
            print(f"{key}: {value}")

        # except Exception as e:
        #     error_message = traceback.format_exc()
        #     with open("failed_combinations.txt", "a") as f:
        #         f.write(f"{feature_extractor_key}\n")
        #         f.write(f"{error_message}\n")
        #         f.write("\n")

        
            
    ################################################ STORE RESULTS ##############################################################
    df = pd.DataFrame(all_results)
    df.to_csv("output.csv", index = False)
