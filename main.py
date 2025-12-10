import os
os.environ["BEARTYPE_IS_BEING_TYPE_CHECKED"] = "0" # Enable or disable beartype

from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.pipeline import *
from benchmark.matching import MatchSet, MatchRankingProperty, greedy_maximum_bipartite_matching_descriptor_distance
from benchmark.utils import load_HPSequences, compare_rankings_and_visualize_across_sets
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
import traceback
import warnings
from config import *

## Load dataset.    
dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release") 

####################################### SETUP BENCHMARK HERE #############################################################


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
SIFT_HIGH_SIG = cv2.SIFT_create(sigma = 6)
SIMPLEBLOB = cv2.SimpleBlobDetector_create()
BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
DAISY = cv2.xfeatures2d.DAISY_create()
FREAK = cv2.xfeatures2d.FREAK_create()
HARRISLAPLACE = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
LATCH = cv2.xfeatures2d.LATCH.create()
LUCID = cv2.xfeatures2d.LUCID.create()
MSD = cv2.xfeatures2d.MSDDetector_create()
STARDETECTOR = cv2.xfeatures2d.StarDetector_create()


features2d = {
    "AGAST" : AGAST,
    "AKAZE" : AKAZE,
    #"BRISK" : BRISK,
    "FAST" : FAST,
    "GFTT" : GFTT,
    #"KAZE" : KAZE,
    #"MSER" : MSER,
    "ORB" : ORB,
    "SIFT" : SIFT,
    "SIFT_HIGH_SIG" : SIFT_HIGH_SIG,
    # "SIMPLEBLOB" : SIMPLEBLOB,
    #"BRIEF" : BRIEF,
    #"DAISY" : DAISY,
    #"FREAK" : FREAK,
    # "HARRISLAPLACE" : HARRISLAPLACE,
    # "LATCH" : LATCH,
    # "LUCID" : LUCID,
    # "MSD" : MSD,
    #"STARDETECTOR" : STARDETECTOR 
}

test_combinations: dict[str, FeatureExtractor] = {} # {Printable name of feature extraction method: feature extractor wrapper}
for detector_key in features2d.keys():
    for descriptor_key in features2d.keys():
        distance_type = ""
        if descriptor_key in ["BRISK", "ORB", "AKAZE"]: 
            distance_type = cv2.NORM_HAMMING
        else: 
            distance_type = cv2.NORM_L2
        test_combinations[detector_key + "+" + descriptor_key] = FeatureExtractor.from_opencv(features2d[detector_key].detect, features2d[descriptor_key].compute, distance_type)

SKIP = []

## Setup matching approach
distance_match_rank_property = MatchRankingProperty("distance", False)
average_response_match_rank_property = MatchRankingProperty("average_response", True)
distinctiveness_match_rank_property = MatchRankingProperty("distinctiveness", True)
match_properties = [distance_match_rank_property, average_response_match_rank_property, distinctiveness_match_rank_property]

matching_approach = greedy_maximum_bipartite_matching_descriptor_distance

#############################################################################################################################
all_results = []

warnings.filterwarnings("once", category=UserWarning)
image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)

for feature_extractor_key in tqdm(test_combinations.keys(), leave=False, desc="Calculating for all combinations"):
    print(f"Calculating for {feature_extractor_key}")   
    
    try:
        feature_extractor: FeatureExtractor = test_combinations[feature_extractor_key]

        speed = 0
        if "speedtest" not in SKIP:
            speed = speed_test(feature_extractor, dataset_image_sequences)

        find_all_features_for_dataset(feature_extractor, dataset_image_sequences, image_feature_set, MAX_FEATURES)
        set_numbers_of_possible_correct_matches, set_repeatabilities =  calculate_valid_matches(image_feature_set, dataset_homography_sequence, FEATURE_OVERLAP_THRESHOLD)

        if "matching" not in SKIP:
            matching_match_sets: list[MatchSet] = calculate_matching_evaluation(feature_extractor, image_feature_set, matching_approach)
        else:
            matching_match_sets: list[MatchSet] = [MatchSet()]
        
        if "verification" not in SKIP:
            verification_match_sets: list [MatchSet] = calculate_verification_evaluation(feature_extractor, image_feature_set, VERIFICATION_CORRECT_TO_RANDOM_RATIO, matching_approach)
        else:
            verification_match_sets: list [MatchSet] = [MatchSet()]
        
        if "retrieval" not in SKIP:
            retrieval_match_sets : list[MatchSet] = calculate_retrieval_evaluation(feature_extractor, image_feature_set, RETRIEVAL_CORRECT_TO_RANDOM_RATIO, MAX_NUM_RETRIEVAL_FEATURES, matching_approach)
        else:
            retrieval_match_sets : list [MatchSet] = [MatchSet()]        

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
            "combination": f"{feature_extractor_key}",
            "speed": speed,
            "cm_total: mean" : np.mean(set_numbers_of_possible_correct_matches),
            "cm_total: std" : np.std(set_numbers_of_possible_correct_matches),
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
            mAP = np.average([match_set.get_average_precision_score(match_ranking_property, True) for match_set in retrieval_match_sets])
            results[f"Retrieval {match_ranking_property.name} mAP"] = mAP

        spearman_rank_correlation_distance_distinctiveness = compare_rankings_and_visualize_across_sets(matching_match_sets, match_properties)[0][2]
        results["distance-distinctiveness correlation"] = spearman_rank_correlation_distance_distinctiveness

        all_results.append(results)

        ################################################ STORE RESULTS AFTER EACH COMBINATION ###################################
        for metric, result in results.items():
            print(metric, result)
        df = pd.DataFrame(all_results)
        df.to_csv("output.csv", index = False)

    except Exception as e:
        error_message = traceback.format_exc()
        with open("failed_combinations.txt", "a") as f:
            f.write(f"{feature_extractor_key}\n")
            f.write(f"{error_message}\n")
            f.write("\n")


        