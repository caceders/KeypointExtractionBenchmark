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
SIFT_SIGMA_5 = cv2.SIFT_create(sigma = 5)
SIFT_SIGMA_10 = cv2.SIFT_create(sigma = 10)
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
    #"AGAST" : AGAST,
    #"AKAZE" : AKAZE,
    #"BRISK" : BRISK,
    #"FAST" : FAST,
    #"GFTT" : GFTT,
    #"KAZE" : KAZE,
    # "MSER" : MSER,
    "ORB" : ORB,
    #"SIFT" : SIFT,
    #"SIFT_SIGMA_5" : SIFT_SIGMA_5,
    #"SIFT_SIGMA_10" : SIFT_SIGMA_10,
    # "SIMPLEBLOB" : SIMPLEBLOB,
    #"BRIEF" : BRIEF,
    # "DAISY" : DAISY,
    # "FREAK" : FREAK,
    # "HARRISLAPLACE" : HARRISLAPLACE,
    # "LATCH" : LATCH,
    # # "LUCID" : LUCID,
    # "MSD" : MSD,
    # "STARDETECTOR" : STARDETECTOR 
}

test_combinations: dict[str, FeatureExtractor] = {} # {Printable name of feature extraction method: feature extractor wrapper}
sigmas = [0.8,1.6,2.4,3.2,4,4.8,6.4,8,10]
for sigma in sigmas:
    SIFT = cv2.SIFT_create(sigma = sigma)
    #for detector_key in features2d.keys():
        #for descriptor_key in features2d.keys():
    #distance_type = ""
    #if descriptor_key in ["BRISK", "ORB", "AKAZE", "BRIEF", "FREAK", "LATCH"]: 
    #    distance_type = cv2.NORM_HAMMING
    #else: 
    distance_type = cv2.NORM_L2
    test_combinations["SIFT" + "+" +  str(sigma)] = FeatureExtractor.from_opencv(SIFT.detect, SIFT.compute, distance_type)

SKIP = ["find_features", "verification", "matching", "retrieval"]

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

        if "find_features" not in SKIP:
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

        if "matching" or "find_features" in SKIP:
            results = {"combination": f"{feature_extractor_key}",
                       "speed": speed
                       }
        else:
            ## Store results
            # Flatten matching matches once
            all_matches = [m for s in matching_match_sets for m in s]
            num_matches = len(all_matches)

            # Pre-extract fields (vectorized)
            is_correct = np.fromiter((m.is_correct for m in all_matches), bool, count=num_matches)
            match_rank = np.fromiter((m.match_properties["match rank"] for m in all_matches), int, count=num_matches)
            distances = np.fromiter((m.match_properties["distance"] for m in all_matches), float, count=num_matches)
            distinctiveness = np.fromiter((m.match_properties["distinctiveness"] for m in all_matches), float, count=num_matches)
            sizes = np.fromiter(((m.reference_feature.keypoint.size + m.related_feature.keypoint.size) / 2 for m in all_matches), float, count=num_matches)
            responses = np.fromiter(((m.reference_feature.keypoint.response + m.related_feature.keypoint.response) / 2 for m in all_matches), float, count=num_matches)

            # Precompute masks
            correct_mask = is_correct
            incorrect_mask = ~is_correct

            # Total possible matches (vectorized)
            set_numbers_of_possible_correct_matches = np.array(set_numbers_of_possible_correct_matches, dtype=object)
            total_possible_correct_matches = set_numbers_of_possible_correct_matches.sum()

            # Totals
            total_correct_matches = is_correct.sum()
            ratio_correct = total_correct_matches / num_matches
            ratio_possible_found = total_correct_matches / total_possible_correct_matches

            # --- Rank-based stats ---
            max_rank = NUM_BEST_MATCHES
            match_rank_totals = np.bincount(match_rank, minlength=max_rank)
            match_rank_correct = np.bincount(match_rank[correct_mask], minlength=max_rank)

            match_rank_ratios = np.divide(
                match_rank_correct.astype(float),
                match_rank_totals.astype(float),
                out=np.zeros_like(match_rank_correct, dtype=float),
                where=match_rank_totals != 0
            )

            # --- Size, distance, response, distinctiveness stats ---
            def safe_mean(x):
                return float(np.mean(x)) if len(x) else 0.0

            avg_size = sizes.mean()
            std_size = sizes.std()
            min_size = sizes.min()
            max_size = sizes.max()
            unique_sizes_count = len(np.unique(sizes))

            # Correct-only subsets
            sizes_correct = sizes[correct_mask]
            distances_correct = distances[correct_mask]
            responses_correct = responses[correct_mask]
            distinctiveness_correct = distinctiveness[correct_mask]

            # Additional requested metrics
            total_num_features = sum(len(image) for sequence in image_feature_set for image in sequence)

            # Per-image metrics
            correct_per_sequence = np.array([sum(m.is_correct for m in s) for s in matching_match_sets])
            avg_correct_per_sequence = correct_per_sequence.mean()
            std_correct_per_sequence = correct_per_sequence.std()

            avg_size_correct = safe_mean(sizes_correct)
            ratio_size_correct = avg_size_correct / avg_size if avg_size != 0 else 0

            norm_std_size = std_size / avg_size if avg_size != 0 else 0

            avg_dist = distances.mean()
            avg_dist_correct = safe_mean(distances_correct)
            ratio_dist_correct = avg_dist_correct / avg_dist if avg_dist != 0 else 0

            avg_resp = responses.mean()
            avg_resp_correct = safe_mean(responses_correct)
            ratio_resp_correct = avg_resp_correct / avg_resp if avg_resp != 0 else 0

            avg_distinct = distinctiveness.mean()
            avg_distinct_correct = safe_mean(distinctiveness_correct)
            ratio_distinct_correct = avg_distinct_correct / avg_distinct if avg_distinct != 0 else 0

            # Rank stats: all + correct
            avg_rank_all = match_rank.mean()
            std_rank_all = match_rank.std()
            avg_rank_correct = match_rank[correct_mask].mean() if correct_mask.any() else 0
            std_rank_correct = match_rank[correct_mask].std() if correct_mask.any() else 0

            outside_num_best_matches_all = np.mean(match_rank > NUM_BEST_MATCHES//2)
            outside_num_best_matches_correct = np.mean(match_rank[correct_mask] > NUM_BEST_MATCHES//2) if correct_mask.any() else 0


            # ========================
            # STORE RESULTS
            # ========================

        
            results = {
                "combination": f"{feature_extractor_key}",
                "speed": speed,
                "repeatability mean": np.mean(set_repeatabilities),
                "repeatability std": np.std(set_repeatabilities),

                "total num matches": num_matches,
                "number possible correct matches": total_possible_correct_matches,
                "total correct matches": total_correct_matches,
                "ratio correct/total matches": ratio_correct,
                "ratio correct/possible correct matches": ratio_possible_found,

                # Size metrics
                "size mean": avg_size,
                "size std": std_size,
                "size normalized std": norm_std_size,
                "size min": min_size,
                "size max": max_size,
                "size unique count": unique_sizes_count,
                "size correct: avg": avg_size_correct,
                "size correct/all ratio": ratio_size_correct,

                "total num keypoints": total_num_features,
                "correct matches per sequence: avg": avg_correct_per_sequence,
                "correct matches per sequence: std": std_correct_per_sequence,

                "distance correct/all ratio": ratio_dist_correct,

                "response correct/all ratio": ratio_resp_correct,

                "distinctiveness all: avg": avg_distinct,
                "distinctiveness correct: avg": avg_distinct_correct,
                "distinctiveness correct/all ratio": ratio_distinct_correct,

                # Rank metrics
                "match rank: avg": avg_rank_all,
                "match rank: std": std_rank_all,
                "match rank correct: avg": avg_rank_correct,
                "match rank correct: std": std_rank_correct,
                f"ratio rank >{NUM_BEST_MATCHES//2} / all": outside_num_best_matches_all,
                f"ratio rank >{NUM_BEST_MATCHES//2} correct / rank >{NUM_BEST_MATCHES//2}": outside_num_best_matches_correct,
            }

            # Results from matching
            for match_rank_property in match_properties:
                mAP = np.average([match_set.get_average_precision_score(match_rank_property) for match_set in matching_match_sets])
                results[f"Matching {match_rank_property.name} mAP"] =  mAP

            # Results from verification
            if "verification" not in SKIP:
                total_verification_set = MatchSet()
                for match_set in verification_match_sets:
                    for match in match_set:
                        total_verification_set.add_match(match)
                        
                for match_ranking_property in match_properties:
                    AP = total_verification_set.get_average_precision_score(match_ranking_property)
                    results[f"Verification {match_ranking_property.name} AP"] = AP

            # Results from retrieval
            if "retrieval" not in SKIP:
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
        df.to_csv("output_sigma_speed.csv", index = False)

    except Exception as e:
        error_message = traceback.format_exc()
        with open("failed_combinations.txt", "a") as f:
            f.write(f"{feature_extractor_key}\n")
            f.write(f"{error_message}\n")
            f.write("\n")
