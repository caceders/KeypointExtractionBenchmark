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
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import os

DB_PATH = "optuna_GFTT_SIFT.db"
CSV_PATH = "history_GFTT_SIFT.csv"

## Load dataset.    
dataset_image_sequences, dataset_homography_sequence = load_HPSequences(r"hpatches-sequences-release") 

## Setup matching approach
distance_match_rank_property = MatchRankingProperty("distance", False)
average_response_match_rank_property = MatchRankingProperty("average_response", True)
distinctiveness_match_rank_property = MatchRankingProperty("distinctiveness", True)
match_properties = [distance_match_rank_property, average_response_match_rank_property, distinctiveness_match_rank_property]

matching_approach = greedy_maximum_bipartite_matching_descriptor_distance

# def simulation(sift_nOctaveLayers, sift_contrastThreshold, sift_edgeThreshold, sift_sigma, akaze_descriptor_size, akaze_descriptor_channels, akaze_threshold, akaze_nOctaves, akaze_nOctaveLayers):
#     AKAZE = cv2.AKAZE_create(descriptor_size=akaze_descriptor_size, descriptor_channels=akaze_descriptor_channels, threshold=akaze_threshold, nOctaves=akaze_nOctaves, nOctaveLayers=akaze_nOctaveLayers)
#     SIFT = cv2.SIFT_create(nOctaveLayers=sift_nOctaveLayers, contrastThreshold=sift_contrastThreshold, edgeThreshold=sift_edgeThreshold, sigma=sift_sigma)
#     feature_extractor=FeatureExtractor.from_opencv(AKAZE.detect, SIFT.compute, cv2.NORM_L2)
#     image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
#     find_all_features_for_dataset(feature_extractor, dataset_image_sequences, image_feature_set, MAX_FEATURES)
#     _, _ =  calculate_valid_matches(image_feature_set, dataset_homography_sequence, FEATURE_OVERLAP_THRESHOLD)
#     matching_match_sets: list[MatchSet] = calculate_matching_evaluation(feature_extractor, image_feature_set, matching_approach)

#     score = np.average([match_set.get_average_precision_score(distance_match_rank_property) for match_set in matching_match_sets])

#     return score

def simulation(sift_nOctaveLayers, sift_contrastThreshold, sift_edgeThreshold, sift_sigma,GFTT_qualityLevel, GFTT_minDistance, GFTT_blockSize, GFTT_k):
    GFTT = cv2.GFTTDetector_create(qualityLevel = GFTT_qualityLevel, minDistance = GFTT_minDistance, blockSize = GFTT_blockSize, k = GFTT_k)
    SIFT = cv2.SIFT_create(nOctaveLayers=sift_nOctaveLayers, contrastThreshold=sift_contrastThreshold, edgeThreshold=sift_edgeThreshold, sigma=sift_sigma)
    feature_extractor=FeatureExtractor.from_opencv(GFTT.detect, SIFT.compute, cv2.NORM_L2)

    # BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # feature_extractor=FeatureExtractor.from_opencv(GFTT.detect, BRIEF.compute, cv2.NORM_HAMMING)
    image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
    find_all_features_for_dataset(feature_extractor, dataset_image_sequences, image_feature_set, MAX_FEATURES)
    _, _ =  calculate_valid_matches(image_feature_set, dataset_homography_sequence, FEATURE_OVERLAP_THRESHOLD)
    matching_match_sets: list[MatchSet] = calculate_matching_evaluation(feature_extractor, image_feature_set, matching_approach)

    score = np.average([match_set.get_average_precision_score(distance_match_rank_property) for match_set in matching_match_sets])

    return score

## Optimize
def objective(trial: optuna.Trial):
    try:
        sift_nOctaveLayers = trial.suggest_int("sift_nOctaveLayers", 1, 7)
        sift_contrastThreshold = trial.suggest_float("sift_contrastThreshold", 0.01, 0.1)
        sift_edgeThreshold = trial.suggest_float("sift_edgeThreshold", 5, 20)
        sift_sigma = trial.suggest_float("sift_sigma", 0.5, 5)
        # akaze_descriptor_size = trial.suggest_int("akaze_descriptor_size", 0, 10)
        # akaze_descriptor_channels = trial.suggest_int("akaze_descriptor_channels", 1, 3)
        # akaze_threshold = trial.suggest_float("akaze_threshold", 0.0001, 0.1)
        # akaze_nOctaves = trial.suggest_int("akaze_nOctaves", 1, 10)
        # akaze_nOctaveLayers = trial.suggest_int("akaze_nOctaveLayers", 1, 10)

        # value = simulation(sift_nOctaveLayers, sift_contrastThreshold, sift_edgeThreshold, sift_sigma, akaze_descriptor_size, akaze_descriptor_channels, akaze_threshold, akaze_nOctaves, akaze_nOctaveLayers)
    
        GFTT_qualityLevel = trial.suggest_float("GFTT_qualityLevel", 0.001, 1)
        GFTT_minDistance = trial.suggest_float("GFTT_minDistance", 0.1, 10)
        GFTT_blockSize  = trial.suggest_int("GFTT_blockSize", 1, 10)
        GFTT_k = trial.suggest_float("GFTT_k", 0.004, 0.4)

        value = simulation(sift_nOctaveLayers, sift_contrastThreshold, sift_edgeThreshold, sift_sigma, GFTT_qualityLevel, GFTT_minDistance, GFTT_blockSize, GFTT_k)

    except:
        error_message = traceback.format_exc()
        with open("errors.txt", "a") as f:
            f.write(f"{error_message}\n")
            f.write("\n")
        raise optuna.TrialPruned()
    # append to CSV
    row = {
        "trial": trial.number,
        "GFTT_qualityLevel" : GFTT_qualityLevel,
        "GFTT_minDistance" : GFTT_minDistance,
        "GFTT_blockSize" : GFTT_blockSize,
        "GFTT_k" : GFTT_k,
        "value": value,
    }

    write_header = not os.path.exists(CSV_PATH)
    pd.DataFrame([row]).to_csv(
        CSV_PATH, mode="a", header=write_header, index=False
    )

    return value

if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=500,
        multivariate=True,
        constant_liar=True,
        group=True,
        seed=42,
    )

    study = optuna.create_study(
        study_name="GFTT_SIFT",
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True,
    )

    study.optimize(objective, timeout=60*60*24)

    print("Best parameters:")
    print(study.best_params)
    print("Best value:", study.best_value)

    # ---- plot history ----
    df = pd.read_csv(CSV_PATH)

    plt.figure()
    plt.plot(df["trial"], df["value"], marker="o")
    plt.xlabel("Trial")
    plt.ylabel("Objective value")
    plt.tight_layout()
    plt.show()