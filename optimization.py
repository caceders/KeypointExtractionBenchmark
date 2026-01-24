from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.pipeline import *
from benchmark.matching import MatchSet, MatchRankingProperty, greedy_maximum_bipartite_matching_descriptor_distance
from benchmark.utils import load_HPSequences
import cv2
import numpy as np
import traceback
from config import *
import optuna
import os
from tqdm import tqdm
from timeit import default_timer as timer


detector_names = ["BRISK", "ORB", "FAST", "SIFT", "AKAZE", "GFTT"]
descriptor_names = ["BRISK", "ORB", "SIFT", "AKAZE", "BRIEF"]

skip = ["BRISK + BRISK", "BRISK + ORB", "BRISK + SIFT", "BRISK + AKAZE", "AKAZE + BIRSK", "AKAZE + ORB", "AKAZE + SIFT", "AKAZE + BRIEF"]
do = ["GFTT + BRIEF"]
dataset_image_sequences, dataset_homography_sequence = load_HPSequences(
    r"hpatches-sequences-release"
)

distance_match_rank_property = MatchRankingProperty("distance", False)
matching_approach = greedy_maximum_bipartite_matching_descriptor_distance

active_detector_name: str
active_descriptor_name: str
active_detector: cv2.Feature2D
active_descriptor: cv2.Feature2D

failed = False
def objective(trial: optuna.Trial):
    try:
        global active_detector_name, active_descriptor_name
        global active_detector, active_descriptor

        # ---------------- BRISK ----------------
        if active_detector_name == "BRISK" or active_descriptor_name == "BRISK":
            BRISK_thresh = trial.suggest_int("BRISK_thresh", 0, 100)
            BRISK_octaves = trial.suggest_int("BRISK_octaves", 0, 10)
            BRISK_pattern_scale = trial.suggest_float("BRISK_pattern_scale", 0, 5)

            BRISK = cv2.BRISK_create(
                thresh=BRISK_thresh,
                octaves=BRISK_octaves,
                patternScale=BRISK_pattern_scale
            )
            if active_detector_name == "BRISK":
                active_detector = BRISK
            if active_descriptor_name == "BRISK":
                active_descriptor = BRISK

        # ---------------- ORB ----------------
        if active_detector_name == "ORB" or active_descriptor_name == "ORB":
            ORB_scaleFactor = trial.suggest_float("ORB_scaleFactor", 0, 6)
            ORB_nlevels = trial.suggest_int("ORB_nlevels", 0, 40)
            ORB_patchSize = trial.suggest_int("ORB_patchSize", 0, 100)
            ORB_edgeThreshold = trial.suggest_int("ORB_edgeThreshold", 0, 100)
            ORB_WTA_K = trial.suggest_categorical("ORB_WTA_K", [2, 3, 4])
            ORB_fastThreshold = trial.suggest_int("ORB_fastThreshold", 0, 120)

            ORB = cv2.ORB_create(
                scaleFactor=ORB_scaleFactor,
                nlevels=ORB_nlevels,
                edgeThreshold=ORB_edgeThreshold,
                WTA_K=ORB_WTA_K,
                patchSize=ORB_patchSize,
                fastThreshold=ORB_fastThreshold
            )
            if active_detector_name == "ORB":
                active_detector = ORB
            if active_descriptor_name == "ORB":
                active_descriptor = ORB

        # ---------------- FAST ----------------
        if active_detector_name == "FAST":
            FAST_threshold = trial.suggest_int("FAST_threshold", 0, 100)
            FAST_type = trial.suggest_categorical(
                "FAST_type",
                [
                    cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
                    cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
                    cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
                ]
            )
            active_detector = cv2.FastFeatureDetector_create(
                threshold=FAST_threshold,
                type=FAST_type
            )


        # ---------------- SIFT ----------------
        if active_detector_name == "SIFT" or active_descriptor_name == "SIFT":
            SIFT_nOctaveLayers = trial.suggest_int("SIFT_nOctaveLayers", 0, 30)
            SIFT_contrastThreshold = trial.suggest_float("SIFT_contrastThreshold", 0, 0.4)
            SIFT_edgeThreshold = trial.suggest_float("SIFT_edgeThreshold", 0, 100)
            SIFT_sigma = trial.suggest_float("SIFT_sigma", 0, 10)

            SIFT = cv2.SIFT_create(
                nOctaveLayers=SIFT_nOctaveLayers,
                contrastThreshold=SIFT_contrastThreshold,
                edgeThreshold=SIFT_edgeThreshold,
                sigma=SIFT_sigma
            )
            if active_detector_name == "SIFT":
                active_detector = SIFT
            if active_descriptor_name == "SIFT":
                active_descriptor = SIFT

        # ---------------- BRIEF ----------------
        if active_descriptor_name == "BRIEF":
            active_descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # ---------------- AKAZE ----------------
        if active_detector_name == "AKAZE" or active_descriptor_name =="AKAZE":

            AKAZE_descriptor_channels = trial.suggest_int("AKAZE_descriptor_channels", 0, 3)
            AKAZE_threshold = trial.suggest_float("AKAZE_threshold", 0, 0.015)
            AKAZE_nOctaves = trial.suggest_int("AKAZE_nOctaves", 0, 40)
            AKAZE_nOctaveLayers = trial.suggest_int("AKAZE_nOctaveLayers", 0, 40)
            AKAZE_diffusivity = trial.suggest_int("AKAZE_diffusivity", 0, 3)

            AKAZE = cv2.AKAZE_create(
            descriptor_channels = AKAZE_descriptor_channels,
            threshold = AKAZE_threshold,
            nOctaves = AKAZE_nOctaves,
            nOctaveLayers = AKAZE_nOctaveLayers,
            diffusivity = AKAZE_diffusivity,
            )

            if active_detector_name == "AKAZE":
                active_detector = AKAZE
            if active_descriptor_name == "AKAZE":
                active_descriptor = AKAZE

        # ---------------- GFTT ----------------
        if active_detector_name == "GFTT":
            GFTT_qualityLevel = trial.suggest_float("GFTT_qualityLevel", 0, 0.1)
            GFTT_minDistance = trial.suggest_float("GFTT_minDistance", 0, 10)
            GFTT_blockSize = trial.suggest_int("GFTT_blockSize", 0, 30)
            GFTT_k = trial.suggest_float("GFTT_k", 0, 0.4)

            active_detector = cv2.GFTTDetector_create(
                qualityLevel=GFTT_qualityLevel,
                minDistance=GFTT_minDistance,
                blockSize=GFTT_blockSize,
                k=GFTT_k
            )

        distance_type = cv2.NORM_L2 if active_descriptor_name == "SIFT" else cv2.NORM_HAMMING

        feature_extractor = FeatureExtractor.from_opencv(
            active_detector.detect,
            active_descriptor.compute,
            distance_type
        )

        image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
        find_all_features_for_dataset(
            feature_extractor,
            dataset_image_sequences,
            image_feature_set,
            MAX_FEATURES
        )

        _, _ = calculate_valid_matches(
            image_feature_set,
            dataset_homography_sequence,
            FEATURE_OVERLAP_THRESHOLD
        )

        matching_match_sets = calculate_matching_evaluation(
            feature_extractor,
            image_feature_set,
            matching_approach
        )

        return np.mean([
            ms.get_average_precision_score(distance_match_rank_property)
            for ms in matching_match_sets
        ])

    except Exception:
        with open("failed_combinations.txt", "a") as f:
            f.write(f"{active_detector_name} + {active_descriptor_name}\n")
            f.write(traceback.format_exc() + "\n\n")
        global failed
        failed = True
        return -1


if __name__ == "__main__":
    TARGET_TRIALS = 100
    TIMEOUT = 60*60*6

    for detector_name in tqdm(detector_names, leave=False, desc="All combinations"):
        for descriptor_name in tqdm(descriptor_names, leave=False, desc = f"{detector_name}"):
            if len(do) != 0:
                if not f"{detector_name} + {descriptor_name}" in do:
                    continue
            elif f"{detector_name} + {descriptor_name}" in skip:
                continue
            active_detector_name = detector_name
            active_descriptor_name = descriptor_name

            study = optuna.create_study(
                study_name=f"{detector_name} + {descriptor_name}",
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=30,
                    multivariate=True,
                    group=True),
                storage=f"sqlite:///{detector_name}-{descriptor_name}.db",
                load_if_exists=True
            )
            
            # Force check default parameters
            DEFAULT_PARAMS =  {}
            if len(study.trials) == 0:
                if detector_name == "BRISK" or descriptor_name == "BRISK":
                    DEFAULT_PARAMS["BRISK_thresh"] = 30
                    DEFAULT_PARAMS["BRISK_octaves"] = 3
                    DEFAULT_PARAMS["BRISK_pattern_scale"] = 1

                if detector_name == "ORB" or descriptor_name == "ORB":
                    DEFAULT_PARAMS["ORB_scaleFactor"] = 1.2
                    DEFAULT_PARAMS["ORB_nlevels"] = 8
                    DEFAULT_PARAMS["ORB_patchSize"] = 31
                    DEFAULT_PARAMS["ORB_edgeThreshold"] = 31
                    DEFAULT_PARAMS["ORB_WTA_K"] = 2
                    DEFAULT_PARAMS["ORB_fastThreshold"] = 20

                if detector_name == "SIFT" or descriptor_name == "SIFT":
                    DEFAULT_PARAMS["SIFT_nOctaveLayers"] = 3
                    DEFAULT_PARAMS["SIFT_contrastThreshold"] = 0.04
                    DEFAULT_PARAMS["SIFT_edgeThreshold"] = 10
                    DEFAULT_PARAMS["SIFT_sigma"] = 1.6


                if detector_name == "GFTT" or descriptor_name == "GFTT":
                    DEFAULT_PARAMS["GFTT_qualityLevel"] = 0.01
                    DEFAULT_PARAMS["GFTT_minDistance"] = 1
                    DEFAULT_PARAMS["GFTT_blockSize"] = 3
                    DEFAULT_PARAMS["GFTT_k"] = 0.04

                if detector_name == "AKAZE" or descriptor_name == "AKAZE":
                    DEFAULT_PARAMS["AKAZE_descriptor_channels"] = 3
                    DEFAULT_PARAMS["AKAZE_threshold"] = 0.001
                    DEFAULT_PARAMS["AKAZE_nOctaves"] = 4
                    DEFAULT_PARAMS["AKAZE_nOctaveLayers"] = 4
                    DEFAULT_PARAMS["AKAZE_diffusivity"] = 1

            if len(DEFAULT_PARAMS) > 0:
                study.enqueue_trial(DEFAULT_PARAMS)

            start = timer()
            correct = 0
            while correct < TARGET_TRIALS and (timer() - start < TIMEOUT):
                study.optimize(objective, n_trials=1, catch=(Exception,))
                if not failed:
                    correct += 1
                else:
                    failed = False
                print(f"Valid trials: {correct}")
