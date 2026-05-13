from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.pipeline import *
from benchmark.matching import MatchSet, MatchRankingProperty, greedy_maximum_bipartite_matching_descriptor_distance
from benchmark.utils import load_HPSequences
import cv2
import numpy as np
import traceback
from trial_timer import TrialTimer, TrialTimeoutError
from config import *
import optuna
from tqdm import tqdm
from pathlib import Path

#########################################################
# ======================== CONFIG =======================
#########################################################

# "KEEM"  – maximize mean average precision on KEEM
# "KITTI" – minimize RPE translational RMSE on KITTI
OBJECTIVE = "KITTI"

methods = ["SIFT", "BRISK", "ORB", "AKAZE", "GFTT_SIFT"]
RESULTS_DIR = Path("optimization_results")
TIMEOUT = None  # max seconds per method (None = unlimited)
N_TRIALS = 500  # max new successful trials per method (None = unlimited)
ROUND_ROBIN = True  # True: cycle all methods N_TRIALS times; False: run each method N_TRIALS before moving on
TRIAL_TIMEOUT = 60 * 60 * 24  # max seconds per individual trial

#########################################################
# =================== OBJECTIVE SETUP ===================
#########################################################

if OBJECTIVE == "KITTI":
    from KITTI_main import evaluate_kitti

elif OBJECTIVE == "KEEM":
    dataset_image_sequences, dataset_homography_sequence = load_HPSequences(
        r"hpatches-sequences-release"
    )
    distance_match_rank_property = MatchRankingProperty("distance", False)
    matching_approach = greedy_maximum_bipartite_matching_descriptor_distance

#########################################################
# ==================== DEFAULT PARAMS ===================
#########################################################

DEFAULT_PARAMS = {
    "BRISK": {"BRISK_thresh": 30, "BRISK_octaves": 3, "BRISK_pattern_scale": 1},
    "ORB": {
        "ORB_scaleFactor": 1.2, "ORB_nlevels": 8, "ORB_patchSize": 31,
        "ORB_edgeThreshold": 31, "ORB_WTA_K": 2,
        "ORB_scoreType": cv2.ORB_HARRIS_SCORE, "ORB_fastThreshold": 20,
    },
    "SIFT": {
        "SIFT_nOctaveLayers": 3, "SIFT_contrastThreshold": 0.04,
        "SIFT_edgeThreshold": 10, "SIFT_sigma": 1.6,
    },
    "AKAZE": {
        "AKAZE_descriptor_channels": 3, "AKAZE_threshold": 0.001,
        "AKAZE_nOctaves": 4, "AKAZE_nOctaveLayers": 4, "AKAZE_diffusivity": 1,
    },
    "KAZE": {
        "KAZE_extended": False, "KAZE_upright": False, "KAZE_threshold": 0.001,
        "KAZE_nOctaves": 4, "KAZE_nOctaveLayers": 4, "KAZE_diffusivity": 1,
    },
    "SURF": {
        "SURF_hessianThreshold": 100, "SURF_nOctaves": 4,
        "SURF_nOctaveLayers": 3, "SURF_extended": False, "SURF_upright": False,
    },
    "FAST_SIFT": {
        "FAST_threshold": 10, "FAST_nonmaxSuppression": True, "FAST_type": 2,
        "SIFT_nOctaveLayers": 3, "SIFT_contrastThreshold": 0.04,
        "SIFT_edgeThreshold": 10, "SIFT_sigma": 1.6,
    },
    "GFTT_SIFT": {
        "GFTT_qualityLevel": 0.01, "GFTT_minDistance": 1.0,
        "GFTT_blockSize": 3, "GFTT_useHarrisDetector": False, "GFTT_k": 0.04,
        "SIFT_nOctaveLayers": 3, "SIFT_contrastThreshold": 0.04,
        "SIFT_edgeThreshold": 10, "SIFT_sigma": 1.6,
    },
    "MSER_SIFT": {
        "MSER_delta": 5, "MSER_min_area": 60, "MSER_max_area": 14400,
        "MSER_max_variation": 0.25, "MSER_min_diversity": 0.2,
        "SIFT_nOctaveLayers": 3, "SIFT_contrastThreshold": 0.04,
        "SIFT_edgeThreshold": 10, "SIFT_sigma": 1.6,
    },
    "ORB_DAISY": {
        "ORB_scaleFactor": 1.2, "ORB_nlevels": 8, "ORB_patchSize": 31,
        "ORB_edgeThreshold": 31, "ORB_scoreType": cv2.ORB_HARRIS_SCORE, "ORB_fastThreshold": 20,
        "DAISY_radius": 15.0, "DAISY_q_radius": 3, "DAISY_q_theta": 8, "DAISY_q_hist": 8,
        "DAISY_norm": 100, "DAISY_interpolation": True, "DAISY_use_orientation": False,
    },
    "ORB_FREAK": {
        "ORB_scaleFactor": 1.2, "ORB_nlevels": 8, "ORB_patchSize": 31,
        "ORB_edgeThreshold": 31, "ORB_scoreType": cv2.ORB_HARRIS_SCORE, "ORB_fastThreshold": 20,
        "FREAK_orientationNormalized": True, "FREAK_scaleNormalized": True,
        "FREAK_patternScale": 22.0, "FREAK_nOctaves": 4,
    },
}

#########################################################
# ==================== TRIAL OBJECTIVE ==================
#########################################################

active_method_name: str


def _build_sift_descriptor(trial: optuna.Trial) -> cv2.Feature2D:
    return cv2.SIFT_create(
        nOctaveLayers=trial.suggest_int("SIFT_nOctaveLayers", 1, 8),
        contrastThreshold=trial.suggest_float("SIFT_contrastThreshold", 0, 1.0),
        edgeThreshold=trial.suggest_float("SIFT_edgeThreshold", 0, 300),
        sigma=trial.suggest_float("SIFT_sigma", 0.1, 20),
    )


def _build_orb_detector(trial: optuna.Trial) -> cv2.Feature2D:
    return cv2.ORB_create(
        scaleFactor=trial.suggest_float("ORB_scaleFactor", 1.01, 4),
        nlevels=trial.suggest_int("ORB_nlevels", 1, 16),
        edgeThreshold=trial.suggest_int("ORB_edgeThreshold", 0, 200),
        scoreType=trial.suggest_categorical("ORB_scoreType", [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]),
        patchSize=trial.suggest_int("ORB_patchSize", 2, 200),
        fastThreshold=trial.suggest_int("ORB_fastThreshold", 0, 255),
    )


def build_method(trial: optuna.Trial, method_name: str) -> tuple:
    if method_name == "BRISK":
        method = cv2.BRISK_create(
            thresh=trial.suggest_int("BRISK_thresh", 0, 255),
            octaves=trial.suggest_int("BRISK_octaves", 0, 10),
            patternScale=trial.suggest_float("BRISK_pattern_scale", 0.1, 10),
        )
        return method.detect, method.compute, cv2.NORM_HAMMING
    elif method_name == "ORB":
        method = cv2.ORB_create(
            scaleFactor=trial.suggest_float("ORB_scaleFactor", 1.01, 4),
            nlevels=trial.suggest_int("ORB_nlevels", 1, 16),
            edgeThreshold=trial.suggest_int("ORB_edgeThreshold", 0, 200),
            WTA_K=trial.suggest_categorical("ORB_WTA_K", [2, 3, 4]),
            scoreType=trial.suggest_categorical("ORB_scoreType", [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE]),
            patchSize=trial.suggest_int("ORB_patchSize", 2, 200),
            fastThreshold=trial.suggest_int("ORB_fastThreshold", 0, 255),
        )
        return method.detect, method.compute, cv2.NORM_HAMMING
    elif method_name == "SIFT":
        method = _build_sift_descriptor(trial)
        return method.detect, method.compute, cv2.NORM_L2
    elif method_name == "AKAZE":
        method = cv2.AKAZE_create(
            descriptor_channels=trial.suggest_int("AKAZE_descriptor_channels", 1, 3),
            threshold=trial.suggest_float("AKAZE_threshold", 0, 0.1),
            nOctaves=trial.suggest_int("AKAZE_nOctaves", 1, 8),
            nOctaveLayers=trial.suggest_int("AKAZE_nOctaveLayers", 1, 8),
            diffusivity=trial.suggest_int("AKAZE_diffusivity", 0, 3),
        )
        return method.detect, method.compute, cv2.NORM_HAMMING
    elif method_name == "KAZE":
        method = cv2.KAZE_create(
            extended=trial.suggest_categorical("KAZE_extended", [True, False]),
            upright=trial.suggest_categorical("KAZE_upright", [True, False]),
            threshold=trial.suggest_float("KAZE_threshold", 0, 0.1),
            nOctaves=trial.suggest_int("KAZE_nOctaves", 1, 8),
            nOctaveLayers=trial.suggest_int("KAZE_nOctaveLayers", 1, 8),
            diffusivity=trial.suggest_int("KAZE_diffusivity", 0, 3),
        )
        return method.detect, method.compute, cv2.NORM_L2
    elif method_name == "SURF":
        method = cv2.xfeatures2d.SURF_create(
            hessianThreshold=trial.suggest_float("SURF_hessianThreshold", 0, 5000),
            nOctaves=trial.suggest_int("SURF_nOctaves", 1, 8),
            nOctaveLayers=trial.suggest_int("SURF_nOctaveLayers", 1, 8),
            extended=trial.suggest_categorical("SURF_extended", [True, False]),
            upright=trial.suggest_categorical("SURF_upright", [True, False]),
        )
        return method.detect, method.compute, cv2.NORM_L2
    elif method_name == "FAST_SIFT":
        detector = cv2.FastFeatureDetector_create(
            threshold=trial.suggest_int("FAST_threshold", 0, 255),
            nonmaxSuppression=trial.suggest_categorical("FAST_nonmaxSuppression", [True, False]),
            type=trial.suggest_categorical("FAST_type", [0, 1, 2]),
        )
        return detector.detect, _build_sift_descriptor(trial).compute, cv2.NORM_L2
    elif method_name == "GFTT_SIFT":
        detector = cv2.GFTTDetector_create(
            maxCorners=10000,
            qualityLevel=trial.suggest_float("GFTT_qualityLevel", 0, 1.0),
            minDistance=trial.suggest_float("GFTT_minDistance", 0, 100),
            blockSize=trial.suggest_int("GFTT_blockSize", 3, 51),
            useHarrisDetector=trial.suggest_categorical("GFTT_useHarrisDetector", [True, False]),
            k=trial.suggest_float("GFTT_k", 0, 0.5),
        )
        return detector.detect, _build_sift_descriptor(trial).compute, cv2.NORM_L2
    elif method_name == "MSER_SIFT":
        detector = cv2.MSER_create(
            delta=trial.suggest_int("MSER_delta", 1, 50),
            min_area=trial.suggest_int("MSER_min_area", 1, 1000),
            max_area=trial.suggest_int("MSER_max_area", 100, 100000),
            max_variation=trial.suggest_float("MSER_max_variation", 0, 1.0),
            min_diversity=trial.suggest_float("MSER_min_diversity", 0, 1.0),
        )
        return detector.detect, _build_sift_descriptor(trial).compute, cv2.NORM_L2
    elif method_name == "ORB_DAISY":
        descriptor = cv2.xfeatures2d.DAISY_create(
            radius=trial.suggest_float("DAISY_radius", 1, 100),
            q_radius=trial.suggest_int("DAISY_q_radius", 1, 8),
            q_theta=trial.suggest_int("DAISY_q_theta", 1, 32),
            q_hist=trial.suggest_int("DAISY_q_hist", 1, 32),
            norm=trial.suggest_categorical("DAISY_norm", [100, 101, 102, 103]),
            interpolation=trial.suggest_categorical("DAISY_interpolation", [True, False]),
            use_orientation=trial.suggest_categorical("DAISY_use_orientation", [True, False]),
        )
        return _build_orb_detector(trial).detect, descriptor.compute, cv2.NORM_L2
    elif method_name == "ORB_FREAK":
        descriptor = cv2.xfeatures2d.FREAK_create(
            orientationNormalized=trial.suggest_categorical("FREAK_orientationNormalized", [True, False]),
            scaleNormalized=trial.suggest_categorical("FREAK_scaleNormalized", [True, False]),
            patternScale=trial.suggest_float("FREAK_patternScale", 1, 100),
            nOctaves=trial.suggest_int("FREAK_nOctaves", 1, 8),
        )
        return _build_orb_detector(trial).detect, descriptor.compute, cv2.NORM_HAMMING


def objective(trial: optuna.Trial):
    timer = TrialTimer(TRIAL_TIMEOUT)
    try:
        detect_fn, compute_fn, distance_type = build_method(trial, active_method_name)
        feature_extractor = FeatureExtractor.from_opencv(
            detect_fn,
            compute_fn,
            distance_type,
        )

        if OBJECTIVE == "KITTI":
            return evaluate_kitti(feature_extractor, timer=timer)

        image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
        find_all_features_for_dataset(
            feature_extractor,
            dataset_image_sequences,
            image_feature_set,
            MAX_FEATURES,
            1,
            FORCE_CONSTANT_ANGLE,
            0,
            DOWNSAMPLE_FACTOR,
            DOWNSAMPLE_SIGMA,
            DOWNSAMPLE_INTERPOLATION_TYPE,
            timer=timer,
        )
        _, _ = calculate_valid_matches(
            image_feature_set,
            dataset_homography_sequence,
        )
        matching_match_sets = calculate_matching_evaluation(
            feature_extractor,
            image_feature_set,
            matching_approach,
            dataset_image_sequences,
            dataset_homography_sequence,
            VISUALIZE,
            SEQUENCE_TO_VISUALIZE,
            0,
            DOWNSAMPLE_FACTOR,
            DOWNSAMPLE_SIGMA,
            DOWNSAMPLE_INTERPOLATION_TYPE,
            timer=timer,
        )
        return np.mean([
            ms.get_average_precision_score(distance_match_rank_property)
            for ms in matching_match_sets
        ])

    except TrialTimeoutError:
        raise optuna.TrialPruned(f"Trial exceeded {TRIAL_TIMEOUT}s")
    except optuna.TrialPruned:
        raise
    except Exception:
        with open("failed_combinations.txt", "a") as f:
            f.write(f"{active_method_name}\n")
            f.write(traceback.format_exc() + "\n\n")
        raise
    finally:
        timer.cancel()

#########################################################
# ========================= MAIN ========================
#########################################################

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)

    def _make_study(method_name):
        study = optuna.create_study(
            study_name=f"{method_name}_{OBJECTIVE}",
            direction="minimize" if OBJECTIVE == "KITTI" else "maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100,
                multivariate=True,
                group=True,
            ),
            storage=f"sqlite:///{RESULTS_DIR / 'studies.db'}",
            load_if_exists=True,
        )
        if len(study.trials) == 0 and method_name in DEFAULT_PARAMS:
            study.enqueue_trial(DEFAULT_PARAMS[method_name])
        return study

    def _run_n_successful(study, method_name, n):
        global active_method_name
        active_method_name = method_name
        completed_before = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )

        def _stop_callback(study, _trial):
            if n is None:
                return
            new_complete = (
                sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
                - completed_before
            )
            if new_complete >= n:
                study.stop()

        study.optimize(objective, timeout=TIMEOUT, callbacks=[_stop_callback], catch=(Exception,))

    if ROUND_ROBIN:
        studies = {m: _make_study(m) for m in tqdm(methods, desc="Loading studies")}
        round_num = 0
        with tqdm(desc="Rounds", total=N_TRIALS) as pbar:
            while N_TRIALS is None or round_num < N_TRIALS:
                for method_name in tqdm(methods, desc="Methods", leave=False):
                    _run_n_successful(studies[method_name], method_name, 1)
                round_num += 1
                pbar.update(1)
    else:
        for method_name in tqdm(methods, desc="Methods"):
            _run_n_successful(_make_study(method_name), method_name, N_TRIALS)
