from benchmark.pipeline import calculate_matching_evaluation
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.feature import Feature
from benchmark.matching import MatchSet, Match
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

NUM_RELATED_IMAGES = 2
NUM_SEQUENCES = 10
NUM_FEATURES = 100

@pytest.fixture()
def sample_detect_keypoints_callable() -> Callable:
    def detect_keypoints_callable_mock(img : np.ndarray) -> list[cv2.KeyPoint]:
        return [cv2.KeyPoint(img[0][0], i//100, 1) for i in range(NUM_FEATURES)]
    return detect_keypoints_callable_mock


@pytest.fixture()
def sample_describe_keypoints_callable() -> Callable:
    def describe_keypoints_callable_mock(img : np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        return [np.ones(128) * keypoints[0].pt[0] for i in range(NUM_FEATURES)]
    return describe_keypoints_callable_mock


@pytest.fixture()
def sample_feature_extractor(sample_detect_keypoints_callable, sample_describe_keypoints_callable) -> FeatureExtractor:
    return FeatureExtractor(
        sample_detect_keypoints_callable,
        sample_describe_keypoints_callable,
        cv2.NORM_HAMMING)

@pytest.fixture()
def sample_features_1():
    features = []
    for i in range(NUM_FEATURES):
        kp = cv2.KeyPoint(100 + i, 200 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 1, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_features_2():
    features = []
    for i in range(NUM_FEATURES):
        kp = cv2.KeyPoint(200 + i, 300 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 2, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_image_feature_set() -> ImageFeatureSet:
    image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
    for sequence_index, image_feature_sequence in enumerate(image_feature_set):
        for image_index, image_features in enumerate(image_feature_sequence):
            for _ in range(NUM_FEATURES):
                keypoint = cv2.KeyPoint(1, 1, 1)
                description = np.ones(128)
                image_features.append(Feature(keypoint, description, sequence_index, image_index))
    return image_feature_set


@pytest.fixture()
def sample_matching_approach():
    def matching_approach_mock(reference_features : list[Feature], related_features: list[Feature], distance_type):
        size_of_shortest_length = min(len(reference_features), len(related_features))
        matches = [Match(reference_features[feature_index], related_features[feature_index]) for feature_index in range(size_of_shortest_length)]
        return matches
    return matching_approach_mock

### Test that invalid arguments fail ###

@pytest.mark.parametrize("bad_argument",
                        ["feature_extractor", "dataset_image_sequences", "matching_approach"],
                        ids=["Bad argument feature_extractor",
                             "Bad argument dataset_image_sequences",
                             "Bad argument matching_approach"]
                            )
def test_invalid_arguments_calculate_matching_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach, bad_argument):
    feature_extractor = sample_feature_extractor
    image_feature_set = sample_image_feature_set
    matching_approach = sample_matching_approach

    if bad_argument == "feature_extractor":
        feature_extractor = None
    elif bad_argument == "dataset_image_sequences":
        image_feature_set = None
    elif bad_argument == "matching_approach":
        matching_approach = None

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        calculate_matching_evaluation(feature_extractor, image_feature_set, matching_approach)



### Test that valid arguments pass ###

def test_valid_arguments_calculate_matching_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach):
    calculate_matching_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach)




### Test that general cases behave expectedly ###

def test_expected_calculate_matching_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach):
    match_sets = calculate_matching_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach)
    for match_set_index, match_set in enumerate(match_sets):
        assert len(match_set) == NUM_FEATURES * NUM_RELATED_IMAGES
        for match in match_set:
            assert match.reference_feature.image_index == 0, "The reference features should all be form the reference image with index 0"
            assert match.related_feature.image_index != 0, "The related features should never be in the reference image with index 0"
            assert match.reference_feature.sequence_index == match_set_index, ""
            assert match.related_feature.sequence_index == match_set_index, ""