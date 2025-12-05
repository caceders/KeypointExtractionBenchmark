from benchmark.pipeline import calculate_retrieval_evaluation
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
import random

NUM_RELATED_IMAGES = 3
NUM_SEQUENCES = 10
NUM_FEATURES = 100
CORRECT_TO_RANDOM_RATIO = 100

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
        for _ in range(NUM_FEATURES):
            # Make sure the related feature has one valid match in all other images
            keypoint = cv2.KeyPoint(1, 1, 1)
            description = np.ones(128)
            reference_feature = Feature(keypoint, description, sequence_index, 0)
            related_features = [Feature(keypoint, description, sequence_index, image_index + 1) for image_index in range(NUM_RELATED_IMAGES)]
            image_feature_sequence.reference_image.append(reference_feature)
            for image_index, related_feature in enumerate(related_features):
                reference_feature.store_valid_match_for_image(image_index+1, related_feature, 1)
                related_feature.store_valid_match_for_image(0, reference_feature, 1)
                image_feature_sequence[image_index + 1].append(related_feature)
    return image_feature_set


@pytest.fixture()
def sample_matching_approach():
    def matching_approach_mock(reference_features : list[Feature], related_features: list[Feature], distance_type):
        size_of_shortest_length = min(len(reference_features), len(related_features))
        mixed_reference_features = reference_features.copy()
        mixed_related_features = related_features.copy()
        random.shuffle(mixed_reference_features)
        random.shuffle(mixed_related_features)
        matches = [Match(mixed_reference_features[feature_index], mixed_related_features[feature_index]) for feature_index in range(size_of_shortest_length)]
        return matches
    return matching_approach_mock

### Test that invalid arguments fail ###

@pytest.mark.parametrize("bad_argument",
                        ["feature_extractor", "correct_to_random_ratio", "dataset_image_sequences", "matching_approach"],
                        ids=["Bad argument feature_extractor",
                             "Bad argument correct_to_random_ratio",
                             "Bad argument dataset_image_sequences",
                             "Bad argument matching_approach"]
                            )
def test_invalid_arguments_calculate_retrieval_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach, bad_argument):
    feature_extractor = sample_feature_extractor
    image_feature_set = sample_image_feature_set
    matching_approach = sample_matching_approach
    correct_to_random_ratio = CORRECT_TO_RANDOM_RATIO

    if bad_argument == "feature_extractor":
        feature_extractor = None
    elif bad_argument == "correct_to_random_ratio":
        image_feature_set = None
    elif bad_argument == "dataset_image_sequences":
        correct_to_random_ratio = None
    elif bad_argument == "matching_approach":
        matching_approach = None

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        calculate_retrieval_evaluation(feature_extractor, image_feature_set, correct_to_random_ratio, matching_approach)


### Test that valid arguments pass ###

def test_valid_arguments_calculate_retrieval_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach):
    calculate_retrieval_evaluation(sample_feature_extractor, sample_image_feature_set, CORRECT_TO_RANDOM_RATIO, sample_matching_approach)


### Test that general cases behave expectedly ###

@pytest.mark.parametrize("run", range(10))
def test_expected_calculate_retrieval_evaluation(sample_feature_extractor, sample_image_feature_set, sample_matching_approach, run):

    match_sets = calculate_retrieval_evaluation(sample_feature_extractor, sample_image_feature_set, CORRECT_TO_RANDOM_RATIO, sample_matching_approach)

    for _, match_set in enumerate(match_sets):
        assert len(match_set) == NUM_FEATURES
        for match in match_set:
            assert match.reference_feature.image_index == 0, "The reference features should all be form the reference image with index 0"
