from benchmark.pipeline import find_all_features_for_dataset
from benchmark.feature_extractor import FeatureExtractor
from benchmark.image_feature_set import ImageFeatureSet
from benchmark.feature import Feature
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

NUM_RELATED_IMAGES = 2
NUM_SEQUENCES = 10


@pytest.fixture()
def sample_detect_keypoints_callable() -> Callable:
    def detect_keypoints_callable_mock(img : np.ndarray) -> list[cv2.KeyPoint]:
        return [cv2.KeyPoint(img[0][0], i//100, 1) for i in range(1000)]
    return detect_keypoints_callable_mock


@pytest.fixture()
def sample_describe_keypoints_callable() -> Callable:
    def describe_keypoints_callable_mock(img : np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        return [np.ones(128) * keypoints[0].pt[0] for i in range(1000)]
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
    for i in range(60):
        kp = cv2.KeyPoint(100 + i, 200 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 1, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_features_2():
    features = []
    for i in range(30):
        kp = cv2.KeyPoint(200 + i, 300 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 2, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_image_feature_set(sample_features_1, sample_features_2) -> ImageFeatureSet:
    image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
    for feature in sample_features_1:
        image_feature_set[0][feature.image_index].append(feature)
    for feature in sample_features_2:
        image_feature_set[1][feature.image_index].append(feature)
    return image_feature_set

### Test that invalid arguments fail ###


@pytest.mark.parametrize("bad_argument",
                        ["feature_extractor", "dataset_image_sequences", "image_feature_set", "max_features"],
                        ids=["Bad argument feature_extractor",
                             "Bad argument dataset_image_sequences",
                             "Bad argument image_feature_set",
                             "Bad argument max_features"]
                            )
def test_invalid_arguments_find_all_features_for_dataset(sample_feature_extractor, sample_image_feature_set, bad_argument):
    feature_extractor = sample_feature_extractor
    dataset_image_sequences = [[np.ones((640,480))]]
    image_feature_set = sample_image_feature_set
    max_features = 100

    if bad_argument == "feature_extractor":
        feature_extractor = None
    elif bad_argument == "dataset_image_sequences":
        dataset_image_sequences = None
    elif bad_argument == "image_feature_set":
        image_feature_set = None
    elif bad_argument == "max_features":
        max_features = None

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        find_all_features_for_dataset(feature_extractor, dataset_image_sequences, image_feature_set, max_features)

### Test that valid arguments pass ###

def test_valid_arguments_find_all_features_for_dataset(sample_feature_extractor, sample_image_feature_set):
    find_all_features_for_dataset(sample_feature_extractor, [[np.ones((640,480))]], sample_image_feature_set, 100)

### Test that general cases behave expectedly ###

def test_find_all_features_for_dataset_store_correct_features_for_correct_image(sample_feature_extractor, sample_image_feature_set):
    
    # Image one has all 1s, image 2 all 2s and so on. Keypoint x value same as image[0][0], so for
    # image 1 keypoint.x == 1, image 2 keypoint.x == 2 and so on. The descriptor is filled with the keypoint.x value.
    dataset_image_sequences = []
    image_number = 0
    for sequence_index in range(NUM_SEQUENCES):
        sequence = []
        for image_index in range(NUM_RELATED_IMAGES + 1):
            image = np.ones((640, 480)) * image_number
            image_number += 1
            sequence.append(image)
        dataset_image_sequences.append(sequence)
    
    find_all_features_for_dataset(sample_feature_extractor, dataset_image_sequences, sample_image_feature_set, 1000)

    image_number = 0
    for sequence_index in range(NUM_SEQUENCES):
        for image_index in range(NUM_RELATED_IMAGES + 1):
            features = sample_image_feature_set[sequence_index][image_index]
            assert all(feature.keypoint.pt[0] == image_number for feature in features)
            assert all(feature.description[0] == image_number for feature in features)
            image_number += 1

    