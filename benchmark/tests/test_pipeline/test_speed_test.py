from benchmark.pipeline import speed_test
from benchmark.feature_extractor import FeatureExtractor
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

@pytest.fixture()
def sample_keypoints_1() -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(i%100, i//100, 1, i, i, i, i) for i in range(1000)]


@pytest.fixture()
def sample_descriptions_1() -> list[np.ndarray]:
    return [np.ones(128) * i for i in range(1000)]


@pytest.fixture()
def sample_detect_keypoints_callable(sample_keypoints_1) -> Callable:
    def detect_keypoints_callable_mock(img : np.ndarray) -> list[cv2.KeyPoint]:
        return sample_keypoints_1
    return detect_keypoints_callable_mock


@pytest.fixture()
def sample_describe_keypoints_callable(sample_descriptions_1) -> Callable:
    def describe_keypoints_callable_mock(img : np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        return sample_descriptions_1
    return describe_keypoints_callable_mock


@pytest.fixture()
def sample_feature_extractor(sample_detect_keypoints_callable, sample_describe_keypoints_callable) -> FeatureExtractor:
    return FeatureExtractor(
        sample_detect_keypoints_callable,
        sample_describe_keypoints_callable,
        cv2.NORM_HAMMING)



### Test that invalid arguments fail ###


@pytest.mark.parametrize("bad_argument",
                        ["feature_extractor", "dataset_image_sequences", "single_element_in_list"],
                        ids=["Bad argument feature_extractor",
                             "Bad argument dataset_image_sequences",
                             "Bad argument single element in dataset_image_sequences"]
                            )
def test_invalid_arguments_speed_test(sample_feature_extractor, bad_argument):
    feature_extractor = sample_feature_extractor
    dataset_image_sequences = [[np.ones((640,480))]]

    if bad_argument == "feature_extractor":
        feature_extractor = None
    elif bad_argument == "dataset_image_sequences":
        dataset_image_sequences = None
    elif bad_argument == "single_element_in_list":
        dataset_image_sequences = [[np.ones((640,480))]] + [["not the same"]]

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        speed_test(feature_extractor, dataset_image_sequences)

### Test that valid arguments pass ###

def test_valid_arguments_speed_test(sample_feature_extractor):
    speed_test(sample_feature_extractor, [[np.ones((640,480))]])
