from benchmark.feature_extractor import FeatureExtractor
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time



@pytest.fixture()
def sample_keypoints_1() -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(i%100, i//100, 1, i, i, i, i) for i in range(1000)]

@pytest.fixture()
def sample_descriptions_1() -> list[np.ndarray]:
    return [np.ones(128) * i for i in range(1000)]


@pytest.fixture()
def sample_keypoints_2() -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(i%100, i//100, 1, i, i, i, i) for i in range(1000, 2000)]

@pytest.fixture()
def sample_descriptions_2() -> list[np.ndarray]:
    return [np.ones(128) * i for i in range(1000, 2000)]


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


@pytest.fixture()
def sample_opencv_detect_keypoints_callable(sample_keypoints_2) -> Callable:
    def detect_keypoints_callable_mock(img : np.ndarray) -> list[cv2.KeyPoint]:
        return sample_keypoints_2
    return detect_keypoints_callable_mock


@pytest.fixture()
def sample_opencv_describe_keypoints_callable(sample_descriptions_2) -> Callable:
    def describe_keypoints_callable_mock(img : np.ndarray, keypoints: list[cv2.KeyPoint]) -> Tuple[list[cv2.KeyPoint], list[np.ndarray]]:
        return keypoints, sample_descriptions_2
    return describe_keypoints_callable_mock


@pytest.fixture()
def sample_opencv_feature_extractor(sample_opencv_detect_keypoints_callable, sample_opencv_describe_keypoints_callable) -> FeatureExtractor:
    return FeatureExtractor.from_opencv(
        sample_opencv_detect_keypoints_callable,
        sample_opencv_describe_keypoints_callable,
        cv2.NORM_HAMMING)


### Test that invalid arguments fail ###

@pytest.mark.parametrize(
    "bad_argument",
    ["detect", "describe", "distance"],
    ids=[
        "All but detect",
        "All but describe",
        "All but distance"
    ]
)
def test_invalid_arguments_constructor(
    sample_detect_keypoints_callable,
    sample_describe_keypoints_callable,
    bad_argument,
    ):
    detect = sample_detect_keypoints_callable
    describe = sample_describe_keypoints_callable
    distance = cv2.NORM_HAMMING

    if bad_argument == "detect":
        detect = None
    elif bad_argument == "describe":
        describe = None
    elif bad_argument == "distance":
        distance = None

    with pytest.raises(TypeError):
        FeatureExtractor(detect, describe, distance)


@pytest.mark.parametrize(
    "bad_argument",
    ["detect", "describe", "distance"],
    ids=[
        "All but detect",
        "All but describe",
        "All but distance"
    ]
)
def test_invalid_from_opencv(
    sample_opencv_detect_keypoints_callable,
    sample_opencv_describe_keypoints_callable,
    bad_argument
    ):
    detect = sample_opencv_detect_keypoints_callable
    describe = sample_opencv_describe_keypoints_callable
    distance = cv2.NORM_HAMMING

    if bad_argument == "detect":
        detect = None
    elif bad_argument == "describe":
        describe = None
    elif bad_argument == "distance":
        distance = None

    with pytest.raises(TypeError):
        FeatureExtractor.from_opencv(detect, describe, distance)


def test_invalid_detect_keypoints(sample_feature_extractor):
    with pytest.raises(TypeError):
        sample_feature_extractor.detect_keypoints(None)


@pytest.mark.parametrize(
    "bad_argument",
    ["image", "keypoints"],
    ids=[
        "All but image",
        "All but keypoints"
    ]
)
def test_invalid_describe_keypoints(sample_feature_extractor, sample_keypoints_1, bad_argument):
    image = np.ones((640,480))
    keypoints = sample_keypoints_1

    if bad_argument == "image":
        image = None
    elif bad_argument == "keypoints":
        keypoints = None

    with pytest.raises(TypeError):
        sample_feature_extractor.describe_keypoints(image, keypoints)


def test_invalid_get_extraction_time_on_image(sample_feature_extractor):

    with pytest.raises(TypeError):
        sample_feature_extractor.get_extraction_time_on_image(None)



### Test that valid arguments pass ###


def test_valid_arguments_constructor(sample_detect_keypoints_callable, sample_describe_keypoints_callable):
    FeatureExtractor(sample_detect_keypoints_callable, sample_describe_keypoints_callable, cv2.NORM_L2)


def test_valid_from_opencv(sample_opencv_detect_keypoints_callable, sample_opencv_describe_keypoints_callable):
    FeatureExtractor.from_opencv(sample_opencv_detect_keypoints_callable, sample_opencv_describe_keypoints_callable, cv2.NORM_HAMMING)


def test_valid_argument_detect_keypoints(sample_feature_extractor):
    sample_feature_extractor.detect_keypoints(np.ones((640, 480)))


def test_valid_argument_describe_keypoints(sample_feature_extractor, sample_keypoints_1):
    sample_feature_extractor.describe_keypoints(np.ones((640, 480)), sample_keypoints_1)


def test_valid_empty_argument_describe_keypoints(sample_feature_extractor):
    sample_feature_extractor.describe_keypoints(np.ones((640, 480)), [])


def test_valid_argument_get_extraction_time_on_image(sample_feature_extractor):
    sample_feature_extractor.get_extraction_time_on_image(np.ones((640, 480)))



### Test that general cases behave expectedly ###

def test_detect_callaback_is_called(sample_feature_extractor, sample_keypoints_1):
    keypoints = sample_feature_extractor.detect_keypoints(np.ones((640,480)))
    for keypoint_index in range(len(keypoints)):
        assert keypoints[keypoint_index].pt[0] == sample_keypoints_1[keypoint_index].pt[0]
        assert keypoints[keypoint_index].pt[1] == sample_keypoints_1[keypoint_index].pt[1]
        assert keypoints[keypoint_index].size == sample_keypoints_1[keypoint_index].size
        assert keypoints[keypoint_index].angle == sample_keypoints_1[keypoint_index].angle
        assert keypoints[keypoint_index].response == sample_keypoints_1[keypoint_index].response
        assert keypoints[keypoint_index].octave == sample_keypoints_1[keypoint_index].octave
        assert keypoints[keypoint_index].class_id == sample_keypoints_1[keypoint_index].class_id


def test_describe_callaback_is_called(sample_feature_extractor, sample_keypoints_1, sample_descriptions_1):
    descriptions = sample_feature_extractor.describe_keypoints(np.ones((640,480)), sample_keypoints_1)
    for description_index in range(len(descriptions)):
        assert np.array_equal(descriptions[description_index], sample_descriptions_1[description_index])


def test_opencv_detect_callaback_is_called(sample_opencv_feature_extractor, sample_keypoints_2):
    keypoints = sample_opencv_feature_extractor.detect_keypoints(np.ones((640,480)))
    for keypoint_index in range(len(keypoints)):
        assert keypoints[keypoint_index].pt[0] == sample_keypoints_2[keypoint_index].pt[0]
        assert keypoints[keypoint_index].pt[1] == sample_keypoints_2[keypoint_index].pt[1]
        assert keypoints[keypoint_index].size == sample_keypoints_2[keypoint_index].size
        assert keypoints[keypoint_index].angle == sample_keypoints_2[keypoint_index].angle
        assert keypoints[keypoint_index].response == sample_keypoints_2[keypoint_index].response
        assert keypoints[keypoint_index].octave == sample_keypoints_2[keypoint_index].octave
        assert keypoints[keypoint_index].class_id == sample_keypoints_2[keypoint_index].class_id


def test_opencv_describe_callaback_is_called(sample_opencv_feature_extractor, sample_keypoints_2, sample_descriptions_2):
    descriptions = sample_opencv_feature_extractor.describe_keypoints(np.ones((640,480)), sample_keypoints_2)
    for description_index in range(len(descriptions)):
        assert np.array_equal(descriptions[description_index], sample_descriptions_2[description_index])