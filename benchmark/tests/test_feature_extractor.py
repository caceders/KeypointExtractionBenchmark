from benchmark.feature_extractor import FeatureExtractor
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation



IMAGE_RESOLUTION = (640, 480)
NUM_KEYPOINTS = 1000



@pytest.fixture()
def sample_keypoints1() -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(i%100, i//100, 1, i, i, i, i) for i in range(NUM_KEYPOINTS)]


@pytest.fixture()
def sample_descriptions1() -> list[np.ndarray]:
    return [np.ones(128) * i for i in range(NUM_KEYPOINTS)]


@pytest.fixture()
def sample_keypoints2() -> list[cv2.KeyPoint]:
    # A sligtly different range than sample_keypoints1
    return [cv2.KeyPoint(i%100, i//100, 1, i, i, i, i) for i in range(NUM_KEYPOINTS, 2 * NUM_KEYPOINTS)]


@pytest.fixture()
def sample_descriptions2() -> list[np.ndarray]:
    return [np.ones(128) * i for i in range(NUM_KEYPOINTS, 2 * NUM_KEYPOINTS)]


@pytest.fixture()
def sample_detect_keypoints_callable(sample_keypoints1) -> Callable:
    def detect_keypoints_callable(img : np.ndarray) -> list[cv2.KeyPoint]:
        return sample_keypoints1
    return detect_keypoints_callable


@pytest.fixture()
def sample_describe_keypoints_callable(sample_descriptions1) -> Callable:
    def describe_keypoints_callable(img : np.ndarray, keypoints: list[cv2.KeyPoint]) -> list[np.ndarray]:
        return sample_descriptions1
    return describe_keypoints_callable


@pytest.fixture()
def sample_feature_extractor(sample_detect_keypoints_callable, sample_describe_keypoints_callable) -> FeatureExtractor:
    return FeatureExtractor(
        sample_detect_keypoints_callable,
        sample_describe_keypoints_callable,
        cv2.NORM_HAMMING)


@pytest.fixture()
def sample_opencv_detect_keypoints_callable(sample_keypoints2) -> Callable:
    def detect_keypoints_callable(img : np.ndarray) -> list[cv2.KeyPoint]:
        return sample_keypoints2
    return detect_keypoints_callable


@pytest.fixture()
def sample_opencv_describe_keypoints_callable(sample_descriptions2) -> Callable:
    def describe_keypoints_callable(img : np.ndarray, keypoints: list[cv2.KeyPoint]) -> Tuple[list[cv2.KeyPoint], list[np.ndarray]]:
        return keypoints, sample_descriptions2
    return describe_keypoints_callable


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

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
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

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        FeatureExtractor.from_opencv(detect, describe, distance)


def test_invalid_detect_keypoints(sample_feature_extractor):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature_extractor.detect_keypoints(None)


@pytest.mark.parametrize(
    "bad_argument",
    ["image", "keypoints"],
    ids=[
        "Bad argument image",
        "Bad argument keypoints"
    ]
)
def test_invalid_describe_keypoints(sample_feature_extractor, sample_keypoints1, bad_argument):
    image = np.ones(IMAGE_RESOLUTION)
    keypoints = sample_keypoints1

    if bad_argument == "image":
        image = None
    elif bad_argument == "keypoints":
        keypoints = None

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature_extractor.describe_keypoints(image, keypoints)


def test_invalid_get_extraction_time_on_image(sample_feature_extractor):

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature_extractor.get_extraction_time_on_image(None)



### Test that valid arguments pass ###


def test_valid_arguments_constructor(sample_detect_keypoints_callable, sample_describe_keypoints_callable):
    FeatureExtractor(sample_detect_keypoints_callable, sample_describe_keypoints_callable, cv2.NORM_L2)


def test_valid_from_opencv(sample_opencv_detect_keypoints_callable, sample_opencv_describe_keypoints_callable):
    FeatureExtractor.from_opencv(sample_opencv_detect_keypoints_callable, sample_opencv_describe_keypoints_callable, cv2.NORM_HAMMING)


def test_valid_argument_detect_keypoints(sample_feature_extractor):
    sample_feature_extractor.detect_keypoints(np.ones(IMAGE_RESOLUTION))


def test_valid_argument_describe_keypoints(sample_feature_extractor, sample_keypoints1):
    sample_feature_extractor.describe_keypoints(np.ones(IMAGE_RESOLUTION), sample_keypoints1)


def test_valid_empty_argument_describe_keypoints(sample_feature_extractor):
    sample_feature_extractor.describe_keypoints(np.ones(IMAGE_RESOLUTION), [])


def test_valid_argument_get_extraction_time_on_image(sample_feature_extractor):
    sample_feature_extractor.get_extraction_time_on_image(np.ones(IMAGE_RESOLUTION))



### Test that general cases behave expectedly ###


def test_detect_callaback_is_called(sample_feature_extractor, sample_keypoints1):
    # Keypoints should be identical to sample_keypoints1 fixture
    keypoints = sample_feature_extractor.detect_keypoints(np.ones((640,480)))
    assert len(keypoints) == NUM_KEYPOINTS, "detect_keypoint function in FeatureExtractor did not return same amount of keypoints as detect_keypoint callback function"
    for keypoint_index in range(len(keypoints)):
        keypoint_same = (keypoints[keypoint_index].pt[0] == sample_keypoints1[keypoint_index].pt[0] and
                         keypoints[keypoint_index].pt[1] == sample_keypoints1[keypoint_index].pt[1] and
                         keypoints[keypoint_index].size == sample_keypoints1[keypoint_index].size and
                         keypoints[keypoint_index].angle == sample_keypoints1[keypoint_index].angle and
                         keypoints[keypoint_index].response == sample_keypoints1[keypoint_index].response and
                         keypoints[keypoint_index].octave == sample_keypoints1[keypoint_index].octave and
                         keypoints[keypoint_index].class_id == sample_keypoints1[keypoint_index].class_id)
    
        assert keypoint_same, "Keypoint returned from detect_keypoint function in FeatureExtractor was different than the keypoint returned from the detect_keypoint callback function"


def test_describe_callaback_is_called(sample_feature_extractor, sample_keypoints1, sample_descriptions1):
    # Descriptions should be identical to sample_descriptions1 fixture
    descriptions = sample_feature_extractor.describe_keypoints(np.ones((640,480)), sample_keypoints1)
    assert len(descriptions) == NUM_KEYPOINTS, "describe_keypoint function in FeatureExtractor did not return same amount of descriptions as describe_keypoint callback function"
    for description_index in range(len(descriptions)):
        assert np.array_equal(descriptions[description_index], sample_descriptions1[description_index]), "Description returned from describe_keypoint function in FeatureExtractor was different than the keypoint returned from the describe_keypoint callback function"


def test_opencv_detect_callaback_is_called(sample_opencv_feature_extractor, sample_keypoints2):
    # Keypoints should be identical to sample_keypoints1 fixture
    keypoints = sample_opencv_feature_extractor.detect_keypoints(np.ones((640,480)))
    assert len(keypoints) == NUM_KEYPOINTS, "detect_keypoint function in FeatureExtractor generated from from_opencv() class method did not return same amount of keypoints as detect_keypoint callback function"
    for keypoint_index in range(len(keypoints)):
        for keypoint_index in range(len(keypoints)):
            keypoint_same = (keypoints[keypoint_index].pt[0] == sample_keypoints2[keypoint_index].pt[0] and
                            keypoints[keypoint_index].pt[1] == sample_keypoints2[keypoint_index].pt[1] and
                            keypoints[keypoint_index].size == sample_keypoints2[keypoint_index].size and
                            keypoints[keypoint_index].angle == sample_keypoints2[keypoint_index].angle and
                            keypoints[keypoint_index].response == sample_keypoints2[keypoint_index].response and
                            keypoints[keypoint_index].octave == sample_keypoints2[keypoint_index].octave and
                            keypoints[keypoint_index].class_id == sample_keypoints2[keypoint_index].class_id)


def test_opencv_describe_callaback_is_called(sample_opencv_feature_extractor, sample_keypoints2, sample_descriptions2):
    # Descriptions should be identical to sample_descriptions1 fixture
    descriptions = sample_opencv_feature_extractor.describe_keypoints(np.ones((640,480)), sample_keypoints2)
    assert len(descriptions) == NUM_KEYPOINTS, "describe_keypoint function in FeatureExtractor generated from from_opencv() did not return same amount of descriptions as describe_keypoint callback function"
    for description_index in range(len(descriptions)):
        assert np.array_equal(descriptions[description_index], sample_descriptions2[description_index]), "Description returned from describe_keypoint function in FeatureExtractor generated from from_opencv() was different than the keypoint returned from the describe_keypoint callback function"