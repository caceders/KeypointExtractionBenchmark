from benchmark.matching import Match
from benchmark.feature import Feature
import cv2
import numpy as np
import pytest

@pytest.fixture()
def sample_feature_1() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 1)

@pytest.fixture()
def sample_feature_2() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 1)

@pytest.fixture()
def sample_feature_3() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 2)

@pytest.fixture()
def sample_feature_4() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 2, 1)



### Test that invalid arguments fail ###


@pytest.mark.parametrize("bad_argument",
                        ["feature1", "feature2"],
                        ids=["Bad argument feature1",
                             "Bad argument feature2"]
                            )
def test_invalid_arguments_constructor(sample_feature_1, sample_feature_2, bad_argument):
    feature1 = sample_feature_1
    feature2 = sample_feature_2

    if bad_argument == "feature1":
        feature1 = None
    elif bad_argument == "feature2":
        feature2 = None

    with pytest.raises(TypeError):
        Match(feature1, feature2)



### Test that general cases behave expectedly ###


def test_is_in_same_sequence(sample_feature_1, sample_feature_2, sample_feature_3, sample_feature_4):
    match = Match(sample_feature_1, sample_feature_2)
    assert match.is_in_same_sequece

    match = Match(sample_feature_1, sample_feature_3)
    assert match.is_in_same_sequece

    match = Match(sample_feature_1, sample_feature_4)
    assert not match.is_in_same_sequece


def test_is_correct(sample_feature_1, sample_feature_2, sample_feature_3, sample_feature_4):
    sample_feature_1.store_valid_match_for_image(1, sample_feature_2, 10)
    match = Match(sample_feature_1, sample_feature_2)
    assert match.is_correct

    match = Match(sample_feature_1, sample_feature_3)
    assert not match.is_correct