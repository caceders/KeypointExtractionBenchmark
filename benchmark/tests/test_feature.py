from benchmark.feature import Feature
import pytest
import cv2
import numpy as np
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

@pytest.fixture()
def sample_feature1() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 1)


@pytest.fixture()
def sample_feature2() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 2)


@pytest.fixture()
def sample_feature_with_valid_matches() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    this_feature = Feature(kp, desc, 1, 2)
    for i in range(3):
        kp = cv2.KeyPoint(100 + i, 200 + i, 1)
        desc = np.ones(128) * i
        this_feature.store_valid_match_for_image(i, Feature(kp, desc, 1, 1), i)
    return this_feature



### Test that invalid arguments fail ###


@pytest.mark.parametrize("kp, desc, sequence_index, image_index", [
    (None, np.ones(3), 5, 1),
    (cv2.KeyPoint(1,2,3), None, 5, 1),
    (cv2.KeyPoint(1,2,3), np.ones(3), None, 1),
    (cv2.KeyPoint(1,2,3), np.ones(3), 5, None),
    ], ids = [
        "Bad argument keypoint",
        "Bad argument description",
        "Bad argument sequence index",
        "Bad argument image index"
    ])
def test_invalid_arguments_constructor(kp, desc, sequence_index, image_index):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        Feature(kp, desc, sequence_index, image_index)


@pytest.mark.parametrize("related_image_index, feature, score", [
    (None, Feature(cv2.KeyPoint(1,2,3), np.ones(3), 1, 2), 5),
    (4, None, 5),
    (4, Feature(cv2.KeyPoint(1,2,3), np.ones(3), 1, 2), None),
    ], ids = [
        "Bad argument related_image_index",
        "Bad argument feature",
        "Bad argument score",
    ])
def test_invalid_arguments_store_valid_match_for_image(sample_feature1, related_image_index, feature, score):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature1.store_valid_match_for_image(related_image_index, feature, score)


def test_invalid_arguments_get_valid_matches_for_image(sample_feature_with_valid_matches):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature_with_valid_matches.get_valid_matches_for_image(None)


def test_invalid_arguments_is_match_with_other_valid(sample_feature_with_valid_matches):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature_with_valid_matches.is_match_with_other_valid(None)


def test_invalid_arguments_get_pt_after_homography_transform(sample_feature1):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_feature1.get_pt_after_homography_transform("Not a numpy array")



### Test that valid arguments pass ###


def test_valid_arguments_constructor():
    Feature(cv2.KeyPoint(1,2,3,4,5,6,7), np.ones(3), 1, 1)


def test_valid_arguments_store_valid_match_for_image(sample_feature1, sample_feature2):
    sample_feature1.store_valid_match_for_image(1, sample_feature2, 10)


def test_valid_arguments_get_valid_matches_for_image(sample_feature_with_valid_matches):
    sample_feature_with_valid_matches.get_valid_matches_for_image(2)


def test_valid_arguments_get_all_valid_matches(sample_feature_with_valid_matches):
    sample_feature_with_valid_matches.get_all_valid_matches()


def test_valid_arguments_is_match_with_other_valid(sample_feature_with_valid_matches, sample_feature2):
    sample_feature_with_valid_matches.is_match_with_other_valid(sample_feature2)


def test_valid_arguments_pt(sample_feature1):
    sample_feature1.pt


def test_valid_arguments_get_pt_after_homography_transform(sample_feature1):
    sample_feature1.get_pt_after_homography_transform(np.eye(3))



### Test that general cases behave expectedly ###


def test_store_valid_feature(sample_feature1: Feature, sample_feature2: Feature):
    sample_feature1.store_valid_match_for_image(0, sample_feature2, 10)
    assert len(sample_feature1._image_valid_matches) == 1, "After calling store_valid_match_for_image() with a single feature for the first time, the _image_valid_matches did not contain only one image"
    assert len(sample_feature1._image_valid_matches[0]) == 1, "After calling store_valid_match_for_image() with a single feature for the first time, the related image in _image_valid_matches did not contain ONE feature"
    assert sample_feature2 in sample_feature1._image_valid_matches[0].keys(), "After calling store_valid_match_for_image() with a single feature for the first time, the related image in _image_valid_matches did not contain ONLY that feature"

    assert len(sample_feature1._all_valid_matches) == 1, "After calling store_valid_match_for_image() with a single feature for the first time, there was not in total only ONE feature in _all_valid_matches"
    assert sample_feature2 in sample_feature1._all_valid_matches.keys(), "After calling store_valid_match_for_image() with a single feature for the first time, there was not only that one feature in _all_valid_matches"


def test_get_all_valid_matches(sample_feature_with_valid_matches: Feature, sample_feature2: Feature):
    # Adding sample_features2 which is 3 features long
    all_valid = sample_feature_with_valid_matches.get_all_valid_matches()
    assert len(all_valid) == 3, "sample_feature_with_valid_matches was expected to have 3 valid features, but it did not"

    sample_feature_with_valid_matches.store_valid_match_for_image(500, sample_feature2, 1)
    all_valid = sample_feature_with_valid_matches.get_all_valid_matches()
    assert len(all_valid) == 4, "After storing a single feature there was not 4 valid features in the fixture"

    assert sample_feature2 in all_valid.keys(), "The added feature was not part of the return dictionary from the get_all_valid_matches() function"


def test_get_valid_matches_for_image(sample_feature_with_valid_matches: Feature, sample_feature2: Feature):
    valid = sample_feature_with_valid_matches.get_valid_matches_for_image(0)
    assert len(valid) == 1, "Before storing a feature there was not only 1 valid features for image 0 in the fixture"

    sample_feature_with_valid_matches.store_valid_match_for_image(0, sample_feature2, 1)
    valid = sample_feature_with_valid_matches.get_valid_matches_for_image(0)
    assert len(valid) == 2, "After storing a feature there was not 2 valid features for image 0 in the fixture"

    assert sample_feature2 in valid.keys(), "The added feature was not part of all the dictionary returned from get_valid_matches_for_image for the relevant image in the fixture"


def test_is_match_with_other_valid(sample_feature_with_valid_matches: Feature, sample_feature2: Feature):
    assert not sample_feature_with_valid_matches.is_match_with_other_valid(sample_feature2), "Before storing a feature to the images valid matches the match with that feature was valid"
    sample_feature_with_valid_matches.store_valid_match_for_image(0, sample_feature2, 1)
    assert sample_feature_with_valid_matches.is_match_with_other_valid(sample_feature2), "Afer storing the feature to the images valid matches the match with that feature was not valid"


def test_pt(sample_feature1):
    assert np.array_equal(sample_feature1.pt, np.array([100, 200])), "The pt property did not retrurn an array of the form [x, y]"


def test_homography_transform(sample_feature1: Feature):
    # x, y, w = 100, 200, 1
    #
    # H = 5, 6, 10
    #     4, 7, 20
    #     3, 2, 19
    #
    # x' = 5 ∗ 100 + 6 ∗ 200 + 10 ∗ 1  = 1710
    # y' = 6 * 100 + 7 * 200 + 20 * 1 = 1820
    # w' = 3 * 100 + 2 * 200 + 19 * 1 = 719
    #
    # x_new = 1710/719 = 2.3783031988873433
    # y_new = 1820/719 = 2.5312934631432547
    H = np.array([
        [5, 6, 10],
        [4, 7, 20],
        [3, 2, 19]
    ], dtype=float)

    x, y = sample_feature1.get_pt_after_homography_transform(H)
    assert np.isclose(x, 2.3783031988873433), "The x value for the transformed was different than the expected 2.3783031988873433"
    assert np.isclose(y, 2.5312934631432547), "The y value for the transformed was different than the expected 2.5312934631432547"