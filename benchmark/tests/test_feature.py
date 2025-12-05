from benchmark.feature import Feature
import pytest
import cv2
import numpy as np


@pytest.fixture()
def sample_feature_1() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 1)



@pytest.fixture()
def sample_feature_2() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 2)



@pytest.fixture()
def feature_with_valid_matches() -> Feature:
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
    (cv2.KeyPoint(1,2,3,4,5,6,7), None, 5, 1),
    (cv2.KeyPoint(1,2,3,4,5,6,7), np.ones(3), None, 1),
    (cv2.KeyPoint(1,2,3,4,5,6,7), np.ones(3), 5, None),
    ], ids = [
        "All but keypoint",
        "All but description",
        "All but sequence index",
        "All but image index"
    ])
def test_invalid_arguments_constructor(kp, desc, sequence_index, image_index):
    with pytest.raises(TypeError):
        Feature(kp, desc, sequence_index, image_index)



@pytest.mark.parametrize("related_image_index, feature, score", [
    (None, Feature(cv2.KeyPoint(1,2,3,4,5,6,7), np.ones(3), 1, 2), 5),
    (4, None, 5),
    (4, Feature(cv2.KeyPoint(1,2,3,4,5,6,7), np.ones(3), 1, 2), None),
    ], ids = [
        "All but related_image_index",
        "All but feature",
        "All but score",
    ])
def test_invalid_arguments_store_valid_match_for_image(sample_feature_1, related_image_index, feature, score):
    with pytest.raises(TypeError):
        sample_feature_1.store_valid_match_for_image(related_image_index, feature, score)



def test_invalid_arguments_get_valid_matches_for_image(feature_with_valid_matches):
    with pytest.raises(TypeError):
        feature_with_valid_matches.get_valid_matches_for_image(None)



def test_invalid_arguments_is_match_with_other_valid(feature_with_valid_matches):
    with pytest.raises(TypeError):
        feature_with_valid_matches.is_match_with_other_valid(None)



def test_invalid_arguments_get_pt_after_homography_transform(sample_feature_1):
    with pytest.raises(TypeError):
        sample_feature_1.get_pt_after_homography_transform("Not a numpy array")



### Test that valid arguments pass ###


def test_valid_arguments_constructor():
    Feature(cv2.KeyPoint(1,2,3,4,5,6,7), np.ones(3), 1, 1)


def test_valid_arguments_store_valid_match_for_image(sample_feature_1, sample_feature_2):
    sample_feature_1.store_valid_match_for_image(1, sample_feature_2, 10)


def test_valid_arguments_get_valid_matches_for_image(feature_with_valid_matches):
    feature_with_valid_matches.get_valid_matches_for_image(2)


def test_valid_arguments_get_all_valid_matches(feature_with_valid_matches):
    feature_with_valid_matches.get_all_valid_matches()


def test_valid_arguments_is_match_with_other_valid(feature_with_valid_matches, sample_feature_2):
    feature_with_valid_matches.is_match_with_other_valid(sample_feature_2)


def test_valid_arguments_pt(sample_feature_1):
    sample_feature_1.pt


def test_valid_arguments_get_pt_after_homography_transform(sample_feature_1):
    sample_feature_1.get_pt_after_homography_transform(np.eye(3))



### Test that general cases behave expectedly ###


def test_store_valid_feature(sample_feature_1: Feature, sample_feature_2: Feature):
    sample_feature_1.store_valid_match_for_image(0, sample_feature_2, 10)
    assert len(sample_feature_1._image_valid_matches) == 1, "Adding only one feature should add an entry only one image"
    assert len(sample_feature_1._image_valid_matches[0]) == 1, "Adding only one feature the related image should only contain one feature"
    assert sample_feature_2 in sample_feature_1._image_valid_matches[0].keys(), "Adding only one feature the related image should only contain that feature"

    assert len(sample_feature_1._all_valid_matches) == 1, "Adding only one feature there should in total be only one feature"
    assert sample_feature_2 in sample_feature_1._all_valid_matches.keys(), "Adding only one feature there should only be that one feature"


def test_get_all_valid_matches(feature_with_valid_matches: Feature, sample_feature_2: Feature):
    all_valid = feature_with_valid_matches.get_all_valid_matches()
    assert len(all_valid) == 3, "Before adding a feature there should be 3 valid features in the fixture"

    feature_with_valid_matches.store_valid_match_for_image(500, sample_feature_2, 1)
    all_valid = feature_with_valid_matches.get_all_valid_matches()
    assert len(all_valid) == 4, "After adding a feature there should be 4 valid features in the fixture"

    assert sample_feature_2 in all_valid.keys(), "The added feature should be part of all the valid features in the fixture"


def test_get_valid_matches_for_image(feature_with_valid_matches: Feature, sample_feature_2: Feature):
    valid = feature_with_valid_matches.get_valid_matches_for_image(0)
    assert len(valid) == 1, "Before adding a feature there should be 1 valid features for image 0 in the fixture"

    feature_with_valid_matches.store_valid_match_for_image(0, sample_feature_2, 1)
    valid = feature_with_valid_matches.get_valid_matches_for_image(0)
    assert len(valid) == 2, "After adding a feature there should be 2 valid features for image 0 in the fixture"

    assert sample_feature_2 in valid.keys(), "The added feature should be part of all the valid features for that image in the fixture"


def test_is_match_with_other_valid(feature_with_valid_matches: Feature, sample_feature_2: Feature):
    assert not feature_with_valid_matches.is_match_with_other_valid(sample_feature_2), "Before adding a feature to the images valid matches the match with that feature should not be valid"
    feature_with_valid_matches.store_valid_match_for_image(0, sample_feature_2, 1)
    assert feature_with_valid_matches.is_match_with_other_valid(sample_feature_2), "Afer adding the feature to the images valid matches the match with that feature should be valid"


def test_pt(sample_feature_1):
    assert np.array_equal(sample_feature_1.pt, np.array([100, 200])), "The pt property should return an array of the form [x, y]"

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

def test_homography_transform(sample_feature_1: Feature):
    H = np.array([
        [5, 6, 10],
        [4, 7, 20],
        [3, 2, 19]
    ], dtype=float)

    x, y = sample_feature_1.get_pt_after_homography_transform(H)
    assert np.isclose(x, 2.3783031988873433)
    assert np.isclose(y, 2.5312934631432547)