from benchmark.feature import Feature
import pytest
import cv2
import numpy as np



@pytest.fixture()
def feature() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128) # 128 dimensional SIFT descriptor
    return Feature(kp, desc, 1, 1)



@pytest.fixture
def features() -> list[Feature]:
    feats = []
    for i in range(3):
        kp = cv2.KeyPoint(100 + i, 200 + i, 1)
        desc = np.ones(128) * i # 128 dimensional SIFT descriptor
        feats.append(Feature(kp, desc, 1, 1))
    return feats



@pytest.mark.parametrize("kp, desc", [
    (None, None),
    (1, "F"),
    (np.ones(128), cv2.KeyPoint(100, 200, 1)),
    (cv2.KeyPoint(100, 200, 1), None)
])
def test_invalid_constructor(kp, desc):
    with pytest.raises(TypeError):
        feature = Feature(kp, desc, 1, 1)



def test_store_valid_match_for_image(feature: Feature, features: list[Feature]):
    add_feature = features[0]
    feature.store_valid_match_for_image(0, add_feature, 10)
    assert add_feature in feature._all_valid_matches, "Feature should be stored in all valid features"
    assert add_feature in feature._image_valid_matches[0], "Feature should be stored in valid features for the relevant image"



def test_homography_transform(feature: Feature):
    H = np.array([
        [1, 0, 10],
        [0, 1, 20],
        [0, 0, 1]
    ], dtype=float)

    x, y = feature.get_pt_after_homography_transform(H)
    assert (x, y) == (feature.pt[0] + 10, feature.pt[1] + 20)