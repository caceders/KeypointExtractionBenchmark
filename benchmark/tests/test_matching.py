from benchmark.matching import Match, homographic_optimal_matching, greedy_maximum_bipartite_matching
from benchmark.feature import Feature
import numpy as np
import cv2
import pytest

@pytest.fixture()
def feature_set_1() -> list[Feature]:
    feats = []
    for x in range(50):
        for y in range(50):
            kp = cv2.KeyPoint(x, y, 1)
            desc = np.ones(128) * (y + 50 * x)
            feats.append(Feature(kp, desc))
    
    return feats

@pytest.fixture()
def feature_set_2() -> list[Feature]:
    feats = []
    for x in range(50):
        for y in range(50):
            kp = cv2.KeyPoint(x, y, 1)
            desc = np.ones(128) * (y + 50 * x)
            feats.append(Feature(kp, desc))
    
    return feats



def test_homographical_optimal_matching(feature_set_1, feature_set_2):
    matches = homographic_optimal_matching(feature_set_1, feature_set_2, np.eye(3))
    for match in matches:
        assert match.feature1.pt == match.feature2.pt ## Check that the match actualy got the same point



def test_greedy_maximum_bipartite_matching(feature_set_1, feature_set_2):
    matches = homographic_optimal_matching(feature_set_1, feature_set_2, np.eye(3))
    for match in matches:
        assert match.feature1.pt == match.feature2.pt ## Check that the match actualy got the identical descriptors

