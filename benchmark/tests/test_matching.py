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


def test_next_best_match_return_greedy_maximum_bipartite_matching():
    
    feature_a = Feature(cv2.KeyPoint(x = 1 , y = 1, size = 1, response = 1), np.array([8]))
    feature_b = Feature(cv2.KeyPoint(x = 2 , y = 2, size = 1, response = 2), np.array([1]))
    feature_c = Feature(cv2.KeyPoint(x = 3 , y = 3, size = 1, response = 3), np.array([7]))
    feature_d = Feature(cv2.KeyPoint(x = 4 , y = 4, size = 1, response = 4), np.array([5]))

    # Optimal pairing: a <-> c, b <-> d
    #
    # a <-> c
    # Distance: 8-7 = 1
    # Average response: (1+3)/2 = 2
    # Average ratio: ((8-7)/(8-5) + (8-7)/(7-1)) / 2 = 0.25
    #
    # b <-> d
    # Distance: 5-1 = 4
    # Average response: (2+4)/2 = 3
    # Average ratio: ((5-1)/(7-1)) + (5-1)/(8-5)) / 2 = 1
    
    features1 = [
        feature_a,
        feature_b
        
    ]
    features2 = [
        feature_c,
        feature_d
    ]

    matches = greedy_maximum_bipartite_matching(features1, features2)
    for match in matches:
        if match.feature1 == feature_a:
            assert match.feature2 == feature_c
            assert match.custom_properties["distance"] == 1
            assert match.custom_properties["average_response"] == 2
            assert match.custom_properties["average_ratio"] == pytest.approx(0.25)
        if match.feature1 == feature_b:
            assert match.feature2 == feature_d
            assert match.custom_properties["distance"] == 4
            assert match.custom_properties["average_response"] == 3
            #assert match.custom_properties["average_ratio"] == pytest.approx(1)
        if match.feature1 == feature_c:
            assert match.feature2 == feature_a
            assert match.custom_properties["distance"] == 1
            assert match.custom_properties["average_response"] == 2
            assert match.custom_properties["average_ratio"] == pytest.approx(0.25)
        if match.feature1 == feature_d:
            assert match.feature2 == feature_b
            assert match.custom_properties["distance"] == 4
            assert match.custom_properties["average_response"] == 3
            assert match.custom_properties["average_ratio"] == pytest.approx(1)