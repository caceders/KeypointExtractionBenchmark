from benchmark.matching import greedy_maximum_bipartite_matching_descriptor_distance, greedy_maximum_bipartite_matching
from benchmark.feature import Feature
from benchmark.utils import calculate_overlap_one_circle_to_many
import numpy as np
import cv2
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation


@pytest.fixture()
def feature_set_1() -> list[Feature]:
    feats = []
    for x in range(50):
        for y in range(50):
            kp = cv2.KeyPoint(x, y, 1)
            desc = np.ones(128) * (y + 50 * x)
            feats.append(Feature(kp, desc, 1, 1))
    
    return feats

@pytest.fixture()
def feature_set_2() -> list[Feature]:
    feats = []
    for x in range(50):
        for y in range(50):
            kp = cv2.KeyPoint(x, y, 1)
            desc = np.ones(128) * (y + 50 * x)
            feats.append(Feature(kp, desc, 1, 2))
    
    return feats

def create_valid_pairings(features1 : list[Feature], image_1_index : int, features2 : list[Feature], image_2_index : int):
    for feature_index in range(len(features1)):
        features1[feature_index].store_valid_match_for_image(image_1_index, features2[feature_index], 1)
        features2[feature_index].store_valid_match_for_image(image_2_index, features1[feature_index], 1)

def test_greedy_maximum_bipartite_matching(feature_set_1, feature_set_2):
    create_valid_pairings(feature_set_1, 0, feature_set_2, 1)
    
    feature1_positions = np.array([f.pt for f in feature_set_1])
    feature2_positions = np.array([f.pt for f in feature_set_2])

    # Compute full distance matrix once
    distances = np.linalg.norm(feature2_positions - feature1_positions, axis=1)
    overlaps = []
    for feature in feature_set_1:
        overlap1, overlap2 = calculate_overlap_one_circle_to_many(feature.keypoint.size, [feature.keypoint.size for feature in feature_set_2], distances)
        overlaps.append(np.minimum(overlap1, overlap2))
    scores = np.asarray(overlaps)
    matches = greedy_maximum_bipartite_matching(feature_set_1, feature_set_2, scores)
    assert len(matches) == len(feature_set_1)
    for match in matches:
        assert np.array_equal(match.feature1.pt, match.feature2.pt) ## Check that the match actualy got the same point



def test_greedy_maximum_bipartite_matching_descriptor_distance(feature_set_1, feature_set_2):
    matches = greedy_maximum_bipartite_matching_descriptor_distance(feature_set_1, feature_set_2, cv2.NORM_L2)
    for match in matches:
        assert np.array_equal(match.feature1.pt, match.feature2.pt) ## Check that the match actualy got the identical descriptors


def test_next_best_match_return_greedy_maximum_bipartite_matching():
    
    feature_a = Feature(cv2.KeyPoint(x = 1 , y = 1, size = 1, response = 1), np.array([8]), 1, 1)
    feature_b = Feature(cv2.KeyPoint(x = 2 , y = 2, size = 1, response = 2), np.array([1]), 1, 1)
    feature_c = Feature(cv2.KeyPoint(x = 3 , y = 3, size = 1, response = 3), np.array([7]), 1, 2)
    feature_d = Feature(cv2.KeyPoint(x = 4 , y = 4, size = 1, response = 4), np.array([5]), 1, 2)

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

    matches = greedy_maximum_bipartite_matching_descriptor_distance(features1, features2, cv2.NORM_L2)
    for match in matches:
        if match.feature1 == feature_a:
            assert match.feature2 == feature_c
            assert match.match_properties["distance"] == 1
            assert match.match_properties["average_response"] == 2
            assert match.match_properties["average_ratio"] == pytest.approx(0.25)
        if match.feature1 == feature_b:
            assert match.feature2 == feature_d
            assert match.match_properties["distance"] == 4
            assert match.match_properties["average_response"] == 3
            assert match.match_properties["average_ratio"] == pytest.approx(1)
        if match.feature1 == feature_c:
            assert match.feature2 == feature_a
            assert match.match_properties["distance"] == 1
            assert match.match_properties["average_response"] == 2
            assert match.match_properties["average_ratio"] == pytest.approx(0.25)
        if match.feature1 == feature_d:
            assert match.feature2 == feature_b
            assert match.match_properties["distance"] == 4
            assert match.match_properties["average_response"] == 3
            assert match.match_properties["average_ratio"] == pytest.approx(1)