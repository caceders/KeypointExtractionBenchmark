from benchmark.matching import Match, MatchSet, MatchRankingProperty
from benchmark.feature import Feature
import cv2
import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation


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
def sample_feature_3() -> Feature:
    kp = cv2.KeyPoint(100, 200, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 2, 2)


@pytest.fixture()
def sample_match_rank_property_higher_is_better():
    return MatchRankingProperty("testproperty", True)

@pytest.fixture()
def sample_match_rank_property_lower_is_better():
    return MatchRankingProperty("testproperty", False)

@pytest.fixture()
def sample_match(sample_feature_1, sample_feature_2):
    match = Match(sample_feature_1, sample_feature_2)
    match.match_properties["testproperty"] = 100
    return Match(sample_feature_1, sample_feature_2)


@pytest.fixture()
def sample_match_correct_same_sequence(sample_feature_1, sample_feature_2):
    sample_feature_1.store_valid_match_for_image(1, sample_feature_2, 20)
    match = Match(sample_feature_1, sample_feature_2)
    return match


@pytest.fixture()
def sample_match_correct_different_sequence(sample_feature_1, sample_feature_3):
    sample_feature_1.store_valid_match_for_image(1, sample_feature_3, 20)
    match = Match(sample_feature_1, sample_feature_3)
    return match

@pytest.fixture()
def sample_match_set():
    match_set = MatchSet()
    return match_set



### Test that invalid arguments fail ###

@pytest.mark.parametrize("bad_argument", 
                         ["None",
                          "Single element"],
                        ids = [
                        "Bad argument None",
                        "Bad argument single element None"
                        ])
def test_invalid_argument_add_match(sample_match_set, sample_match, bad_argument):
    if bad_argument == "None":
        argument = None
    elif bad_argument == "Single element":
        argument = ["Something else", sample_match]

    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_match_set.add_match(argument)


@pytest.mark.parametrize("match_rank_property, ignore_negatives_in_same_sequence",
                         [
                            (None, True),
                            (MatchRankingProperty("testproperty", True), None)
                         ],
                         ids=[
                            "Bad argument match rank property",
                            "Bad argument ignore_negative_in_same_sequence"
                         ])
def test_invalid_argument_get_average_precision_score(sample_match_set, match_rank_property, ignore_negatives_in_same_sequence):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_match_set.get_average_precision_score(match_rank_property, ignore_negatives_in_same_sequence)



## Test that valid arguments pass ##


def test_valid_argument_add_match(sample_match_set, sample_match, sample_match_correct_different_sequence):
    sample_match_set.add_match(sample_match)
    sample_match_set.add_match([sample_match_correct_different_sequence, sample_match_correct_different_sequence])
 

def test_valid_argument_get_average_precision_score(sample_match_set, sample_match_rank_property_higher_is_better):
    sample_match_set.get_average_precision_score(sample_match_rank_property_higher_is_better, True)



### Test that general cases behave expectedly ###

@pytest.fixture()
def sample_matches():
    for feature_index in range(22):
        if feature_index < 11:
            sequence = 2
        else:
            sequence = 3
        

# Matches:
# Match_num | is_correct | same_sequence | score
#    1           False         False         1 
#    2           False         True          2 
#    3           True          False         3 
#    4           True          True          4 
#    5           False         False         5 
#    6           True          False         6 
#    7           True          False         7 
#    8           False         True          8 
#    9           True          False         9 
#    10          True          False         10
#    11          True          False         11
#
#
# Ranked list basic higher is better = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
# Ranked list ignore same sequence higher is better = [1, 1, 1, 1, 1, 0, 1, 1, 0]
# Ranked list basic lower is better = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
# Ranked list ignore same sequence lower is better = [0, 1, 1, 0, 1, 1, 1, 1, 1]

def create_matches():
    rows = [
    # (is_correct, same_sequence, score)
    (False, False, 1),
    (False, True, 2),
    (True,  False, 3),
    (True,  True,  4),
    (False, False, 5),
    (True,  False, 6),
    (True,  False, 7),
    (False, True,  8),
    (True,  False, 9),
    (True,  False, 10),
    (True,  False, 11),
    ]

    matches = []

    for is_correct, same_sequence, score in rows:

        f1 = Feature(cv2.KeyPoint(1,2,3), np.ones(3), 1, 1)
        f2 = Feature(cv2.KeyPoint(1,2,3), np.ones(3), (1 if same_sequence else 2), 1) 

        if is_correct:
            f1.store_valid_match_for_image(1, f2, score)
            f2.store_valid_match_for_image(1, f1, score)

        match = Match(f1, f2)

        match.match_properties["testproperty"] = score

        matches.append(match)
    return matches

def average_precision(ranked_list: list):
    sum_precision = 0
    correct = 0
    total = 0
    for element in ranked_list:
        total += 1
        if element == 1:
            correct += 1
            sum_precision += correct/total
    AP = sum_precision/correct
    return AP

def test_correct_AP_score_basic_higher_is_better(sample_match_set, sample_match_rank_property_higher_is_better):
    matches = create_matches()
    sample_match_set.add_match(matches)
    expected_AP = average_precision([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0])
    actual_AP = sample_match_set.get_average_precision_score(sample_match_rank_property_higher_is_better)

    assert np.isclose(expected_AP, actual_AP)


def test_correct_AP_score_ignore_same_sequence_higher_is_better(sample_match_set, sample_match_rank_property_higher_is_better):
    matches = create_matches()
    sample_match_set.add_match(matches)
    expected_AP = average_precision([1, 1, 1, 1, 1, 0, 1, 1, 0])
    actual_AP = sample_match_set.get_average_precision_score(sample_match_rank_property_higher_is_better, True)

    assert np.isclose(expected_AP, actual_AP)


def test_correct_AP_score_basic_lower_is_better(sample_match_set, sample_match_rank_property_lower_is_better):
    matches = create_matches()
    sample_match_set.add_match(matches)
    expected_AP = average_precision([0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1])
    actual_AP = sample_match_set.get_average_precision_score(sample_match_rank_property_lower_is_better)

    assert np.isclose(expected_AP, actual_AP)


def test_correct_AP_score_ignore_same_sequence_lower_is_better(sample_match_set, sample_match_rank_property_lower_is_better):
    matches = create_matches()
    sample_match_set.add_match(matches)
    expected_AP = average_precision([0, 1, 1, 0, 1, 1, 1, 1, 1])
    actual_AP = sample_match_set.get_average_precision_score(sample_match_rank_property_lower_is_better, True)

    assert np.isclose(expected_AP, actual_AP)