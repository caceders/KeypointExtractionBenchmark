from benchmark.image_feature_set import *
from benchmark.feature import Feature
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation


NUM_SEQUENCES = 116
NUM_RELATED_IMAGES = 5

@pytest.fixture()
def sample_feature() -> Feature:
    kp = cv2.KeyPoint(2, 2, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 1)


@pytest.fixture()
def sample_features_1():
    features = []
    for i in range(60):
        kp = cv2.KeyPoint(100 + i, 200 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 1, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_features_2():
    features = []
    for i in range(30):
        kp = cv2.KeyPoint(200 + i, 300 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 2, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_image_feature_set(sample_features_1, sample_features_2) -> ImageFeatureSet:
    image_feature_set = ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)
    for feature in sample_features_1:
        image_feature_set[0][feature.image_index].append(feature)
    for feature in sample_features_2:
        image_feature_set[1][feature.image_index].append(feature)
    return image_feature_set




### Test that invalid arguments fail ###


@pytest.mark.parametrize("num_sequences, num_related_images",
                        [(None, 5),
                        (106, None)],
                        ids=["All but number of related sequences",
                             "All but number of related images"]
                            )
def test_invalid_arguments_constructor(num_sequences, num_related_images):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        ImageFeatureSet(num_sequences, num_related_images)



### Test that valid arguments pass ###


def test_valid_arguments_constructor():
    ImageFeatureSet(NUM_SEQUENCES, NUM_RELATED_IMAGES)


def test_get_features(sample_image_feature_set):
    sample_image_feature_set.get_features()



### Test that general cases behave expectedly ###

def test_itteration(sample_image_feature_set, sample_features_1, sample_features_2):
    for image_feature_sequence_index, image_feature_sequence in enumerate(sample_image_feature_set):
        if image_feature_sequence_index == 0:
            assert len(image_feature_sequence.get_features()) == len(sample_features_1)
        elif image_feature_sequence_index == 1:
            assert len(image_feature_sequence.get_features()) == len(sample_features_2)
        else:
            assert len((image_feature_sequence.get_features())) == 0


def test_get_features_expected_length(sample_image_feature_set, sample_features_1, sample_features_2):
    all_features = sample_image_feature_set.get_features()
    assert(len(all_features) == len(sample_features_1) + len (sample_features_2))