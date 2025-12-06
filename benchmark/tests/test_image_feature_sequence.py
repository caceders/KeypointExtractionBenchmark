from benchmark.image_feature_set import *
from benchmark.feature import Feature
import pytest
import cv2
import numpy as np
from typing import Callable, Tuple
import time
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation



NUM_RELATED_IMAGES = 5



@pytest.fixture()
def sample_feature() -> Feature:
    kp = cv2.KeyPoint(2, 2, 1)
    desc = np.ones(128)
    return Feature(kp, desc, 1, 1)


@pytest.fixture()
def sample_features() -> list[Feature]:
    features = []
    for i in range(30):
        kp = cv2.KeyPoint(100 + i, 200 + i, 1)
        desc = np.ones(128) * i
        features.append(Feature(kp, desc, 1, i%NUM_RELATED_IMAGES))
    return features


@pytest.fixture()
def sample_image_feature_sequence(sample_features) -> ImageFeatureSequence:
    image_feature_sequence = ImageFeatureSequence(NUM_RELATED_IMAGES)
    for image_index in range(6):
        image_feature_sequence[image_index] = []
    for feature in sample_features:
        image_feature_sequence[feature.image_index].append(feature)
    return image_feature_sequence



### Test that invalid arguments fail ###


def test_invalid_arguments_constructor():
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        ImageFeatureSequence(None)


def test_invalid_arguments_related_image(sample_image_feature_sequence):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_image_feature_sequence.related_image(None)


def test_invalid_arguments_set_item(sample_image_feature_sequence):
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        sample_image_feature_sequence[3] = None



### Test that valid arguments pass ###


def test_valid_arguments_constructor():
    ImageFeatureSequence(NUM_RELATED_IMAGES)


def test_valid_arguments_reference_image(sample_image_feature_sequence):
    sample_image_feature_sequence.reference_image


def test_valid_arguments_related_image(sample_image_feature_sequence):
    sample_image_feature_sequence.related_image(NUM_RELATED_IMAGES//2)


def test_valid_arguments_related_images(sample_image_feature_sequence):
    sample_image_feature_sequence.related_images


def test_valid_arguments_set_item(sample_image_feature_sequence, sample_features):
    sample_image_feature_sequence[NUM_RELATED_IMAGES//2] = sample_features



### Test that general cases behave expectedly ###


def test_store_feature_in_reference_image(sample_image_feature_sequence, sample_feature):
    sample_image_feature_sequence.reference_image.append(sample_feature)
    assert sample_feature in sample_image_feature_sequence[0], "A stored feature for the reference image was not found in the first index of the relevant feature sequence"


@pytest.mark.parametrize(
    "image_index",
    [image_index for image_index in range(NUM_RELATED_IMAGES)],
    ids=[str(image_index) for image_index in range(NUM_RELATED_IMAGES)]
)
def test_store_feature_in_related_image(sample_image_feature_sequence, sample_feature, image_index):
    sample_image_feature_sequence.related_image(image_index).append(sample_feature)
    assert sample_feature in sample_image_feature_sequence[image_index + 1], "A stored feature for the related image was not found in the relevant index for the related image"


def test_all_features_in_get_features(sample_image_feature_sequence, sample_features):
    all_features = sample_image_feature_sequence.get_features()
    assert len(all_features) == len(all_features), "sample_features fixture used to set the features, but their sizes was not equal"


def test_all_features_in_refrence_image(sample_image_feature_sequence, sample_features):
    check_image_features = []
    for feature in sample_features:
        if feature.image_index == 0:
            check_image_features.append(feature)

    assert len(sample_image_feature_sequence.reference_image) == len(check_image_features), "The amount of features in the reference image should be equal to the amount of features with index 0"


def test_all_features_in_related_image(sample_image_feature_sequence, sample_features):
    assert len(sample_image_feature_sequence.related_images) == NUM_RELATED_IMAGES, "There should be in total 5 related images"
    for related_image_index in range(len(sample_image_feature_sequence.related_images)):
        assert len(sample_image_feature_sequence.related_images[related_image_index]) == np.sum([1 if feature.image_index == (related_image_index + 1) else 0 for feature in sample_features]), "The amount of features in each related image should be equal to the amount of features with that related image's index as their index"


def test_itteration(sample_image_feature_sequence):
    itterations = 0
    for image_features_index, image_features in enumerate(sample_image_feature_sequence):
        for feature in image_features:
            assert feature.image_index == image_features_index
        itterations += 1
    
    assert itterations == NUM_RELATED_IMAGES + 1, "When itterating we should itterate a number of times equal to the amount of related images pluss the reference image"