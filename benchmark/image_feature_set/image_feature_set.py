from ..feature import Feature
from typing import Iterator

class ImageFeatureSequence:
    """
    ImageFeatureSet contains **ImageFeatureSequences** contains a list of features.

    A container for a collection of Imagefeature objects related to a single sequence.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    """
    def __init__(self, num_related_images):
        self._reference_image_features = []
        self._related_image_features = [[] for _ in range(num_related_images)]
        self._all = [self._reference_image_features] + self._related_image_features
    
    @property
    def reference_image(self) -> list[Feature]:
        return self._reference_image_features

    def related_image(self, related_image_index) -> list[Feature]:
        return self._related_image_features[related_image_index]
    
    @property
    def related_images(self) -> list[list[Feature]]:
        return self._related_image_features.copy()

    def get_features(self) -> list[Feature]:
        features = self._reference_image_features
        for related_image in self.related_images:
            features.extend(related_image)
        return features

    def __iter__(self) -> Iterator[list[Feature]]:
        for image_features in self._all:
            yield image_features

    def __len__(self):
        return len(self._all)

    def __getitem__(self, index) -> list[Feature]:
        return self._all[index]
    
    def __setitem__(self, index, value):
        if not isinstance(value, list) or (len(value) != 0 and not all(isinstance(feature, Feature) for feature in value)): raise TypeError("ImageFeatureSequence elements can only be list[Feature]")
        self._all[index] = value

        # Also set reference and related image features
        if index == 0:
            self._reference_image_features = value
        else:
            self._related_image_features[index - 1] = value
        


class ImageFeatureSet:
    """
    **ImageFeatureSet** contains ImageFeatureSequences contains a list of features.

    A container for a collection of ImageFeatureSequence objects related to an image-set with sequences.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    """
    def __init__(self, num_sequences, num_related_images):
        self._sequences: list[ImageFeatureSequence] = [ImageFeatureSequence(num_related_images) for _ in range(num_sequences)]

    def get_features(self) -> list[Feature]:
        features = []
        for image_feature_sequence in self._sequences:
            features.extend(image_feature_sequence.get_features())
        return features

    def __len__(self):
        return len(self._sequences)

    def __iter__(self) -> Iterator[ImageFeatureSequence]:
        for sequence in self._sequences:
            yield sequence

    def __getitem__(self, index) -> ImageFeatureSequence:
        return self._sequences[index]
    