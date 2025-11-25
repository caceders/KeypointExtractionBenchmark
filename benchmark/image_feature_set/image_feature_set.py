from ..feature import Feature
from typing import Iterator

class ImageFeatures:
    """
    A container for a collection of features related to a single image.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    Features can be added one at a time or in batches, and the internal list
    can be retrieved as a copied list to prevent external modification.
    """
    def __init__(self):
        self._features: list[Feature] = []
    
    def add_feature(self, feature : Feature | list[Feature]):
        if isinstance(feature, Feature):
            self._features.append(feature)
        elif isinstance(feature, list):
            self._features += feature
    
    def get_features(self) -> list[Feature]:
        return self._features.copy()
    
    def __iter__(self) -> Iterator[Feature]:
        for feature in self._features:
            yield feature

    def __len__(self):
        return len(self._features)

    def __getitem__(self, index) -> Feature:
        return self._features[index]
    
    def __setitem__(self, index, value) -> Feature:
        self._features[index] = value

class ImageFeatureSequence:
    """
    A container for a collection of Imagefeature objects related to a single sequence.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    Features can be added one at a time or in batches, and the internal list
    can be retrieved as a copied list to prevent external modification.
    """
    def __init__(self, num_related_images):
        self._reference_image_features = ImageFeatures()
        self._related_image_features = [ImageFeatures() for _ in range(num_related_images)]
        self._all = [self._reference_image_features] + self._related_image_features
    
    @property
    def reference_image(self) -> ImageFeatures:
        return self._reference_image_features

    def related_image(self, related_image_index) -> ImageFeatures:
        return self._related_image_features[related_image_index]
    
    @property
    def related_images(self) -> list[ImageFeatures]:
        return self._related_image_features.copy()

    def get_features(self) -> list[Feature]:
        features = self._reference_image_features.get_features()
        for related_image in self.related_images:
            features.extend(related_image.get_features())
        return features

    def __iter__(self) -> Iterator[ImageFeatures]:
        for image_features in self._all:
            yield image_features

    def __len__(self):
        return len(self._all)

    def __getitem__(self, index) -> ImageFeatures:
        return self._all[index]
    


class ImageFeatureSet:
    """
    A container for a collection of ImageFeatureSequence objects related to a image-set with sequences.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    Features can be added one at a time or in batches, and the internal list
    can be retrieved as a copied list to prevent external modification.
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
    