from ..feature import Feature
from typing import Iterator

class ImageFeatures:
    def __init__(self):
        self._features: list[Feature] = []
    
    def add_feature(self, feature : Feature | list[Feature]):
        if isinstance(feature, Feature):
            self._features.append(feature)
        elif isinstance(feature, list):
            self._features += feature
    
    def get_features(self) -> list[Feature]:
        return self._features
    
    def __iter__(self) -> Iterator[Feature]:
        for feature in self._features:
            yield feature

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx) -> Feature:
        return self._features[idx]
    
    def __setitem__(self, idx, value) -> Feature:
        self._fea

class ImageFeatureSequence:
    def __init__(self, num_related_images):
        self._ref_image_features = ImageFeatures()
        self._rel_image_features = [ImageFeatures() for _ in range(num_related_images)]
        self._all = [self._ref_image_features] + self._rel_image_features
    
    @property
    def ref_image(self) -> ImageFeatures:
        return self._ref_image_features

    def rel_image(self, rel_image_idx) -> ImageFeatures:
        return self._rel_image_features[rel_image_idx]
    
    @property
    def rel_images(self) -> list[ImageFeatures]:
        return self._rel_image_features

    def __iter__(self) -> Iterator[ImageFeatures]:
        for image_features in self._all:
            yield image_features

    def __len__(self):
        return len(self._rel_image_features)

    def __getitem__(self, idx) -> ImageFeatures:
        return self._all[idx]

class ImageFeatureSet:
    def __init__(self, num_sequences, num_related_images):
        self._sequences: list[ImageFeatureSequence] = [ImageFeatureSequence(num_related_images) for _ in range(num_sequences)]

    def __len__(self):
        return len(self._sequences)

    def __iter__(self) -> Iterator[ImageFeatureSequence]:
        for sequence in self._sequences:
            yield sequence

    def __getitem__(self, idx) -> ImageFeatureSequence:
        return self._sequences[idx]