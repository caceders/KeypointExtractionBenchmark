from ..feature import Feature
from typing import Iterator
from beartype import beartype

class ImageFeatureSequence:
    """
    ImageFeatureSet contains **ImageFeatureSequences** contains a list of features.

    A container for a collection of Imagefeature objects related to a single sequence.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    """
    #@beartype
    def __init__(self, num_related_images: int):
        self.reference_image_features : list[Feature] = []
        self.related_images_features : list[list[Feature]] = [[] for _ in range(num_related_images)]
        self._all = [self.reference_image_features] + self.related_images_features

    def __iter__(self) -> Iterator[list[Feature]]:
        for image_features in self._all:
            yield image_features

    def __len__(self):
        return len(self._all)

    def __getitem__(self, index) -> list[Feature]:
        return self._all[index]
    
    #@beartype
    def __setitem__(self, index, value : list[Feature]):
        self._all[index] = value

        # Also set reference and related image features
        if index == 0:
            self.reference_image_features = value
        else:
            self.related_images_features[index - 1] = value
        


class ImageFeatureSet:
    """
    **ImageFeatureSet** contains ImageFeatureSequences contains a list of features.

    A container for a collection of ImageFeatureSequence objects related to an image-set with sequences.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    """
    #@beartype
    def __init__(self, num_sequences, num_related_images):
        self._sequences: list[ImageFeatureSequence] = [ImageFeatureSequence(num_related_images) for _ in range(num_sequences)]

    def __len__(self):
        return len(self._sequences)

    def __iter__(self) -> Iterator[ImageFeatureSequence]:
        for sequence in self._sequences:
            yield sequence

    def __getitem__(self, index) -> ImageFeatureSequence:
        return self._sequences[index]
    