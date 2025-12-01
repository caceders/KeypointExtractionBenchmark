from ..feature import Feature
from sklearn.metrics import average_precision_score
from typing import Callable
from typing import Iterator
import cv2
import numpy as np

class Match:
    '''
    A Match object representing the match between two features.

    Attributes
    ----------
    feature1 : Feature
        The first feature of the match.
    feature2 : Feature
        The second feature of the match.
    score : float
        The match score, a product of the matching approach.
    is_correct : bool
        Whether this match is correct
    is_in_same_sequece : bool
        Wheter or not the features are in the same sequence.
    is_in_same_image : bool
        Wheter or not the features are in the same image.
    match_properties : dict[str, int | float]
        A dictionary of custom properties. An example is match.match_properties["averge_ratio"]
        to get the average ratio of the match.

    '''
    def __init__(self, feature1: Feature, feature2: Feature, score: float = 0):
            self.feature1 : Feature = feature1
            self.feature2 : Feature = feature2
            self.score : float = score
            self.is_correct : bool = feature1.is_match_with_other_valid(feature2)
            self.is_in_same_sequece : bool = feature1.sequence_index == feature2.sequence_index
            self.is_in_same_image : bool = (feature1.sequence_index == feature2.sequence_index) and (feature1.image_index == feature2.image_index)
            self.match_properties : dict[str, int | float] = {}



class MatchRankingProperty: 
    '''
    An container for a match ranking property (a match property used to determine ranking for mAP).

    Attributes
    ----------
    name : str
        The name of the match ranking property.
    higher_is_better : bool
        If true then high values are considered better. If false then low values are considered better.
    '''
    def __init__(self, name: str, higher_is_better: bool):
        self.name : str = name
        self.higher_is_better : bool = higher_is_better



class MatchSet:
    """
    A container for a collection of matches.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    Features can be added one at a time or in batches, and the internal list
    can be retrieved as a copied list to prevent external modification.
    """
    def __init__(self):
        self._matches: list[Match] = []
    
    def add_match(self, match : Match | list[Match]):
        if isinstance(match, Match):
            self._matches.append(match)
        elif isinstance(match, list):
            self._matches += match

    def get_average_precision_score(self, match_rank_property: MatchRankingProperty, ignore_same_sequence: bool = False) -> float:

        labels = [(1 if match.is_correct else 0) for match in self._matches]
        scores = []
        # If no true labels reutrn 0
        if 1 not in labels:
            return 0.

        if match_rank_property.higher_is_better:
            scores = [match.match_properties[match_rank_property.name] for match in self._matches]
        else:
            scores = [ - match.match_properties[match_rank_property.name] for match in self._matches]

        if ignore_same_sequence:
            for match_index, match in enumerate(self._matches):
                if match.is_in_same_sequece:
                    labels.remove(match_index)
                    scores.remove(match_index)
        
        return float(average_precision_score(labels, scores))
    
    def __iter__(self) -> Iterator[Match]:
        for match_sequence in self._matches:
            yield match_sequence

    def __len__(self):
        return len(self._matches)

    def __getitem__(self, index) -> Match:
        return self._matches[index]
    
    


def homographic_optimal_matching(features1: list[Feature], features2: list[Feature], homography1to2: np.ndarray) -> list[Match]:
    """
    Compute the optimal homographic matching between two sets of features with apriori knowledge about
    the homografic transformation between the two images the features were taken from. Uses a greedy
    maximum bipartite matching algorithm.

    Parameters
    ----------
    features1 : list[Feature]
        The list of features from the first image.
    features2 : list[Feature]
        The list of features from the second image.
    homography1to2: np.ndarray
        The homographic transformation matrix between the two images

    Returns
    -------
    list[match]
        The homographical optimal matching based on the homographic transformation between the images
    """

    feature1_pts = np.array([feature.pt for feature in features1])
    feature2_transformed_pts = np.array([feature.get_pt_after_homography_transform(homography1to2)
                        for feature in features2])

    # Cover empty case
    if len(feature1_pts) == 0 or len(feature2_transformed_pts) == 0:
        return []
    
    # Compute the distace matrix
    dists = np.linalg.norm(feature1_pts[:, None, :] - feature2_transformed_pts[None, :, :], axis=2)

    # Create distance pairs and sort by distance
    pairs = [(i, j, dists[i, j]) for i in range(dists.shape[0]) for j in range(dists.shape[1])]
    pairs.sort(key=lambda x: x[2])

    # Greedy one-to-one matching
    matches: list[Match] = []
    matched_feature1_indexes = set()
    matched_feature2_indexes = set()

    for i, j, dist in pairs:
        if i not in matched_feature1_indexes and j not in matched_feature2_indexes:
            matches.append(Match(features1[i], features2[j], dist))
            matched_feature1_indexes.add(i)
            matched_feature2_indexes.add(j)

    return matches



def greedy_maximum_bipartite_matching(features1: list[Feature], features2: list[Feature], distance_type: int) -> list[Match]:
    """
    Compute the greedy maximum bipartite matching between two sets of features based on the descriptor
    distance between the two.

    Parameters
    ----------
    features1 : list[Feature]
        The list of features from the first image.
    features2 : list[Feature]
        The list of features from the second image.

    Returns
    -------
    list[match]
        The greedy maximum bipartite matching between the images
    """
    if not features1 or not features2:
        return []

    feature1_descriptions = np.array([f.description for f in features1])
    feature2_descriptions = np.array([f.description for f in features2])

    N = len(features1)
    M = len(features2)

    # Compute full distance matrix once
    if distance_type == cv2.NORM_L2:
        diffs = feature1_descriptions[:, None, :] - feature2_descriptions[None, :, :]
        distance_matrix = np.linalg.norm(diffs, axis=2)  # shape (N, M)
    elif distance_type == cv2.NORM_HAMMING:
        xor = np.bitwise_xor(feature1_descriptions[:, None, :], feature2_descriptions[None, :, :])
        distance_matrix = np.unpackbits(xor, axis=2).sum(axis=2)
    else:
        raise TypeError(f"Unknown distance type: {distance_type}")

    # Cover singular element cases
    if N == 1:
        f1 = features1[0]
        dists = distance_matrix[0]
        closest_idx, second_closest_idx = np.argpartition(dists, 2)[:2]
        closest_distance = dists[closest_idx]
        second_closest_distance = dists[second_closest_idx]
        closest_feature = features2[closest_idx]

        match = Match(closest_feature, f1, closest_distance)
        match.match_properties["distance"] = closest_distance
        match.match_properties["average_response"] = (closest_feature.keypoint.response + f1.keypoint.response) / 2
        match.match_properties["average_ratio"] = closest_distance / second_closest_distance
        return [match]

    if M == 1:
        f2 = features2[0]
        dists = distance_matrix[:, 0]
        closest_idx, second_closest_idx = np.argpartition(dists, 2)[:2]
        closest_distance = dists[closest_idx]
        second_closest_distance = dists[second_closest_idx]
        closest_feature = features1[closest_idx]

        match = Match(closest_feature, f2, closest_distance)
        match.match_properties["distance"] = closest_distance
        match.match_properties["average_response"] = (closest_feature.keypoint.response + f2.keypoint.response) / 2
        match.match_properties["average_ratio"] = closest_distance / second_closest_distance
        return [match]

    # Non-singular cases
    i_indices, j_indices = np.nonzero(np.ones_like(distance_matrix))  # generate all pairs
    pairs = [(i, j, distance_matrix[i, j]) for i, j in zip(i_indices, j_indices)]
    pairs.sort(key=lambda p: p[2])

    matched_feature1_indexes = set()
    matched_feature2_indexes = set()
    feature1_to_match: dict[int, Match] = {}
    feature2_to_match: dict[int, Match] = {}
    matches: list[Match] = []

    for i, j, dist in pairs:
        if i not in matched_feature1_indexes and j not in matched_feature2_indexes:
            match = Match(features1[i], features2[j], dist)
            matches.append(match)
            matched_feature1_indexes.add(i)
            matched_feature2_indexes.add(j)
            feature1_to_match[i] = match
            feature2_to_match[j] = match
            match.match_properties["distance"] = dist
            match.match_properties["average_response"] = (features1[i].keypoint.response + features2[j].keypoint.response) / 2
            match.match_properties["average_ratio"] = 0.0  # to be updated

    # Feature1 -> Feature2: closest alternative
    for i, match in feature1_to_match.items():
        distances = distance_matrix[i, :]
        matched_j = features2.index(match.feature2)
        mask = np.arange(M) != matched_j
        second_distance = distances[mask].min()
        match.match_properties["average_ratio"] = match.match_properties["distance"] / second_distance

    # Feature2 -> Feature1: closest alternative
    for j, match in feature2_to_match.items():
        distances = distance_matrix[:, j]
        matched_i = features1.index(match.feature1)
        mask = np.arange(N) != matched_i
        second_distance = distances[mask].min()
        match.match_properties["average_ratio"] += match.match_properties["distance"] / second_distance
        match.match_properties["average_ratio"] /= 2

    return matches