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
    custom_properties : dict[str]
        A dictionary of custom properties. An example is match.custom_property["averge_ratio"]
        to get the average ratio of the match.

    '''
    def __init__(self, feature1: Feature, feature2: Feature, score: float = 0):
            self.feature1 : Feature = feature1
            self.feature2 : Feature = feature2
            self.score : float = score
            self.is_correct : bool = feature1.is_match_with_other_valid(feature2)
            self.is_in_same_sequece : bool = feature1.sequence_index == feature2.sequence_index
            self.is_in_same_image : bool = (feature1.sequence_index == feature2.sequence_index) and (feature1.image_index == feature2.image_index)
            self.custom_properties : dict[str] = {}



class MatchRankProperty:
    '''
    An container for a match rank property (a match property used to calculate ranking for mAP).

    Attributes
    ----------
    name : str
        The name of the match rank property.
    ascending : bool
        If true then high values are considered better. If false then low values are considered better.
    '''
    def __init__(self, name: str, ascending: bool):
        self.name : str = name
        self.ascending : bool = ascending



class MatchSequence:
    """
    A container for a collection of matches related to a single sequence.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    Features can be added one at a time or in batches, and the internal list
    can be retrieved as a copied list to prevent external modification.
    """
    def __init__(self):
        self._matches: list[Match] = []

    def get_average_precision_score(self, match_rank_property: MatchRankProperty, ignore_same_sequence: bool = False) -> float:

        labels = [(1 if match.is_correct else 0) for match in self._matches]
        scores = []
        # If no true labels reutrn 0
        if 1 not in labels:
            return 0.

        if match_rank_property.ascending:
            scores = [match.custom_properties[match_rank_property.name] for match in self._matches]
        else:
            # Add small constant to avoid division by zero
            epsilon = 1e-12
            scores = [1/(match.custom_properties[match_rank_property.name] + epsilon)  for match in self._matches]

        if ignore_same_sequence:
            for match_index, match in enumerate(self._matches):
                if match.is_in_same_sequece:
                    labels.remove(match_index)
                    scores.remove(match_index)
        
        return float(average_precision_score(labels, scores))
    
    def add_match(self, match : Match | list[Match]):
        if isinstance(match, Match):
            self._matches.append(match)
        elif isinstance(match, list):
            self._matches += match
    
    def __iter__(self) -> Iterator[Match]:
        for match_sequence in self._matches:
            yield match_sequence

    def __len__(self):
        return len(self._matches)

    def __getitem__(self, index) -> Match:
        return self._matches[index]



class MatchSet:
    """
    A container for a collection of match sequences related to a image-set with sequences.
    The container behaves like a Python list.
    It supports iteration, indexing, assignment, len(), and item access.
    Features can be added one at a time or in batches, and the internal list
    can be retrieved as a copied list to prevent external modification.
    """
    def __init__(self, num_sequences: int):
        self._num_sequences = num_sequences
        self._match_sequences: list[MatchSequence] = [MatchSequence() for i in range(num_sequences)]
    
    def __iter__(self) -> Iterator[MatchSequence]:
        for match_sequence in self._match_sequences:
            yield match_sequence

    def __len__(self):
        return len(self._match_sequences)

    def __getitem__(self, index) -> MatchSequence:
        return self._match_sequences[index]



class MatchingApproach:
    """
    Wrapper class for a matching approach. Wraps around a matching algorithm callback.
    Also works as a container for the relevant match rank properties.

    Attributes
    ----------
    matching_callback : Callable[[list[Feature], list[Feature], int], list[Match]]
        A callback to the matching algorithm. The matching algorithm needs to take in two lists of features
        and produce a list of matches. The matching algorithm should make any neccessary additions to the
        relevant match objects custom property dictionary.
    match_rank_properties : list[MatchRankProperty]
        A list of the match rank properties this matching approach adds.
    """
    def __init__(self,
                matching_callback: Callable[[list[Feature], list[Feature], int], list[Match]],
                match_rank_properties = list[MatchRankProperty]):
        self.matching_callback = matching_callback
        self.match_rank_properties = match_rank_properties



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

    ## Cover empty case
    if not features1 or not features2:
        return []
    
    feature1_descriptions = np.stack([feature.description for feature in features1])
    feature2_descriptions = np.stack([feature.description for feature in features2])


    N = len(features1)
    M = len(features2)
    
    ## Cover singular element cases (needed for performance increase)
    if N == 1:
        # Match the single feature1 to the best feature2
        f1 = features1[0]

        if distance_type == cv2.NORM_L2:
            diffs = feature2_descriptions - f1.description
            dists = np.linalg.norm(diffs, axis=1)
        else:
            xor = np.bitwise_xor(feature2_descriptions, f1.description)
            dists = np.unpackbits(xor, axis=1).sum(axis=1)

        index_and_distances = []
        for j in range(M):
            index_and_distances.append((j, dists[j]))

        index_and_distances.sort(key=lambda tuple: tuple[1])
       
        closest_feature_index, closest_distance = index_and_distances[0]
        _, second_closest_distance = index_and_distances[1]

        closest_feature = features2[closest_feature_index]

        match = Match(closest_feature, f1, closest_distance)
        match.custom_properties["distance"] = closest_distance
        match.custom_properties["average_response"] = (closest_feature.keypoint.response + f1.keypoint.response) / 2

    elif M == 1:
        # Match the single feature2 to the best feature1
        f2 = features2[0]

        if distance_type == cv2.NORM_L2:
            diffs = feature1_descriptions - f2.description
            dists = np.linalg.norm(diffs, axis=1)
        else:
            xor = np.bitwise_xor(feature1_descriptions, f2.description)
            dists = np.unpackbits(xor, axis=1).sum(axis=1)

        index_and_distances = []
        for i in range(N):
            index_and_distances.append((i, dists[i]))

        index_and_distances.sort(key=lambda tuple: tuple[1])
       
        closest_feature_index, closest_distance = index_and_distances[0]
        _, second_closest_distance = index_and_distances[1]

        closest_feature = features1[closest_feature_index]

        match = Match(closest_feature, f2, closest_distance)
        match.custom_properties["distance"] = closest_distance
        match.custom_properties["average_response"] = (closest_feature + f2.keypoint.response) / 2

        # no alternative: ratio = 1
        match.custom_properties["average_ratio"] = closest_distance/second_closest_distance
        return [match]

    
    ## Cover non-singular cases

    # Compute all pairwise distances in batches to avoid memory issues
    pairs = []

    for i in range(N):

        if distance_type == cv2.NORM_L2:
            differences = feature2_descriptions - feature1_descriptions[i]
            distances = np.linalg.norm(differences, axis=1)

        elif distance_type == cv2.NORM_HAMMING:
            xor = np.bitwise_xor(feature2_descriptions, feature1_descriptions[i])
            distances = np.unpackbits(xor, axis=1).sum(axis=1)

        for j in range(M):
            pairs.append((i, j, float(distances[j])))

    # Sort all pairs by ascending distance
    pairs.sort(key=lambda pair_tuple: pair_tuple[2])

    # Do maximum greedy matching ---
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

            match.custom_properties["distance"] = dist
            match.custom_properties["average_response"] = (features1[i].keypoint.response + features2[j].keypoint.response) / 2
            match.custom_properties["average_ratio"] = 0.0 # Fill later when we find closest alternative feature

    # Find closest alternative feature (needs to be done after )
    # Feature1 -> Feature2
    for i, match in feature1_to_match.items():
        feature1_description = feature1_descriptions[i]

        # compute distances
        if distance_type == cv2.NORM_L2:
            distance = np.linalg.norm(feature2_descriptions - feature1_description, axis=1)
        else:
            xor = np.bitwise_xor(feature2_descriptions, feature1_description)
            distance = np.unpackbits(xor, axis=1).sum(axis=1)

        # find the two smallest distances
        two_smallest = np.sort(distance)[:2]  # first two smallest values

        # handle case where matched distance might be the smallest
        if np.isclose(two_smallest[0], match.custom_properties["distance"]):
            second = two_smallest[1]  # second smallest
        else:
            second = two_smallest[0]  # first smallest is not the matched one

        # Set initial average ratio
        match.custom_properties["average_ratio"] = (
            match.custom_properties["distance"] / second
        )

    # Feature2 -> Feature1
    for j, match in feature2_to_match.items():
        feature2_desription = feature2_descriptions[j]

        # compute distances
        if distance_type == cv2.NORM_L2:
            distance = np.linalg.norm(feature1_descriptions - feature2_desription, axis=1)
        else:
            xor = np.bitwise_xor(feature1_descriptions, feature2_desription)
            distance = np.unpackbits(xor, axis=1).sum(axis=1)

        # if there is only one distance then we can't find the next best
        if len(distance) == 1:
            continue

        # find the two smallest distances
        two_smallest = np.partition(distance, 1)[:2]

        # pick the one that is not the matched feature
        if two_smallest[0] == match.custom_properties["distance"]:
            second = two_smallest[1]
        else:
            second = two_smallest[0]

        # update average ratio
        match.custom_properties["average_ratio"] += (match.custom_properties["distance"]/second)
        match.custom_properties["average_ratio"] /= 2

    return matches