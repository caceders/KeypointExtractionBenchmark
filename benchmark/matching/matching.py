from ..feature import Feature
from sklearn.metrics import average_precision_score
from typing import Iterator
import cv2
import numpy as np
from beartype import beartype



class Match:
    '''
    A Match object representing the match between two features.

    Attributes
    ----------
    reference_feature : Feature
        The first feature of the match.
    related_feature : Feature
        The second feature of the match.
    is_correct : bool
        Whether this match is correct
    is_in_same_sequece : bool
        Wheter or not the features are in the same sequence.
    match_properties : dict[str, int | float]
        A dictionary of custom properties. An example is match.match_properties["averge_ratio"]
        to get the average ratio of the match.

    '''
    @beartype
    def __init__(self, reference_feature: Feature, related_feature: Feature):

            self.reference_feature : Feature = reference_feature
            self.related_feature : Feature = related_feature
            self.is_correct : bool = reference_feature.is_match_with_other_valid(related_feature)
            self.is_in_same_sequece : bool = reference_feature.sequence_index == related_feature.sequence_index
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
    
    @beartype
    def add_match(self, matches : Match | list[Match]):

        if isinstance(matches, Match):
            self._matches.append(matches)
        elif isinstance(matches, list):
            if not all(isinstance(match, Match) for match in matches): raise TypeError("All elements in matches needs to be of type Match")
            self._matches += matches
        

    @beartype
    def get_average_precision_score(self, match_ranking_property: MatchRankingProperty, ignore_negatives_in_same_sequence: bool = False) -> float:

        labels = [(1 if match.is_correct else 0) for match in self._matches]
        scores = []
        # If no true labels reutrn 0
        if 1 not in labels:
            return 0.

        if match_ranking_property.higher_is_better:
            scores = [float(match.match_properties[match_ranking_property.name]) for match in self._matches]
        else:
            scores = [- float(match.match_properties[match_ranking_property.name])  for match in self._matches]

        if ignore_negatives_in_same_sequence:

            matches_indices_to_ignore = []
            for match_index, match in enumerate(self._matches):
                if not match.is_correct and match.is_in_same_sequece:
                    matches_indices_to_ignore.append(match_index)

            matches_indices_to_ignore.reverse()
            
            for match_index in matches_indices_to_ignore:
                labels.pop(match_index)
                scores.pop(match_index)
            
        return float(average_precision_score(labels, scores))
    
    def __iter__(self) -> Iterator[Match]:
        for match_sequence in self._matches:
            yield match_sequence

    def __len__(self):
        return len(self._matches)

    def __getitem__(self, index) -> Match:
        return self._matches[index]


@beartype
def greedy_maximum_bipartite_matching(reference_features: list[Feature], related_features: list[Feature], score_matrix: np.ndarray, match_only_if_valid: bool = False) -> list[Match]:
    """
    Compute the greedy maximum bipartite matching between two sets of features based on a score matrix

    Parameters
    ----------
    reference_features : list[Feature]
        The list of features from the first image.
    related_features : list[Feature]
        The list of features from the second image.
    score_matrix : np.ndarray
        The score matrix with scores from reference_features to related_features

    Returns
    -------
    list[match]
        The greedy maximum bipartite matching between the images
    """
    if not reference_features or not related_features:
        return []


    # Cover singular element cases
    if len(reference_features) == 1:
        singular = (reference_features[0], related_features)
    elif len(related_features) == 1:
        singular = (related_features[0], reference_features)
    else:
        singular = None
    
    if singular:
        singular_feature, feature_set = singular
        scores = score_matrix[0]

        if match_only_if_valid: # filter out invalid matches
            mask = np.array([ singular_feature.is_match_with_other_valid(f2) for f2 in feature_set])
            feature_set = [f2 for f2, keep in zip(feature_set, mask) if keep]
            scores = scores[mask]

        best_idx = np.argpartition(scores, 0)[0]
        best_score = scores[best_idx]
        best_feature = feature_set[best_idx]

        if len(scores) >= 2:
            next_best_idx = np.argpartition(scores, 1)[1]
            next_best_score = scores[next_best_idx]
        else:
            next_best_score = float("inf") # There is no next closest feature so simulate this with "infinite" distance.


        match = Match(singular_feature, best_feature)
        match.match_properties["distance"] = best_score
        match.match_properties["average_response"] = (best_feature.keypoint.response + singular_feature.keypoint.response) / 2
        if next_best_score == 0:
            match.match_properties["average_ratio"] = 1
        else:
            match.match_properties["average_ratio"] = best_score / next_best_score
        return [match]
    

    # Non-singular cases
    reference_feature_indices, related_feature_indices = np.nonzero(np.ones_like(score_matrix))  # generate all pairs
    pairs = [(i, j, score_matrix[i, j]) for i, j in zip(reference_feature_indices, related_feature_indices)]

    pairs.sort(key=lambda p: p[2], reverse=True)

    matched_reference_feature_indices = set()
    matched_related_feature_indices = set()
    reference_feature_to_match: dict[int, Match] = {}
    related_feature_to_match: dict[int, Match] = {}
    matches: list[Match] = []

    for reference_feature_index, related_feature_index, dist in pairs:

        singular_feature = reference_features[reference_feature_index]
        related_feature = related_features[related_feature_index]

        feature_already_matched = reference_feature_index in matched_reference_feature_indices or related_feature_index in matched_related_feature_indices

        if feature_already_matched:
            continue

        if match_only_if_valid and not singular_feature.is_match_with_other_valid(related_feature):
            continue

        match = Match(reference_features[reference_feature_index], related_features[related_feature_index])
        matches.append(match)
        matched_reference_feature_indices.add(reference_feature_index)
        matched_related_feature_indices.add(related_feature_index)
        reference_feature_to_match[reference_feature_index] = match
        related_feature_to_match[related_feature_index] = match
        match.match_properties["distance"] = float(dist)
        match.match_properties["average_response"] = (reference_features[reference_feature_index].keypoint.response + related_features[related_feature_index].keypoint.response) / 2
        match.match_properties["average_ratio"] = 0.0  # to be updated


    # Feature1 -> Feature2: best alternative
    for reference_feature_index, match in reference_feature_to_match.items():
        scores = score_matrix[reference_feature_index, :]

        # mask out the already matched feature
        matched_related_feature_index = related_features.index(match.related_feature)
        mask = np.arange(len(related_features)) != matched_related_feature_index

        # additionally mask out invalid matches
        if match_only_if_valid:
            valid_mask = np.array([
                match.reference_feature.is_match_with_other_valid(f2)
                for f2 in related_features
            ])
            mask = mask & valid_mask  # combine masks
        remaining_scores = scores[mask]

        # if no remaining valid features, use inf
        if remaining_scores.size > 0:
            second_best_score = remaining_scores.min()
        else:
            second_best_score = float("inf")

        # compute average_ratio
        if second_best_score == 0:
            match.match_properties["average_ratio"] = 1
        else:
            match.match_properties["average_ratio"] = match.match_properties["distance"] / second_best_score
            
    # Feature2 -> Feature1: best alternative
    for related_feature_index, match in related_feature_to_match.items():
        scores = score_matrix[:, related_feature_index]

        # mask out the already matched feature
        matched_reference_feature_index = reference_features.index(match.reference_feature)
        mask = np.arange(len(reference_features)) != matched_reference_feature_index

        # additionally mask out invalid matches
        if match_only_if_valid:
            valid_mask = np.array([
                f1.is_match_with_other_valid(match.related_feature)
                for f1 in reference_features
            ])
            mask = mask & valid_mask  # combine masks
        remaining_scores = scores[mask]

        # if no remaining valid features, use inf
        if remaining_scores.size > 0:
            second_best_score = remaining_scores.min()
        else:
            second_best_score = float("inf")

        # compute average_ratio
        if second_best_score == 0:
            match.match_properties["average_ratio"] += 1
        else:
            match.match_properties["average_ratio"] += match.match_properties["distance"] / second_best_score

        # average the ratio from the two directions
        match.match_properties["average_ratio"] /= 2

    return matches


@beartype
def greedy_maximum_bipartite_matching_descriptor_distance(reference_features: list[Feature], related_features: list[Feature], distance_type: int) -> list[Match]:
    """
    Compute the greedy maximum bipartite matching between two sets of features based on the descriptor
    distance between the two.

    Parameters
    ----------
    reference_features : list[Feature]
        The list of features from the first image.
    related_features : list[Feature]
        The list of features from the second image.
    distance_type : int
        Either cv2.NORM_L2 or cv2.NORM_HAMMING dependent on wether to use hamming or euclidian distance
        
    Returns
    -------
    list[match]
        The greedy maximum bipartite matching between the images
    """
    if not reference_features or not related_features:
        return []

    reference_feature_descriptions = np.array([f.description for f in reference_features])
    related_feature_descriptions = np.array([f.description for f in related_features])

    # Compute full distance matrix once
    if distance_type == cv2.NORM_L2:
        differences = reference_feature_descriptions[:, None, :] - related_feature_descriptions[None, :, :]
        distance_matrix = -np.linalg.norm(differences, axis=2)
    elif distance_type == cv2.NORM_HAMMING:
        xor = np.bitwise_xor(reference_feature_descriptions[:, None, :], related_feature_descriptions[None, :, :])
        distance_matrix = -np.unpackbits(xor, axis=2).sum(axis=2)
    else:
        raise TypeError(f"Unknown distance type: {distance_type}")
    
    score_matrix = - distance_matrix

    return greedy_maximum_bipartite_matching(reference_features, related_features, score_matrix)
