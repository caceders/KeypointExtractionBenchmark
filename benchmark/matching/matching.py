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
def greedy_maximum_bipartite_matching(reference_features: list[Feature], related_features: list[Feature], similarity_score_matrix: np.ndarray, similarity_higher_is_better: bool, calculate_match_properties: bool) -> list[Match]:
    """
    Compute the greedy maximum bipartite matching between two sets of features based on a similarity matrix, with either low values indicating similarity, like distance or high values, like overlap

    Parameters
    ----------
    reference_features : list[Feature]
        The list of features from the first image.
    related_features : list[Feature]
        The list of features from the second image.
    similarity_matrix : np.ndarray
        The similarity matrix with values from reference_features to related_features
    similarity_higher_is_better: bool
        True means algorithm optimizes for high similiarity values in the matches, opposite for False

    Returns
    -------
    list[match]
        The greedy maximum bipartite matching between the images
    """
    if not reference_features or not related_features:
        return []
    
    NUM_SCORES_DISTINCTIVNESS = 10
    NUM_BEST_MATCHES = 20

    num_ref_features, num_rel_features = similarity_score_matrix.shape
    num_best_matches = min(NUM_BEST_MATCHES, num_rel_features)


    best_rel_feature_idxs = np.empty((num_ref_features, num_best_matches), dtype=np.int64)
    best_similarity_scores = np.empty((num_ref_features, num_best_matches), dtype=similarity_score_matrix.dtype)

    for ref_feature_idx in range(num_ref_features):
        similarity_scores = similarity_score_matrix[ref_feature_idx]
        if similarity_higher_is_better:
            best_matches_idxs = np.argpartition(-similarity_scores, num_best_matches-1)[:num_best_matches]      # NUM_BEST_MATCHES largest (unordered)
            best_matches_idx_order = np.argsort(-similarity_scores[best_matches_idxs])             # sort those descending
        else:
            best_matches_idxs = np.argpartition(similarity_scores, num_best_matches-1)[:num_best_matches]       # NUMB_BEST_MATCHES smallest (unordered)
            best_matches_idx_order = np.argsort(similarity_scores[best_matches_idxs])              # sort those ascending

        best_matches_ordered = best_matches_idxs[best_matches_idx_order]
        best_rel_feature_idxs[ref_feature_idx] = best_matches_ordered
        best_similarity_scores[ref_feature_idx] = similarity_scores[best_matches_ordered]

    pairs = [(ref_feature_idx, best_rel_feature_idxs[ref_feature_idx][best_matches_idx], best_similarity_scores[ref_feature_idx][best_matches_idx]) 
             for ref_feature_idx in range(num_ref_features) 
             for best_matches_idx in range(num_best_matches)]
    
    pairs_sorted = sorted(pairs, key= lambda x: -x[2] if similarity_higher_is_better else x[2])

    matched_reference_feature_idxs = set()
    matched_related_feature_idxs = set()
    matches: list[Match] = []

    for ref_feature_idx, rel_feature_idx, similarity_score in pairs_sorted:
        if ref_feature_idx in matched_reference_feature_idxs or rel_feature_idx in matched_related_feature_idxs:
            continue
        matched_reference_feature_idxs.add(ref_feature_idx)
        matched_related_feature_idxs.add(rel_feature_idx)
        match = Match(reference_features[ref_feature_idx], related_features[rel_feature_idx])
        matches.append(match)
        #Only when used as matching approach
        if calculate_match_properties:
            match.match_properties["distance"] = float(similarity_score)
            match.match_properties["average_response"] = (match.reference_feature.keypoint.response + match.related_feature.keypoint.response)/2
            scores = best_similarity_scores[ref_feature_idx][:NUM_SCORES_DISTINCTIVNESS]

            # Linear preference: higher weight for lower scores
            N = len(scores)
            pref = (np.max(scores) - scores)
            pref = np.clip(pref, 0.0, None)
            if pref.sum() == 0:
                pref = np.ones_like(scores)
            pref /= pref.sum()  # normalize to sum=1

            # Fair weights: mean = 1
            strength = 1.0  # adjust emphasis (0 = uniform, 1 = strong)
            w_fair = 1.0 + strength * N * (pref - 1.0 / N)

            # Weighted average with fair weights
            fair_avg = np.mean(w_fair * scores)

            match.match_properties["distinctiveness"] = fair_avg/(similarity_score+1e-12)

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
        distance_matrix = np.linalg.norm(differences, axis=2)
    elif distance_type == cv2.NORM_HAMMING:
        xor = np.bitwise_xor(reference_feature_descriptions[:, None, :], related_feature_descriptions[None, :, :])
        distance_matrix = np.unpackbits(xor, axis=2).sum(axis=2)
    else:
        raise TypeError(f"Unknown distance type: {distance_type}")

    return greedy_maximum_bipartite_matching(reference_features, related_features, distance_matrix, False, True)
