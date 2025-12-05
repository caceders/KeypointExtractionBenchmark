from ..feature import Feature
from sklearn.metrics import average_precision_score
from typing import Iterator, Tuple
import cv2
import numpy as np

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

        # Cast to float to stop scalar negative overflow error
        if match_rank_property.higher_is_better:
            scores = [float(match.match_properties[match_rank_property.name]) for match in self._matches]
        else:
            scores = [- float(match.match_properties[match_rank_property.name])  for match in self._matches]

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
    NUM_BEST_MATCHES = 10

    num_ref_features, num_rel_features = similarity_score_matrix.shape
    num_best_matches = min(NUM_BEST_MATCHES, num_rel_features)

    
    # --- Fast path: single reference feature ---
    if num_ref_features == 1:
        # Select the best related feature for the single reference without full sorting
        ref_feature = reference_features[0]
        similarity_scores = similarity_score_matrix[0]
        if num_best_matches == 0:
            return []

        if similarity_higher_is_better:
            # Top-k highest (unordered)
            topk_unordered = np.argpartition(-similarity_scores, num_best_matches - 1)[:num_best_matches]
            topk_scores = similarity_scores[topk_unordered]
            # Choose the best among the k without sorting all k
            best_local = int(np.argmax(topk_scores))
            # For distinctiveness, order these k scores descending
            order_for_dist = np.argsort(-topk_scores)
        else:
            # Top-k lowest (unordered)
            topk_unordered = np.argpartition(similarity_scores, num_best_matches - 1)[:num_best_matches]
            topk_scores = similarity_scores[topk_unordered]
            # Choose the best among the k without sorting all k
            best_local = int(np.argmin(topk_scores))
            # For distinctiveness, order these k scores ascending
            order_for_dist = np.argsort(topk_scores)

        rel_feature_idx = int(topk_unordered[best_local])
        best_score = float(topk_scores[best_local])

        match = Match(ref_feature, related_features[rel_feature_idx])

        if calculate_match_properties:
            match.match_properties["distance"] = best_score
            match.match_properties["average_response"] = (
                match.reference_feature.keypoint.response + match.related_feature.keypoint.response
            ) / 2.0
            # Distinctiveness: mean of the first NUM_SCORES_DISTINCTIVNESS scores in ordered top-k,
            # divided by the chosen score, consistent with your original definition.
            d = min(NUM_SCORES_DISTINCTIVNESS, topk_scores.size)
            base_mean = float(topk_scores[order_for_dist][:d].mean())
            match.match_properties["distinctiveness"] = base_mean / (best_score + 1e-12)

        return [match]


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
            match.match_properties["distinctiveness"] = np.mean(scores)/(similarity_score + 1e-12)

    return matches


# def greedy_maximum_bipartite_matching(reference_features: list[Feature], related_features: list[Feature], distance_matrix: np.ndarray) -> list[Match]:
#     """
#     Compute the greedy maximum bipartite matching between two sets of features based on a distance matrix

#     Parameters
#     ----------
#     reference_features : list[Feature]
#         The list of features from the first image.
#     related_features : list[Feature]
#         The list of features from the second image.
#     distance_matrix : np.ndarray
#         The distance matrix with distances from reference_features to related_features

#     Returns
#     -------
#     list[match]
#         The greedy maximum bipartite matching between the images
#     """
#     if not reference_features or not related_features:
#         return []
    
#     reference_features_length = len(reference_features)
#     related_features_length = len(related_features)

#     # Cover singular element cases

#     if reference_features_length == 1:
#         reference_feature = reference_features[0]
#         dists = distance_matrix[0]
        
#         if dists.size >= 2:
#             idxs = np.argpartition(dists, 1)[:2]       # two smallest, unordered
#             # find which of the two is the smallest
#             closest_idx = idxs[np.argmin(dists[idxs])]
#             # the other one is the second-smallest
#             second_idx = idxs[1] if closest_idx == idxs[0] else idxs[0]

#             closest_distance = float(dists[closest_idx])
#             second_closest_distance = float(dists[second_idx])
#         else:
#             closest_idx = 0
#             closest_distance = float(dists[0])
#             second_closest_distance = float('inf') 


#         closest_feature = related_features[closest_idx]

#         match = Match(reference_feature, closest_feature)
#         match.match_properties["distance"] = closest_distance
#         match.match_properties["average_response"] = (closest_feature.keypoint.response + reference_feature.keypoint.response) / 2
#         if second_closest_distance != 0:
#             match.match_properties["average_ratio"] = closest_distance / second_closest_distance
#         else:
#             match.match_properties["average_ratio"] = 1
#         return [match]

#     if related_features_length == 1:
#         related_feature = related_features[0]
#         dists = distance_matrix[:, 0]

#         if dists.size >= 2:
#             idxs = np.argpartition(dists, 1)[:2]       # two smallest, unordered
#             # find which of the two is the smallest
#             closest_idx = idxs[np.argmin(dists[idxs])]
#             # the other one is the second-smallest
#             second_idx = idxs[1] if closest_idx == idxs[0] else idxs[0]

#             closest_distance = float(dists[closest_idx])
#             second_closest_distance = float(dists[second_idx])
#         else:
#             closest_idx = 0
#             closest_distance = float(dists[0])
#             second_closest_distance = float('inf') 

#         closest_feature = reference_features[closest_idx]

#         match = Match(closest_feature, related_feature)
#         match.match_properties["distance"] = closest_distance
#         match.match_properties["average_response"] = (closest_feature.keypoint.response + related_feature.keypoint.response) / 2
#         if second_closest_distance != 0:
#             match.match_properties["average_ratio"] = closest_distance / second_closest_distance
#         else:
#             match.match_properties["average_ratio"] = 1
#         return [match]

#     # Non-singular cases
#     reference_feature_indices, related_feature_indices = np.nonzero(np.ones_like(distance_matrix))  # generate all pairs
#     pairs = [(i, j, distance_matrix[i, j]) for i, j in zip(reference_feature_indices, related_feature_indices)]
#     pairs.sort(key=lambda p: p[2])

#     matched_reference_feature_indices = set()
#     matched_related_feature_indices = set()
#     reference_feature_to_match: dict[int, Match] = {}
#     related_feature_to_match: dict[int, Match] = {}
#     matches: list[Match] = []

#     for reference_feature_index, related_feature_index, dist in pairs:
#         if reference_feature_index not in matched_reference_feature_indices and related_feature_index not in matched_related_feature_indices:
#             match = Match(reference_features[reference_feature_index], related_features[related_feature_index])
#             matches.append(match)
#             matched_reference_feature_indices.add(reference_feature_index)
#             matched_related_feature_indices.add(related_feature_index)
#             reference_feature_to_match[reference_feature_index] = match
#             related_feature_to_match[related_feature_index] = match
#             match.match_properties["distance"] = float(dist)
#             match.match_properties["average_response"] = (reference_features[reference_feature_index].keypoint.response + related_features[related_feature_index].keypoint.response) / 2
#             match.match_properties["average_ratio"] = 0.0  # to be updated

#     # reference_feature -> related_feature: closest alternative
#     for reference_feature_index, match in reference_feature_to_match.items():
#         distances = distance_matrix[reference_feature_index, :]
#         matched_related_feature_index = related_features.index(match.related_feature)
#         mask = np.arange(related_features_length) != matched_related_feature_index
#         second_closest_distance = distances[mask].min()
#         if second_closest_distance != 0:
#             match.match_properties["average_ratio"] = match.match_properties["distance"] / second_closest_distance
#         else:
#             match.match_properties["average_ratio"] = 1
            
#     # related_feature -> reference_feature: closest alternative
#     for related_feature_index, match in related_feature_to_match.items():
#         distances = distance_matrix[:, related_feature_index]
#         matched_reference_feature_index = reference_features.index(match.reference_feature)
#         mask = np.arange(reference_features_length) != matched_reference_feature_index
#         second_closest_distance = distances[mask].min()
#         if second_closest_distance != 0:
#             match.match_properties["average_ratio"] += match.match_properties["distance"] / second_closest_distance
#         else:
#             match.match_properties["average_ratio"] += 1
#         match.match_properties["average_ratio"] /= 2

#     return matches



def greedy_maximum_bipartite_matching_homographic_distance(reference_features: list[Feature], related_features: list[Feature], homography1to2: np.ndarray) -> list[Match]:
    """
    Compute the optimal homographic matching between two sets of features with apriori knowledge about
    the homografic transformation between the two images the features were taken from. Uses a greedy
    maximum bipartite matching algorithm.

    Parameters
    ----------
    reference_features : list[Feature]
        The list of features from the first image.
    related_features : list[Feature]
        The list of features from the second image.
    homography1to2: np.ndarray
        The homographic transformation matrix between the two images

    Returns
    -------
    list[match]
        The homographical optimal matching based on the homographic transformation between the images
    """
    if not reference_features or not related_features:
        return []

    reference_feature_positions = np.array([feature.pt for feature in reference_features])
    related_feature_transformed_positions = np.array([feature.get_pt_after_homography_transform(homography1to2)
                        for feature in related_features])

    # Compute full distance matrix once
    differences = reference_feature_positions[:, None, :] - related_feature_transformed_positions[None, :, :]
    distance_matrix = np.linalg.norm(differences, axis=2)
    return distance_matrix
    return greedy_maximum_bipartite_matching(reference_features, related_features, distance_matrix)



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
    #return greedy_maximum_bipartite_matching(reference_features, related_features, distance_matrix)
