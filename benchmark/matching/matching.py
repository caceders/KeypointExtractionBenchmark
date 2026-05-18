from ..feature import Feature
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from typing import Iterator
import heapq
import cv2
import numpy as np
from beartype import beartype
from config import *



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
    #@beartype
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
    
    #@beartype
    def add_match(self, matches : Match | list[Match]):

        if isinstance(matches, Match):
            self._matches.append(matches)
        elif isinstance(matches, list):
            if not all(isinstance(match, Match) for match in matches): raise TypeError("All elements in matches needs to be of type Match")
            self._matches += matches
        

    #@beartype
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


#@beartype
def greedy_maximum_bipartite_matching(reference_features: list[Feature], related_features: list[Feature], similarity_score_matrix: np.ndarray, similarity_higher_is_better: bool, calculate_match_properties: bool, use_mnn : bool = False) -> list[Match]:
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

    num_ref_features, num_rel_features = similarity_score_matrix.shape
    num_best_matches = min(NUM_BEST_MATCHES, num_rel_features, num_ref_features)
    num_scores_for_distinctivess = min(NUM_SCORES_DISTINCTIVNESS, num_best_matches)


    row_idx = np.arange(num_ref_features)[:, None]  # (n_ref, 1) for advanced indexing

    if similarity_higher_is_better:
        part_idxs = np.argpartition(-similarity_score_matrix, num_best_matches - 1, axis=1)[:, :num_best_matches]
        part_scores = similarity_score_matrix[row_idx, part_idxs]
        order = np.argsort(-part_scores, axis=1)
    else:
        part_idxs = np.argpartition(similarity_score_matrix, num_best_matches - 1, axis=1)[:, :num_best_matches]
        part_scores = similarity_score_matrix[row_idx, part_idxs]
        order = np.argsort(part_scores, axis=1)

    best_rel_feature_idxs = part_idxs[row_idx, order]        # (n_ref, k)
    best_similarity_scores = part_scores[row_idx, order]     # (n_ref, k)

    if use_mnn:
        # Forward: ref → best rel
        forward_best = best_rel_feature_idxs[:, 0]

        # Reverse: rel → best ref
        if similarity_higher_is_better:
            reverse_best = np.argmax(similarity_score_matrix, axis=0)
        else:
            reverse_best = np.argmin(similarity_score_matrix, axis=0)

        # Keep only mutual matches
        valid_refs = np.where(reverse_best[forward_best] == np.arange(num_ref_features))[0]

        # Filter arrays
        valid_refs = valid_refs.astype(np.int64)

        best_rel_feature_idxs = best_rel_feature_idxs[valid_refs]
        best_similarity_scores = best_similarity_scores[valid_refs]
        reference_features = [reference_features[i] for i in valid_refs]

        num_ref_features = len(reference_features)

    # Precompute softmax over top-k scores for all ref features at once
    if calculate_match_properties:
        s_block = best_similarity_scores[:, :num_scores_for_distinctivess].astype(np.float64)
        shifted = -(s_block - s_block.min(axis=1, keepdims=True)) / 10.0
        exps = np.exp(shifted)
        softmax_matrix = exps / exps.sum(axis=1, keepdims=True)  # (n_ref, num_scores_for_dist)

    # Convert to Python lists for zero-overhead element access in the hot loop
    best_rel_py = best_rel_feature_idxs.tolist()
    best_scores_py = best_similarity_scores.tolist()

    # Heap-based greedy matching: start with each ref feature's rank-0 candidate.
    # When a candidate's rel feature is already taken, push rank+1 for that ref.
    # This visits only the pairs actually needed regardless of NUM_BEST_MATCHES.
    heap: list = []
    sign = -1 if similarity_higher_is_better else 1
    for ref_idx in range(num_ref_features):
        heapq.heappush(heap, (sign * best_scores_py[ref_idx][0], ref_idx, best_rel_py[ref_idx][0], 0))

    matched_reference_feature_idxs = set()
    matched_related_feature_idxs = set()
    matches: list[Match] = []

    while heap:
        key, ref_feature_idx, rel_feature_idx, rank_in_row = heapq.heappop(heap)

        if rel_feature_idx in matched_related_feature_idxs:
            next_rank = rank_in_row + 1
            if next_rank < num_best_matches:
                heapq.heappush(heap, (sign * best_scores_py[ref_feature_idx][next_rank], ref_feature_idx, best_rel_py[ref_feature_idx][next_rank], next_rank))
            continue

        matched_reference_feature_idxs.add(ref_feature_idx)
        matched_related_feature_idxs.add(rel_feature_idx)
        match = Match(reference_features[ref_feature_idx], related_features[rel_feature_idx])
        matches.append(match)

        if calculate_match_properties:
            match.match_properties["distance"] = best_scores_py[ref_feature_idx][rank_in_row]
            match.match_properties["average_response"] = (match.reference_feature.keypoint.response + match.related_feature.keypoint.response) / 2.0
            if rank_in_row < num_scores_for_distinctivess:
                match.match_properties["distinctiveness"] = float(softmax_matrix[ref_feature_idx, rank_in_row])
                match.match_properties["match rank"] = rank_in_row
            else:
                match.match_properties["distinctiveness"] = 0.0
                match.match_properties["match rank"] = NUM_BEST_MATCHES

        if len(matches) == num_ref_features:
            break

    return matches

def knn_ratio_ransac_matching(
    reference_features: list[Feature],
    related_features: list[Feature],
    distance_type: int,
    use_mnn: bool,
    ratio_threshold: float = 1,
    ransac_reproj_threshold: float = 10.0,
    calculate_match_properties: bool = True,
) -> list[Match]:
    """
    Matching pipeline:
        1. KNN descriptor matching (k=2)
        2. Lowe ratio test
        3. RANSAC homography filtering

    Returns
    -------
    list[Match]
        Matches in same format as greedy_maximum_bipartite_matching
    """

    if not reference_features or not related_features:
        return []

    # --- 1. Build descriptor arrays ---
    ref_desc = np.array([f.description for f in reference_features])
    rel_desc = np.array([f.description for f in related_features])

    # Ensure float32 for OpenCV
    #ref_desc = ref_desc.astype(np.float32)
    #rel_desc = rel_desc.astype(np.float32)

    # --- 2. KNN matching ---
    bf = cv2.BFMatcher(distance_type, crossCheck=False)
    

    good_matches = []
    if (len(ref_desc) > 1 and len(rel_desc) > 1 and ratio_threshold < 1):
        knn_matches = bf.knnMatch(ref_desc, rel_desc, k=2)
        for m, n in knn_matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    else:
        good_matches = bf.match(ref_desc, rel_desc)

    if use_mnn:
        matches_reverse = bf.match(rel_desc, ref_desc)
        nn_reverse = {m.queryIdx: m.trainIdx for m in matches_reverse}
        good_matches= [m for m in good_matches if nn_reverse.get(m.trainIdx) == m.queryIdx]

    if len(good_matches) < 4:
        return []

    # --- 4. Collect point correspondences ---
    ref_pts = np.float32([reference_features[m.queryIdx].pt for m in good_matches])
    rel_pts = np.float32([related_features[m.trainIdx].pt for m in good_matches])

    # --- 5. RANSAC homography ---
    H, mask = cv2.findHomography(rel_pts, ref_pts, cv2.RANSAC, ransac_reproj_threshold)

    if mask is None:
        return []

    mask = mask.ravel().astype(bool)

    # --- 6. Build Match objects ---
    matches: list[Match] = []

    for i, m in enumerate(good_matches):
        if not mask[i]:
            continue

        ref_f = reference_features[m.queryIdx]
        rel_f = related_features[m.trainIdx]

        match = Match(ref_f, rel_f)

        if calculate_match_properties:
            match.match_properties["distance"] = float(m.distance)
            match.match_properties["average_response"] = 0
            match.match_properties["match rank"] = 0
            match.match_properties["distinctiveness"] = 1.0

        matches.append(match)

    return matches

def nearest_neighbor_matching(
    reference_features: list[Feature],
    related_features: list[Feature],
    distance_type: int,
    use_mutual: bool = False,
    calculate_match_properties: bool = True,
) -> list[Match]:
    """
    Matching pipeline:
        1. Nearest neighbor matching
        2. Optional mutual nearest neighbor filtering (cross-check)

    Returns
    -------
    list[Match]
        Matches in same format as greedy_maximum_bipartite_matching
    """

    if not reference_features or not related_features:
        return []

    # --- 1. Build descriptor arrays ---
    ref_desc = np.array([f.description for f in reference_features])
    rel_desc = np.array([f.description for f in related_features])

    bf = cv2.BFMatcher(distance_type, crossCheck=False)

    # --- 2. Forward matching (ref -> rel) ---
    forward_matches = bf.match(ref_desc, rel_desc)

    if not use_mutual:
        valid_matches = forward_matches
    else:
        # --- 3. Reverse matching (rel -> ref) ---
        backward_matches = bf.match(rel_desc, ref_desc)

        # Build lookup: rel_idx -> best ref_idx
        rel_to_ref = {m.queryIdx: m.trainIdx for m in backward_matches}

        # Keep only mutual matches
        valid_matches = []
        for m in forward_matches:
            ref_idx = m.queryIdx
            rel_idx = m.trainIdx

            if rel_idx in rel_to_ref and rel_to_ref[rel_idx] == ref_idx:
                valid_matches.append(m)

    # --- 4. Build Match objects ---
    matches: list[Match] = []

    for m in valid_matches:
        ref_f = reference_features[m.queryIdx]
        rel_f = related_features[m.trainIdx]

        match = Match(ref_f, rel_f)

        if calculate_match_properties:
            match.match_properties["distance"] = float(m.distance)
            match.match_properties["average_response"] = 0
            match.match_properties["match rank"] = 0
            match.match_properties["distinctiveness"] = 1.0

        matches.append(match)

    return matches

#@beartype
def greedy_maximum_bipartite_matching_descriptor_distance(reference_features: list[Feature], related_features: list[Feature], distance_type: int, use_mnn: bool = False) -> list[Match]:
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
        distance_matrix = cdist(reference_feature_descriptions, related_feature_descriptions, metric='euclidean')
    elif distance_type == cv2.NORM_HAMMING:
        xor = np.bitwise_xor(reference_feature_descriptions[:, None, :], related_feature_descriptions[None, :, :])
        distance_matrix = np.unpackbits(xor, axis=2).sum(axis=2)
    else:
        raise TypeError(f"Unknown distance type: {distance_type}")

    return greedy_maximum_bipartite_matching(reference_features, related_features, distance_matrix, False, True, use_mnn)
