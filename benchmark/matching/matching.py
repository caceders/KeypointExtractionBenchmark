from ..feature import Feature
import numpy as np

class Match:
    def __init__(self, feature1: Feature, feature2: Feature, score:float = 0):
            self.feature1 = feature1
            self.feature2 = feature2
            self.score = score
            self.is_correct = feature1.is_match_with_other_valid(feature2)
            self.custom_properties = {}

def homographic_optimal_matching(features1: list[Feature], features2: list[Feature], homography1to2: np.ndarray) -> list[Match]:
    # Compute pairwise distance matrix
    ref_pts = np.array([f.pt for f in features1])
    rel_pts = np.array([f.get_pt_after_homography_transform(homography1to2)
                        for f in features2])

    if len(ref_pts) == 0 or len(rel_pts) == 0:
        matches: list[Match] = []
    else:
        dists = np.linalg.norm(ref_pts[:, None, :] - rel_pts[None, :, :], axis=2)

        # Greedy one-to-one matching
        matches: list[Match] = []
        used_ref = set()
        used_rel = set()

        # Sort all pairs by distance
        pairs = [(i, j, dists[i, j]) for i in range(dists.shape[0]) for j in range(dists.shape[1])]
        pairs.sort(key=lambda x: x[2])

        for i, j, dist in pairs:
            if i not in used_ref and j not in used_rel:
                matches.append(Match(features1[i], features2[j], dist))
            used_ref.add(i)
            used_rel.add(j)


    matches.reverse()  # reverse so it is again sorted by distance
    return matches

def greedy_maximum_bipartite_matching(features1: list[Feature], features2: list[Feature]) -> list[Match]:

    matches: list[Match] = []

    # --- Step 1: Build descriptor arrays ---
    ref_desc = np.array([f.desc for f in features1])  # shape: (N, D)
    rel_desc = np.array([f.desc for f in features2])  # shape: (M, D)

    # --- Step 2: Compute pairwise distances ---
    dists = np.linalg.norm(ref_desc[:, None, :] - rel_desc[None, :, :], axis=2)  # shape: (N, M)

    # --- Step 3: Greedy maximum bipartite matching ---
    used_ref = set()
    used_rel = set()
    ref_to_match: dict[int, Match] = {}
    rel_to_match: dict[int, Match] = {}

    # Flatten all pairs and sort by distance
    pairs = [(i, j, dists[i, j]) for i in range(len(features1)) for j in range(len(features2))]
    pairs.sort(key=lambda x: x[2])

    for i, j, dist in pairs:
        if i not in used_ref and j not in used_rel:
            match = Match(features1[i], features2[j], dist)
            matches.append(match)
            used_ref.add(i)
            used_rel.add(j)
            ref_to_match[i] = match
            rel_to_match[j] = match

            # Store additional properties
            match.custom_properties["distance"] = dist
            match.custom_properties["average_response"] = (features1[i].kp.response + features2[j].kp.response) / 2
            match.custom_properties["average_ratio"] = 0  # to be computed next

    # --- Step 4: Compute next-best distances and average_ratio ---
    for i, match in ref_to_match.items():
        # Distances to all other features in features2 (excluding matched one)
        unmatched_dists = [dists[i, j2] for j2 in range(len(features2)) if j2 != features2.index(match.feature2)]
        if unmatched_dists:
            second_best_dist = min(unmatched_dists)
            match.custom_properties["average_ratio"] = match.custom_properties["distance"] / second_best_dist
        else:
            # If no alternative exists, ratio = 1
            match.custom_properties["average_ratio"] = 1.0

    for j, match in rel_to_match.items():
        unmatched_dists = [dists[i2, j] for i2 in range(len(features1)) if i2 != features1.index(match.feature1)]
        if unmatched_dists:
            second_best_dist = min(unmatched_dists)
            # If average_ratio was already set from ref side, average the two contributions
            match.custom_properties["average_ratio"] = (
                match.custom_properties["average_ratio"] + match.custom_properties["distance"] / second_best_dist
            ) / 2
        else:
            match.custom_properties["average_ratio"] = match.custom_properties.get("average_ratio", 1.0)

    return matches