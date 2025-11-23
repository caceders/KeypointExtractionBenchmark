from ..feature import Feature
import numpy as np
import cv2

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

def greedy_maximum_bipartite_matching(
        features1: list[Feature],
        features2: list[Feature],
        distance_type: int
    ) -> list[Match]:

    # --- Step 0: empty case ---
    if not features1 or not features2:
        return []

    N = len(features1)
    M = len(features2)

    # Convert to numpy now to avoid repeated conversion later
    ref_desc = [np.asarray(f.desc) for f in features1]
    rel_desc = [np.asarray(f.desc) for f in features2]

    # --- Step 1: compute all pairwise distances (without storing NxM arrays) ---
    pairs = []   # (i, j, dist)

    for i in range(N):
        a = ref_desc[i]

        if distance_type == cv2.NORM_L2:
            diff = np.stack(rel_desc) - a
            d = np.linalg.norm(diff, axis=1)      # (M,)

        elif distance_type == cv2.NORM_HAMMING:
            xor = np.bitwise_xor(np.stack(rel_desc), a)
            d = np.unpackbits(xor, axis=1).sum(axis=1)   # (M,)

        else:
            raise ValueError("Unsupported distance type")

        for j in range(M):
            pairs.append((i, j, float(d[j])))

    # Sort all pairs by ascending distance
    pairs.sort(key=lambda x: x[2])

    # --- Step 2: greedy matching ---
    used_ref = set()
    used_rel = set()
    ref_to_match: dict[int, Match] = {}
    rel_to_match: dict[int, Match] = {}
    matches: list[Match] = []

    for i, j, dist in pairs:
        if i not in used_ref and j not in used_rel:
            m = Match(features1[i], features2[j], dist)
            matches.append(m)

            used_ref.add(i)
            used_rel.add(j)

            ref_to_match[i] = m
            rel_to_match[j] = m

            m.custom_properties["distance"] = dist
            m.custom_properties["average_response"] = (
                features1[i].kp.response + features2[j].kp.response
            ) / 2
            m.custom_properties["average_ratio"] = 0.0

    # --- Step 3: next-best distances (ratio test) ---

    # Pre-stack descriptors for vectorized second-best search
    ref_desc_np = np.stack(ref_desc)
    rel_desc_np = np.stack(rel_desc)

    # For reference → related direction
    for i, match in ref_to_match.items():
        a = ref_desc_np[i]

        if distance_type == cv2.NORM_L2:
            d = np.linalg.norm(rel_desc_np - a, axis=1)
        else:
            xor = np.bitwise_xor(rel_desc_np, a)
            d = np.unpackbits(xor, axis=1).sum(axis=1)

        j_match = features2.index(match.feature2)
        d[j_match] = np.inf  # exclude the matched one
        second = float(d.min())

        match.custom_properties["average_ratio"] = (
            match.custom_properties["distance"] / second
        )

    # For related → reference direction
    for j, match in rel_to_match.items():
        b = rel_desc_np[j]

        if distance_type == cv2.NORM_L2:
            d = np.linalg.norm(ref_desc_np - b, axis=1)
        else:
            xor = np.bitwise_xor(ref_desc_np, b)
            d = np.unpackbits(xor, axis=1).sum(axis=1)

        i_match = features1.index(match.feature1)
        d[i_match] = np.inf
        second = float(d.min())

        prev = match.custom_properties["average_ratio"]
        match.custom_properties["average_ratio"] = (
            prev + match.custom_properties["distance"] / second
        ) / 2.0

    return matches