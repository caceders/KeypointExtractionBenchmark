from ..feature import Feature
import numpy as np

class Match:
    def __init__(self, feature1: Feature, feature2: Feature, score:float = 0):
            self.feature1 = feature1
            self.feature2 = feature2
            self.score = score

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