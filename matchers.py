"""
Matching functions for mma_benchmark.py and KITTI_main.py.

Three matchers
--------------
  match_nn   — forward nearest-neighbour (k-NN, k=2); apply apply_ratio_uni afterwards.
  match_mnn  — mutual nearest-neighbour (k-NN k=2 both ways); apply apply_ratio_bi afterwards.
  match_keem — greedy maximum bipartite matching on descriptor distance; no ratio test.

Ratio-test helpers
------------------
  apply_ratio_uni — Lowe's unidirectional ratio test (for NN matches).
  apply_ratio_bi  — bidirectional ratio test (for MNN matches).

Convenience
-----------
  get_matches — dispatches to the right matcher + ratio test in one call.
                Returns list[RawMatch] regardless of matcher.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RawMatch:
    query_idx: int
    train_idx: int
    distance:  float


@dataclass
class NNMatch:
    """Forward kNN pair (k=2). Carries enough info for unidirectional ratio test."""
    best:   RawMatch
    second: Optional[RawMatch]


@dataclass
class MNNMatch:
    """MNN triple. Carries enough info for bidirectional ratio test."""
    best:       RawMatch
    fwd_second: Optional[RawMatch]  # 2nd nearest rel for ref[query_idx]
    rev_second: Optional[RawMatch]  # 2nd nearest ref for rel[train_idx]


# ---------------------------------------------------------------------------
# Internal: descriptor distance matrix
# ---------------------------------------------------------------------------

def _distance_matrix(
    desc_ref: np.ndarray,
    desc_rel: np.ndarray,
    distance_type: int,
) -> np.ndarray:
    if distance_type == cv2.NORM_L2:
        a = np.asarray(desc_ref, dtype=np.float32)
        b = np.asarray(desc_rel, dtype=np.float32)
        dist_sq = (
            np.sum(a ** 2, axis=1)[:, None]
            + np.sum(b ** 2, axis=1)[None, :]
            - 2.0 * (a @ b.T)
        )
        return np.sqrt(np.maximum(dist_sq, 0.0))

    if distance_type == cv2.NORM_HAMMING:
        d = desc_ref.shape[1]
        if d % 8 == 0:
            r64 = np.ascontiguousarray(desc_ref).view(np.uint64)
            l64 = np.ascontiguousarray(desc_rel).view(np.uint64)
            xor64 = np.bitwise_xor(r64[:, None, :], l64[None, :, :])
            return np.bitwise_count(xor64).sum(axis=-1).astype(np.float32)
        xor = np.bitwise_xor(desc_ref[:, None, :], desc_rel[None, :, :])
        return np.bitwise_count(xor).sum(axis=-1).astype(np.float32)

    raise ValueError(f"Unknown distance_type: {distance_type}")


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------

def match_nn(
    desc_ref: np.ndarray,
    desc_rel: np.ndarray,
    distance_type: int,
) -> list[NNMatch]:
    """Forward k-NN (k=2). Apply apply_ratio_uni for ratio test."""
    if len(desc_ref) == 0 or len(desc_rel) == 0:
        return []
    bf = cv2.BFMatcher(distance_type, crossCheck=False)
    k = min(2, len(desc_rel))
    raw = bf.knnMatch(desc_ref, desc_rel, k=k)
    result = []
    for p in raw:
        best   = RawMatch(p[0].queryIdx, p[0].trainIdx, float(p[0].distance))
        second = (RawMatch(p[1].queryIdx, p[1].trainIdx, float(p[1].distance))
                  if len(p) >= 2 else None)
        result.append(NNMatch(best, second))
    return result


def match_mnn(
    desc_ref: np.ndarray,
    desc_rel: np.ndarray,
    distance_type: int,
) -> list[MNNMatch]:
    """MNN with k-NN (k=2) both ways. Apply apply_ratio_bi for ratio test."""
    if len(desc_ref) == 0 or len(desc_rel) == 0:
        return []
    bf = cv2.BFMatcher(distance_type, crossCheck=False)
    fwd = bf.knnMatch(desc_ref, desc_rel, k=min(2, len(desc_rel)))
    rev = bf.knnMatch(desc_rel, desc_ref, k=min(2, len(desc_ref)))

    rev_best: dict[int, int]                = {}
    rev_sec:  dict[int, Optional[RawMatch]] = {}
    for p in rev:
        j = p[0].queryIdx
        rev_best[j] = p[0].trainIdx
        rev_sec[j]  = (RawMatch(p[1].queryIdx, p[1].trainIdx, float(p[1].distance))
                       if len(p) >= 2 else None)

    result = []
    for p in fwd:
        i, j = p[0].queryIdx, p[0].trainIdx
        if rev_best.get(j) != i:
            continue
        best    = RawMatch(i, j, float(p[0].distance))
        fwd_sec = (RawMatch(p[1].queryIdx, p[1].trainIdx, float(p[1].distance))
                   if len(p) >= 2 else None)
        result.append(MNNMatch(best, fwd_sec, rev_sec.get(j)))
    return result


def match_keem(
    desc_ref: np.ndarray,
    desc_rel: np.ndarray,
    distance_type: int,
) -> list[RawMatch]:
    """
    Greedy maximum bipartite matching on descriptor distance.
    No ratio test — the algorithm itself resolves conflicts by distance rank.
    Mirrors the heap logic in benchmark/matching/matching.py.
    """
    n_ref, n_rel = len(desc_ref), len(desc_rel)
    if n_ref == 0 or n_rel == 0:
        return []

    D = _distance_matrix(desc_ref, desc_rel, distance_type)  # (n_ref, n_rel)

    # Partial sort: we need at most k = min(n_ref, n_rel) candidates per row.
    # Proof: in the worst case all refs compete for the same k rel features,
    # so the last ref needs at most rank k-1 before finding a free slot.
    k = min(n_ref, n_rel)
    if k < n_rel:
        part = np.argpartition(D, k - 1, axis=1)[:, :k]         # (n_ref, k), unordered
        part_D = D[np.arange(n_ref)[:, None], part]             # (n_ref, k)
        order = np.argsort(part_D, axis=1)                      # sort within partition
        sorted_rel  = part[np.arange(n_ref)[:, None], order]    # (n_ref, k)
        sorted_dist = part_D[np.arange(n_ref)[:, None], order]  # (n_ref, k)
    else:
        order = np.argsort(D, axis=1)                           # (n_ref, n_rel)
        sorted_rel  = order
        sorted_dist = D[np.arange(n_ref)[:, None], order]

    # Convert to Python lists: element access in tight Python loops is ~3-5x faster
    # than numpy indexing due to avoided type conversion and bounds checking overhead.
    sorted_rel_list  = sorted_rel.tolist()
    sorted_dist_list = sorted_dist.tolist()

    heap = [(sorted_dist_list[i][0], i, sorted_rel_list[i][0], 0) for i in range(n_ref)]
    heapq.heapify(heap)  # O(n) vs n × heappush O(n log n)

    matched_ref = [False] * n_ref
    matched_rel = [False] * n_rel
    result: list[RawMatch] = []

    while heap:
        dist, i, j, rank = heapq.heappop(heap)
        if matched_ref[i]:
            continue
        if matched_rel[j]:
            nxt = rank + 1
            if nxt < k:
                heapq.heappush(heap, (sorted_dist_list[i][nxt], i, sorted_rel_list[i][nxt], nxt))
            continue
        matched_ref[i] = True
        matched_rel[j] = True
        result.append(RawMatch(i, j, dist))

    return result


# ---------------------------------------------------------------------------
# Ratio-test helpers
# ---------------------------------------------------------------------------

def apply_ratio_uni(nn_matches: list[NNMatch], ratio: float) -> list[RawMatch]:
    """Lowe's unidirectional ratio test. Use after match_nn."""
    return [
        m.best for m in nn_matches
        if m.second is None or m.best.distance < ratio * m.second.distance
    ]


def apply_ratio_bi(mnn_matches: list[MNNMatch], ratio: float) -> list[RawMatch]:
    """Bidirectional ratio test. Use after match_mnn."""
    out = []
    for m in mnn_matches:
        fwd_ok = m.fwd_second is None or m.best.distance < ratio * m.fwd_second.distance
        rev_ok = m.rev_second is None or m.best.distance < ratio * m.rev_second.distance
        if fwd_ok and rev_ok:
            out.append(m.best)
    return out


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def get_matches(
    desc_ref:       np.ndarray,
    desc_rel:       np.ndarray,
    distance_type:  int,
    matcher:        str,
    ratio_threshold: Optional[float] = None,
) -> list[RawMatch]:
    """
    Run the chosen matcher and (for NN/MNN) apply the matching ratio test.

    Parameters
    ----------
    matcher         : "NN" | "MNN" | "KEEM"
    ratio_threshold : applied only for NN (unidirectional) and MNN (bidirectional);
                      ignored for KEEM; None means no ratio test.

    Returns
    -------
    list[RawMatch]  — each entry has .query_idx, .train_idx, .distance
    """
    if matcher == "KEEM":
        return match_keem(desc_ref, desc_rel, distance_type)

    if matcher == "NN":
        nn = match_nn(desc_ref, desc_rel, distance_type)
        if ratio_threshold is not None:
            return apply_ratio_uni(nn, ratio_threshold)
        return [m.best for m in nn]

    if matcher == "MNN":
        mnn = match_mnn(desc_ref, desc_rel, distance_type)
        if ratio_threshold is not None:

            return apply_ratio_bi(mnn, ratio_threshold)
        return [m.best for m in mnn]

    raise ValueError(f"Unknown matcher {matcher!r}. Choose 'NN', 'MNN', or 'KEEM'.")
