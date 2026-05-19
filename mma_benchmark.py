import os

from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import downsample
from tqdm import tqdm
import itertools
import traceback
import cv2
import numpy as np
import pandas as pd
import warnings

try:
    from shi_tomasi_sift import ShiTomasiSift
except ImportError:
    ShiTomasiSift = None

# ============================================================
# CONFIGURATION
# ============================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
HPATCHES_PATH = r"hpatches-sequences-release"
RESULTS_FILE  = "mma_results/mnn_ratio_order_test.csv"

# ── Run tag ───────────────────────────────────────────────────────────────────
# Label for this entire benchmark run. All combinations share this tag.
# Use a different tag for each run you want to compare in display_mma.py.
RUN_TAG = "post_opt"

# ── Feature combinations ──────────────────────────────────────────────────────
features2d = {
    # "SIFT":      cv2.SIFT_create(),
    "ORB_default":       cv2.ORB_create(nfeatures=5000),
    # "BRISK":     cv2.BRISK_create(),
    # "AKAZE":     cv2.AKAZE_create(),
    # "GFTT":      cv2.GFTTDetector_create(maxCorners=5000),
    ## LOW THRESH
    "SIFT":      cv2.SIFT_create(contrastThreshold = 0.001),
    "ORB":       cv2.ORB_create(nfeatures=5000, edgeThreshold = 10),
    "BRISK":     cv2.BRISK_create(thresh = 5),
    "AKAZE":     cv2.AKAZE_create(threshold=0.0000005),
    "GFTT":      cv2.GFTTDetector_create(maxCorners=5000, qualityLevel = 0.001),
}

ONLY_SELF             = True
ONLY_SELF_EXCEPTIONS  = [("GFTT", "SIFT")]
ONLY_USED_AS_DETECTOR = ["GFTT"]

# ── Evaluation thresholds ─────────────────────────────────────────────────────
# Pixel-error thresholds used for MMA, repeatability, and homography accuracy.
DISTANCE_THRESHOLDS = list(range(1, 31))

# ── Matching parameters ───────────────────────────────────────────────────────
# Each parameter is a list; all combinations are benchmarked and stored in the CSV.
MAX_KEYPOINTS    = [500]
USE_MNN         = [True]    # mutual nearest-neighbour filter on/off
RATIO_FIRST = False
RATIO_THRESHOLDS = [0.8]     # Lowe's ratio test threshold
RANSAC_THRESHOLDS   = [3.0]     # RANSAC reprojection error threshold (px)

# ── Downsampling parameters ───────────────────────────────────────────────────
DOWNSAMPLE_LEVELS             = [0]
DOWNSAMPLE_FACTOR             = [2]
DOWNSAMPLE_INTERPOLATION_TYPE = [None]
INITIAL_SIGMAS                 = [0]
INTRINSIC_SIGMA               = [0.5]
APPLY_PROGRESSIVE_BLUR        = [False]

SKIP_AT_ERROR = False


# ============================================================
# BUILD TEST COMBINATIONS
# ============================================================

try:
    _binary_types = (
        cv2.ORB, cv2.BRISK, cv2.AKAZE,
        cv2.xfeatures2d.BriefDescriptorExtractor,
        cv2.xfeatures2d.FREAK,
        cv2.xfeatures2d.LATCH,
    )
except AttributeError:
    _binary_types = (cv2.ORB, cv2.BRISK, cv2.AKAZE)

test_combinations: dict[str, FeatureExtractor] = {}
for det_key in features2d:
    for desc_key in features2d:
        if ONLY_SELF and det_key != desc_key and (det_key, desc_key) not in ONLY_SELF_EXCEPTIONS:
            continue
        if desc_key in ONLY_USED_AS_DETECTOR:
            continue
        dist = cv2.NORM_HAMMING if isinstance(features2d[desc_key], _binary_types) else cv2.NORM_L2
        test_combinations[f"{det_key}+{desc_key}"] = FeatureExtractor.from_opencv(
            features2d[det_key].detect,
            features2d[desc_key].compute,
            dist,
        )

# ============================================================
# DATASET LOADING
# ============================================================

def load_hpatches(path: str):
    """Load HPatches sequences. Returns list of (name, seq_type, images, homographies).
    Homographies are stored as related -> reference (inverted from files on disk)."""
    sequences = []
    for name in sorted(os.listdir(path)):
        subfolder = os.path.join(path, name)
        if not os.path.isdir(subfolder):
            continue
        if name.startswith("i_"):
            seq_type = "illumination"
        elif name.startswith("v_"):
            seq_type = "viewpoint"
        else:
            continue

        imgs, homos = [], []
        for filename in sorted(os.listdir(subfolder)):
            fp = os.path.join(subfolder, filename)
            if filename.lower().endswith(".ppm"):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs.append(img)
            elif filename.startswith("H_"):
                H = np.loadtxt(fp)
                homos.append(np.linalg.inv(H))
        if imgs and homos:
            sequences.append((name, seq_type, imgs, homos))
    return sequences


sequences = load_hpatches(HPATCHES_PATH)
print(f"Loaded {len(sequences)} sequences "
      f"({sum(1 for s in sequences if s[1]=='illumination')} illumination, "
      f"{sum(1 for s in sequences if s[1]=='viewpoint')} viewpoint)")

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def _project(pt, H):
    x, y = pt
    v = H @ np.array([x, y, 1.0])
    if abs(v[2]) < 1e-10:
        return float("inf"), float("inf")
    return v[0] / v[2], v[1] / v[2]


def _in_bounds(pt, w, h):
    x, y = pt
    return 0 <= x < w and 0 <= y < h


def _project_batch(pts_xy, H):
    """Project (N,2) array through 3x3 homography. Returns (N,2); inf for degenerate points."""
    n = len(pts_xy)
    pts_h = np.empty((n, 3), dtype=np.float64)
    pts_h[:, :2] = pts_xy
    pts_h[:, 2] = 1.0
    proj = pts_h @ H.T
    w = proj[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        xy = proj[:, :2] / w[:, np.newaxis]
    xy[np.abs(w) < 1e-10] = np.inf
    return xy


def _top_k_keypoints(kps, k):
    """Return up to k keypoints with highest response, in descending order."""
    if not kps:
        return kps
    resp = np.fromiter((kp.response for kp in kps), dtype=np.float32, count=len(kps))
    if len(kps) > k:
        part = np.argpartition(resp, -k)[-k:]
        idx  = part[np.argsort(resp[part])[::-1]]
    else:
        idx  = np.argsort(resp)[::-1]
    return [kps[i] for i in idx]


def _difficulty_tags(rel_idx):
    """Return all difficulty bucket names for a 0-based related-image index.
    HPatches rel_idx 0-4 map to img2-6; img5 (rel_idx 3) is both normal and hard."""
    diffs = []
    if rel_idx in (0, 1): diffs.append("easy")
    if rel_idx in (2, 3): diffs.append("normal")
    if rel_idx in (3, 4): diffs.append("hard")
    return diffs

# ============================================================
# AGGREGATION
# ============================================================

def aggregate(raw):
    """
    raw: list of (seq_type, img_idx, mma_kps_dict, mma_matches_dict, rep_dict, hom_acc_dict)
         seq_type in {'illumination', 'viewpoint'}
         img_idx  in {2, 3, 4, 5, 6}

    Returns wide-format rows, one per (scope, difficulty, threshold), with columns:
      mma_kps_mean/std/min/max/count      — correct / n_ref_keypoints
      mma_matches_mean/std/min/max/count  — correct / n_putative_matches
      rep_mean/std/min/max/count
      hom_acc_mean/std/min/max/count
    """
    buckets_mma_kps:     dict[tuple, list[float]] = {}
    buckets_mma_matches: dict[tuple, list[float]] = {}
    buckets_rep:         dict[tuple, list[float]] = {}
    buckets_hom_acc:     dict[tuple, list[float]] = {}

    for seq_type, img_idx, mma_kps_dict, mma_matches_dict, rep_dict, hom_acc_dict in raw:
        scope_tag = "illumination" if seq_type == "illumination" else "viewpoint"
        diffs = _difficulty_tags(img_idx - 2)

        for scope in [scope_tag]:
            for diff in diffs:
                for th, val in mma_kps_dict.items():
                    buckets_mma_kps.setdefault((scope, diff, th), []).append(val)
                for th, val in mma_matches_dict.items():
                    buckets_mma_matches.setdefault((scope, diff, th), []).append(val)
                for th, val in rep_dict.items():
                    buckets_rep.setdefault((scope, diff, th), []).append(val)
                for th, val in hom_acc_dict.items():
                    buckets_hom_acc.setdefault((scope, diff, th), []).append(val)

    all_keys = (set(buckets_mma_kps.keys()) | set(buckets_mma_matches.keys())
                | set(buckets_rep.keys()) | set(buckets_hom_acc.keys()))
    rows = []
    for (scope, difficulty, threshold) in all_keys:
        row: dict = {"transformation": scope, "difficulty": difficulty, "distance_threshold": threshold}
        for prefix, buckets in [
            ("mma_kps",     buckets_mma_kps),
            ("mma_matches", buckets_mma_matches),
            ("rep",         buckets_rep),
            ("hom_acc",     buckets_hom_acc),
        ]:
            vals = buckets.get((scope, difficulty, threshold))
            if vals is not None:
                arr = np.array(vals, dtype=np.float64)
                row[f"{prefix}_mean"]  = float(np.mean(arr))
                row[f"{prefix}_std"]   = float(np.std(arr))
                row[f"{prefix}_min"]   = float(np.min(arr))
                row[f"{prefix}_max"]   = float(np.max(arr))
                row[f"{prefix}_count"] = len(arr)
            else:
                row[f"{prefix}_mean"]  = float("nan")
                row[f"{prefix}_std"]   = float("nan")
                row[f"{prefix}_min"]   = float("nan")
                row[f"{prefix}_max"]   = float("nan")
                row[f"{prefix}_count"] = 0
        rows.append(row)
    return rows

# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

warnings.filterwarnings("once", category=UserWarning)
os.makedirs("results", exist_ok=True)

_max_k = max(MAX_KEYPOINTS)

_ds_configs = list(itertools.product(
    DOWNSAMPLE_LEVELS,
    DOWNSAMPLE_FACTOR,
    INITIAL_SIGMAS,
    INTRINSIC_SIGMA,
    APPLY_PROGRESSIVE_BLUR,
    DOWNSAMPLE_INTERPOLATION_TYPE,
))

for combo_key, extractor in tqdm(test_combinations.items(), desc="Combinations", position=0):

    for (ds_level, ds_factor, init_sigma, intr_sigma, prog_blur, interp_type) in tqdm(
            _ds_configs, desc="Downsample config", leave=False, position=1):

        scale = ds_factor ** ds_level

        # ── Initialize sweep accumulators ─────────────────────────────────────
        raw_by_sweep = {}
        match_cnt_by = {}
        for max_keypoints in MAX_KEYPOINTS:
            for use_mnn in USE_MNN:
                for ratio_th in RATIO_THRESHOLDS:
                    for ransac_threshold in RANSAC_THRESHOLDS:
                        sk = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                        raw_by_sweep[sk] = []
                        match_cnt_by[sk] = []
        keypoint_cnts_by_mf = {mf: [] for mf in MAX_KEYPOINTS}

        for name, seq_type, imgs, homos in tqdm(sequences, leave=False, desc="Sequences", position=2):
            h_ref, w_ref = imgs[0].shape[:2]
            img_ref_ds   = downsample(imgs[0], ds_level, ds_factor,
                                      intr_sigma, init_sigma, prog_blur, interp_type)

            for rel_idx, (img_rel_orig, H_rel_to_ref) in enumerate(tqdm(
                    list(zip(imgs[1:], homos)), leave=False, desc="Image pairs", position=3)):

                img_idx    = rel_idx + 2
                pair_label = f"{combo_key} ds={ds_level} seq={name} img={img_idx}"

                try:
                    H_ref_to_rel = np.linalg.inv(H_rel_to_ref)
                    h_rel, w_rel = img_rel_orig.shape[:2]
                    img_rel_ds   = downsample(img_rel_orig, ds_level, ds_factor,
                                              intr_sigma, init_sigma, prog_blur, interp_type)

                    zero_mma = {th: 0.0 for th in DISTANCE_THRESHOLDS}
                    zero_rep = {th: 0.0 for th in DISTANCE_THRESHOLDS}
                    zero_hom = {th: 0.0 for th in DISTANCE_THRESHOLDS}

                    def _append_all_zeros(rep=zero_rep):
                        for max_keypoints in MAX_KEYPOINTS:
                            for use_mnn in USE_MNN:
                                for ratio_th in RATIO_THRESHOLDS:
                                    for ransac_threshold in RANSAC_THRESHOLDS:
                                        sk = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                                        raw_by_sweep[sk].append(
                                            (seq_type, img_idx, zero_mma, zero_mma, rep, zero_hom)
                                        )
                                        match_cnt_by[sk].append(0)

                    # ── Detect keypoints in downsampled coords ────────────────
                    kps_ref_all = extractor.detect_keypoints(img_ref_ds)
                    kps_rel_all = extractor.detect_keypoints(img_rel_ds)

                    # Scale to original coords for geometric filtering
                    if scale != 1:
                        for kp in kps_ref_all:
                            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                        for kp in kps_rel_all:
                            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

                    # Keep only keypoints visible in the other image (vectorized)
                    if kps_ref_all:
                        _pts = np.array([kp.pt for kp in kps_ref_all], dtype=np.float64)
                        _proj = _project_batch(_pts, H_ref_to_rel)
                        _mask = (_proj[:, 0] >= 0) & (_proj[:, 0] < w_rel) & (_proj[:, 1] >= 0) & (_proj[:, 1] < h_rel)
                        kps_ref_all = [kp for kp, m in zip(kps_ref_all, _mask) if m]
                    if kps_rel_all:
                        _pts = np.array([kp.pt for kp in kps_rel_all], dtype=np.float64)
                        _proj = _project_batch(_pts, H_rel_to_ref)
                        _mask = (_proj[:, 0] >= 0) & (_proj[:, 0] < w_ref) & (_proj[:, 1] >= 0) & (_proj[:, 1] < h_ref)
                        kps_rel_all = [kp for kp, m in zip(kps_rel_all, _mask) if m]
                    kps_ref_all = _top_k_keypoints(kps_ref_all, _max_k)
                    kps_rel_all = _top_k_keypoints(kps_rel_all, _max_k)

                    if not kps_ref_all or not kps_rel_all:
                        _append_all_zeros()
                        continue

                    # Scale back to downsampled coords for description
                    if scale != 1:
                        for kp in kps_ref_all:
                            kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                        for kp in kps_rel_all:
                            kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

                    kps_ref_all, descs_ref_all = extractor.describe_keypoints(img_ref_ds, kps_ref_all)
                    kps_rel_all, descs_rel_all = extractor.describe_keypoints(img_rel_ds, kps_rel_all)

                    # Scale back to original coords for all downstream math
                    if scale != 1:
                        for kp in kps_ref_all:
                            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                        for kp in kps_rel_all:
                            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

                    if not descs_ref_all or not descs_rel_all:
                        _append_all_zeros()
                        continue

                    dtype         = np.float32 if extractor.distance_type == cv2.NORM_L2 else np.uint8
                    desc_ref_full = np.array(descs_ref_all, dtype=dtype)
                    desc_rel_full = np.array(descs_rel_all, dtype=dtype)
                    bf            = cv2.BFMatcher(extractor.distance_type, crossCheck=False)

                    corners    = np.array([[0, 0], [w_ref-1, 0],
                                           [w_ref-1, h_ref-1], [0, h_ref-1]], dtype=np.float64)
                    corners_gt = _project_batch(corners, H_ref_to_rel)

                    for max_keypoints in MAX_KEYPOINTS:
                        kps_ref  = kps_ref_all[:max_keypoints]
                        kps_rel  = kps_rel_all[:max_keypoints]
                        n_ref, n_rel = len(kps_ref), len(kps_rel)
                        desc_ref = desc_ref_full[:n_ref]
                        desc_rel = desc_rel_full[:n_rel]
                        keypoint_cnts_by_mf[max_keypoints].extend([n_ref, n_rel])

                        if not kps_ref or not kps_rel:
                            for use_mnn in USE_MNN:
                                for ratio_th in RATIO_THRESHOLDS:
                                    for ransac_threshold in RANSAC_THRESHOLDS:
                                        sk = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                                        raw_by_sweep[sk].append(
                                            (seq_type, img_idx, zero_mma, zero_rep, zero_hom)
                                        )
                                        match_cnt_by[sk].append(0)
                            continue

                        # ── Repeatability (descriptor-free, once per max_features) ──
                        ref_pts      = np.array([kp.pt for kp in kps_ref], dtype=np.float64)
                        rel_pts      = np.array([kp.pt for kp in kps_rel], dtype=np.float64)
                        ref_proj_rel = _project_batch(ref_pts, H_ref_to_rel)
                        rel_proj_ref = _project_batch(rel_pts, H_rel_to_ref)
                        dists_ref    = np.min(np.linalg.norm(ref_proj_rel[:, None] - rel_pts[None], axis=2), axis=1)
                        dists_rel    = np.min(np.linalg.norm(rel_proj_ref[:, None] - ref_pts[None], axis=2), axis=1)
                        rep = {th: float(np.sum(dists_ref < th) + np.sum(dists_rel < th)) / (n_ref + n_rel)
                               for th in DISTANCE_THRESHOLDS}

                       # ── KNN matching (k=2 enables ratio test) ─────────────
                        k_nn    = min(2, n_rel)
                        knn_raw = bf.knnMatch(desc_ref, desc_rel, k=k_nn)

                        for use_mnn in USE_MNN:
                            for ratio_th in RATIO_THRESHOLDS:

                                def apply_ratio(knn_pairs):
                                    if ratio_th is not None and k_nn >= 2:
                                        return [
                                            pair for pair in knn_pairs
                                            if (len(pair) >= 2 and pair[0].distance < ratio_th * pair[1].distance)
                                        ]
                                    else:
                                        return list(knn_pairs)

                                def apply_mnn(knn_pairs):
                                    if not use_mnn:
                                        return knn_pairs

                                    # Build NN maps from CURRENT candidate set
                                    nn12 = {}  # queryIdx -> best trainIdx
                                    nn21 = {}  # trainIdx -> best queryIdx

                                    for pair in knn_pairs:
                                        if len(pair) == 0:
                                            continue
                                        m = pair[0]

                                        # forward NN
                                        if m.queryIdx not in nn12 or m.distance < nn12[m.queryIdx][1]:
                                            nn12[m.queryIdx] = (m.trainIdx, m.distance)

                                        # reverse NN
                                        if m.trainIdx not in nn21 or m.distance < nn21[m.trainIdx][1]:
                                            nn21[m.trainIdx] = (m.queryIdx, m.distance)

                                    # keep only mutual matches
                                    return [
                                        pair for pair in knn_pairs
                                        if (
                                            len(pair) > 0
                                            and nn12.get(pair[0].queryIdx, (None,))[0] == pair[0].trainIdx
                                            and nn21.get(pair[0].trainIdx, (None,))[0] == pair[0].queryIdx
                                        )
                                    ]

                                # ordering switch
                                if RATIO_FIRST:
                                    knn_tmp = apply_ratio(knn_raw)
                                    knn_filtered = apply_mnn(knn_tmp)
                                else:
                                    knn_tmp = apply_mnn(knn_raw)
                                    knn_filtered = apply_ratio(knn_tmp)


                                # if not knn_ratio:
                                #     for ransac_threshold in RANSAC_THRESHOLDS:
                                #         sk = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                                #         raw_by_sweep[sk].append(
                                #             (seq_type, img_idx, zero_mma, zero_mma, rep, zero_hom)
                                #         )
                                #         match_cnt_by[sk].append(0)
                                #     continue

                                if not knn_filtered:
                                    for ransac_threshold in RANSAC_THRESHOLDS:
                                        sk = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                                        raw_by_sweep[sk].append(
                                            (seq_type, img_idx, zero_mma, zero_mma, rep, zero_hom)
                                        )
                                        match_cnt_by[sk].append(0)
                                    continue

                                # ── MMA (computed on filtered pairs, consistent with pipeline) ─────
                                _q = [pair[0].queryIdx for pair in knn_filtered]
                                _t = [pair[0].trainIdx for pair in knn_filtered]
                                _src = np.array([kps_ref[i].pt for i in _q], dtype=np.float64)
                                _dst = np.array([kps_rel[i].pt for i in _t], dtype=np.float64)
                                errors = np.linalg.norm(_project_batch(_src, H_ref_to_rel) - _dst, axis=1)

                                n_putative = len(errors)
                                mma_kps     = {th: float(np.sum(errors < th)) / n_ref      for th in DISTANCE_THRESHOLDS}
                                mma_matches = {th: float(np.sum(errors < th)) / n_putative for th in DISTANCE_THRESHOLDS}

                                # ── Extract matches (ratio already applied!) ─────────
                                matches = [pair[0] for pair in knn_filtered]

                                can_ransac = len(matches) >= 4
                                if can_ransac:
                                    src = np.float32([kps_ref[m.queryIdx].pt for m in matches])
                                    dst = np.float32([kps_rel[m.trainIdx].pt for m in matches])

                                for ransac_threshold in RANSAC_THRESHOLDS:
                                    # ── Homography estimation ───────────────────────
                                    if can_ransac:
                                        H_est, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
                                        if H_est is not None:
                                            corners_est = _project_batch(corners, H_est)
                                            mean_err = float(np.mean(np.linalg.norm(
                                                corners_gt - corners_est, axis=1
                                            )))
                                            hom_acc = {
                                                th: 1.0 if mean_err < th else 0.0
                                                for th in DISTANCE_THRESHOLDS
                                            }
                                        else:
                                            hom_acc = zero_hom
                                    else:
                                        hom_acc = zero_hom

                                    sk = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                                    raw_by_sweep[sk].append(
                                        (seq_type, img_idx, mma_kps, mma_matches, rep, hom_acc)
                                    )
                                    match_cnt_by[sk].append(len(matches))

                except Exception:
                    if SKIP_AT_ERROR:
                        with open("failed_combinations.txt", "a") as f:
                            f.write(f"{pair_label}\n")
                            f.write(traceback.format_exc() + "\n\n")
                    else:
                        raise

        # ── Write results to CSV ──────────────────────────────────────────────
        for max_keypoints in MAX_KEYPOINTS:
            for use_mnn in USE_MNN:
                for ratio_th in RATIO_THRESHOLDS:
                    for ransac_threshold in RANSAC_THRESHOLDS:
                        sk           = (max_keypoints, use_mnn, ratio_th, ransac_threshold)
                        raw_pairs    = raw_by_sweep[sk]
                        match_counts = match_cnt_by[sk]
                        keypoint_counts  = keypoint_cnts_by_mf[max_keypoints]

                        common = {
                            # ── Identity ───────────────────────────────────────────────
                            "method":                   combo_key,
                            "tag":                           RUN_TAG,
                            # ── Downsampling ────────────────────────────────────────────
                            "downsample_level":              ds_level,
                            #"downsample_factor":             ds_factor,
                            "initial_sigma":                 init_sigma,
                            "intrinsic_sigma":               intr_sigma,
                            "apply_progressive_blur":        prog_blur,
                            #"downsample_interpolation_type": str(interp_type),
                            # ── Pipeline parameters ─────────────────────────────────────
                            "max_keypoints":                  max_keypoints,
                            "use_mnn":                       use_mnn,
                            "ratio_threshold":               ratio_th if ratio_th is not None else float("nan"),
                            "ransac_threshold":                 ransac_threshold,
                            # ── Run statistics ──────────────────────────────────────────
                            "avg_num_keypoints":        float(np.mean(keypoint_counts))  if keypoint_counts  else 0.0,
                            "avg_num_missing_keypoints": float(np.mean([max_keypoints - c for c in keypoint_counts])) if keypoint_counts else 0.0,
                            "avg_num_matches":         float(np.mean(match_counts)) if match_counts else 0.0,
                            "avg_num_dropped_matches": float(np.mean([kp - m for kp, m in zip(keypoint_counts, match_counts)])) if match_counts else 0.0,

                        }

                        metric_rows = aggregate(raw_pairs)
                        rows = [{**common, **row} for row in metric_rows]

                        df_out = pd.DataFrame(rows)
                        write_header = not os.path.isfile(RESULTS_FILE)
                        df_out.to_csv(RESULTS_FILE, index=False, header=write_header, mode="a")

print("\nDone. Results written to", RESULTS_FILE)
