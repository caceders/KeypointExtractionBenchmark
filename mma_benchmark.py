import os
import itertools
import traceback
import warnings

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import downsample
from matchers import get_matches

try:
    from shi_tomasi_sift import ShiTomasiSift
except ImportError:
    ShiTomasiSift = None

# ============================================================
# CONFIGURATION
# ============================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
HPATCHES_PATH = r"hpatches-sequences-release"
RESULTS_FILE  = "mma_results/optimize.csv"

# ── Run tag ───────────────────────────────────────────────────────────────────
# Label for this entire benchmark run. All combinations share this tag.
# Use a different tag for each run you want to compare in display_mma.py.
RUN_TAG = "pos"

# ── Feature combinations ──────────────────────────────────────────────────────
features2d = {
    # "SIFT":      cv2.SIFT_create(),
    # "ORB":       cv2.ORB_create(nfeatures=5000),
    # "BRISK":     cv2.BRISK_create(),
    # "AKAZE":     cv2.AKAZE_create(),
    # "GFTT":      cv2.GFTTDetector_create(maxCorners=5000),
    ## LOW THRESH
    "SIFT":      cv2.SIFT_create(contrastThreshold = 0.0001),
    "ORB":       cv2.ORB_create(nfeatures=5000, edgeThreshold = 1, fastThreshold = 3),
    "BRISK":     cv2.BRISK_create(thresh = 1),
    "AKAZE":     cv2.AKAZE_create(threshold=0.000000001),
    "GFTT":      cv2.GFTTDetector_create(maxCorners=5000, qualityLevel = 0.0002),
}

ONLY_SELF             = True
ONLY_SELF_EXCEPTIONS  = [("GFTT", "SIFT")]
ONLY_USED_AS_DETECTOR = ["GFTT"]

# ── Evaluation thresholds ─────────────────────────────────────────────────────
# Pixel-error thresholds for MMA, repeatability, and homography accuracy.
DISTANCE_THRESHOLDS = [10]

# ── Matching parameters ───────────────────────────────────────────────────────
# Each parameter is a list; combinations are benchmarked and stored in the CSV.
MAX_KEYPOINTS    = [500]
MATCHERS         = ["KEEM", "MNN"]  # "NN", "MNN", "KEEM"
RATIO_THRESHOLDS = [0.8]   # applied to NN (unidirectional) and MNN (bidirectional); ignored for KEEM
RANSAC_THRESHOLDS = [3.0]

# ── Downsampling parameters ───────────────────────────────────────────────────
DOWNSAMPLE_LEVELS             = [0]
INITIAL_SIGMAS                = [0]
DOWNSAMPLE_FACTOR             = [2]
DOWNSAMPLE_INTERPOLATION_TYPE = [None]
INTRINSIC_SIGMA               = [0.5]
APPLY_PROGRESSIVE_BLUR        = [False]

VISIBILITY_FILTERS = [False]  # sweepable; True removes kps that project outside the other image

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

# (matcher, ratio_th) sweep: KEEM gets None (no ratio test), NN/MNN get each ratio threshold
_matching_configs: list[tuple[str, float | None]] = []
for _m in MATCHERS:
    if _m == "KEEM":
        _matching_configs.append(("KEEM", None))
    else:
        for _r in RATIO_THRESHOLDS:
            _matching_configs.append((_m, _r))


# ============================================================
# DATASET LOADING
# ============================================================

def load_hpatches(path: str):
    """Load HPatches sequences. Returns list of (name, seq_type, images, homographies).
    Homographies are stored as reference -> related (as on disk)."""
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
                homos.append(H)
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
    """Return difficulty bucket names for a 0-based related-image index.
    HPatches rel_idx 0-4 → img2-6; img5 (rel_idx 3) falls in both normal and hard."""
    diffs = []
    if rel_idx in (0, 1): diffs.append("easy")
    if rel_idx in (2, 3): diffs.append("normal")
    if rel_idx in (3, 4): diffs.append("hard")
    return diffs


# ============================================================
# mAP HELPERS
# ============================================================

def compute_ap(match_pool: list[tuple[float, float]], threshold: float) -> float:
    """AP on matches ranked by descriptor distance, labelled correct if geo_error < threshold."""
    if not match_pool:
        return 0.0
    distances = np.array([d for d, _ in match_pool])
    errors    = np.array([e for _, e in match_pool])
    labels    = (errors < threshold).astype(int)
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, -distances))


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

warnings.filterwarnings("once", category=UserWarning)
_results_dir = os.path.dirname(RESULTS_FILE)
if _results_dir:
    os.makedirs(_results_dir, exist_ok=True)

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

        for seq_id, (seq_name, seq_type, imgs, homos) in enumerate(tqdm(
                sequences, leave=False, desc="Sequences", position=2)):

            h_ref, w_ref = imgs[0].shape[:2]
            img_ref_ds   = downsample(imgs[0], ds_level, ds_factor,
                                      intr_sigma, init_sigma, prog_blur, interp_type)

            # ── Detect + describe reference image ONCE for all pairs ──────────────────
            dtype        = np.float32 if extractor.distance_type == cv2.NORM_L2 else np.uint8
            kps_ref_base = extractor.detect_keypoints(img_ref_ds)
            kps_ref_base = _top_k_keypoints(kps_ref_base, _max_k)
            if kps_ref_base:
                kps_ref_base, _dref = extractor.describe_keypoints(img_ref_ds, kps_ref_base)
                descs_ref_np = np.array(_dref, dtype=dtype) if _dref else None
                if scale != 1:
                    for kp in kps_ref_base:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
            else:
                descs_ref_np = None

            _need_inv_h = any(VISIBILITY_FILTERS)

            seq_rows: list[dict] = []
            # match_pool[(max_kp, matcher, ratio_th, vis_filter, diff)] → per-difficulty pool
            # match_pool_tot[(max_kp, matcher, ratio_th, vis_filter)]   → full-sequence pool
            match_pool:     dict[tuple, list[tuple[float, float]]] = {}
            match_pool_tot: dict[tuple, list[tuple[float, float]]] = {}

            for rel_idx, (img_rel_orig, H_ref_to_rel) in enumerate(tqdm(
                    list(zip(imgs[1:], homos)), leave=False, desc="Image pairs", position=3)):

                img_idx    = rel_idx + 2
                diffs      = _difficulty_tags(rel_idx)
                pair_label = f"{combo_key} ds={ds_level} seq={seq_name} img={img_idx}"

                try:
                    h_rel, w_rel = img_rel_orig.shape[:2]
                    img_rel_ds   = downsample(img_rel_orig, ds_level, ds_factor,
                                              intr_sigma, init_sigma, prog_blur, interp_type)

                    # ── Detect + describe related image ────────────────────────────────
                    kps_rel_base = extractor.detect_keypoints(img_rel_ds)
                    kps_rel_base = _top_k_keypoints(kps_rel_base, _max_k)
                    if kps_rel_base:
                        kps_rel_base, _drel = extractor.describe_keypoints(img_rel_ds, kps_rel_base)
                        descs_rel_np = np.array(_drel, dtype=dtype) if _drel else None
                        if scale != 1:
                            for kp in kps_rel_base:
                                kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                    else:
                        descs_rel_np = None

                    corners    = np.array([[0, 0], [w_ref - 1, 0],
                                           [w_ref - 1, h_ref - 1], [0, h_ref - 1]], dtype=np.float64)
                    corners_gt = _project_batch(corners, H_ref_to_rel)

                    # ── Pre-compute visibility masks once per pair ─────────────────────
                    H_rel_to_ref = np.linalg.inv(H_ref_to_rel) if _need_inv_h else None
                    _vis_ref: dict[bool, np.ndarray | None] = {}
                    _vis_rel: dict[bool, np.ndarray | None] = {}
                    for _vf in set(VISIBILITY_FILTERS):
                        if _vf:
                            if kps_ref_base:
                                _pts = np.array([kp.pt for kp in kps_ref_base], dtype=np.float64)
                                _proj = _project_batch(_pts, H_ref_to_rel)
                                _vis_ref[_vf] = ((_proj[:, 0] >= 0) & (_proj[:, 0] < w_rel) &
                                                 (_proj[:, 1] >= 0) & (_proj[:, 1] < h_rel))
                            else:
                                _vis_ref[_vf] = None
                            if kps_rel_base:
                                _pts = np.array([kp.pt for kp in kps_rel_base], dtype=np.float64)
                                _proj = _project_batch(_pts, H_rel_to_ref)
                                _vis_rel[_vf] = ((_proj[:, 0] >= 0) & (_proj[:, 0] < w_ref) &
                                                 (_proj[:, 1] >= 0) & (_proj[:, 1] < h_ref))
                            else:
                                _vis_rel[_vf] = None
                        else:
                            _vis_ref[_vf] = None
                            _vis_rel[_vf] = None

                    for vis_filter in VISIBILITY_FILTERS:
                        # ── Select visible keypoint subset ─────────────────────────────
                        ref_mask = _vis_ref[vis_filter]
                        rel_mask = _vis_rel[vis_filter]

                        if ref_mask is not None:
                            kps_ref_all  = [kp for kp, m in zip(kps_ref_base, ref_mask) if m]
                            descs_ref_vf = descs_ref_np[ref_mask] if descs_ref_np is not None else None
                        else:
                            kps_ref_all  = kps_ref_base
                            descs_ref_vf = descs_ref_np

                        if rel_mask is not None:
                            kps_rel_all  = [kp for kp, m in zip(kps_rel_base, rel_mask) if m]
                            descs_rel_vf = descs_rel_np[rel_mask] if descs_rel_np is not None else None
                        else:
                            kps_rel_all  = kps_rel_base
                            descs_rel_vf = descs_rel_np

                        have_kps   = bool(kps_ref_all) and bool(kps_rel_all)
                        have_descs = (descs_ref_vf is not None and len(descs_ref_vf) > 0 and
                                      descs_rel_vf is not None and len(descs_rel_vf) > 0)

                        for max_kp in MAX_KEYPOINTS:
                            kps_ref = kps_ref_all[:max_kp] if have_kps else []
                            kps_rel = kps_rel_all[:max_kp] if have_kps else []
                            n_ref   = len(kps_ref)
                            n_rel   = len(kps_rel)

                            # ── Repeatability ─────────────────────────────────────────
                            if n_ref and n_rel:
                                ref_pts      = np.array([kp.pt for kp in kps_ref], dtype=np.float64)
                                rel_pts      = np.array([kp.pt for kp in kps_rel], dtype=np.float64)
                                ref_proj_rel = _project_batch(ref_pts, H_ref_to_rel)
                                dist_matrix  = np.linalg.norm(ref_proj_rel[:, None] - rel_pts[None], axis=2)
                                dists_ref    = np.min(dist_matrix, axis=1)
                                dists_rel    = np.min(dist_matrix, axis=0)
                                rep = {th: float(np.sum(dists_ref < th) + np.sum(dists_rel < th)) / (n_ref + n_rel)
                                       for th in DISTANCE_THRESHOLDS}
                            else:
                                rep = {th: 0.0 for th in DISTANCE_THRESHOLDS}

                            if have_descs and n_ref > 0 and n_rel > 0:
                                desc_ref = descs_ref_vf[:n_ref]
                                desc_rel = descs_rel_vf[:n_rel]
                            else:
                                desc_ref = None
                                desc_rel = None

                            for matcher, ratio_th in _matching_configs:
                                # ── Match descriptors ──────────────────────────────────
                                if desc_ref is not None:
                                    raw_matches = get_matches(desc_ref, desc_rel,
                                                              extractor.distance_type, matcher, ratio_th)
                                else:
                                    raw_matches = []

                                n_matches = len(raw_matches)

                                if n_matches > 0:
                                    _q         = [m.query_idx for m in raw_matches]
                                    _t         = [m.train_idx for m in raw_matches]
                                    _src       = np.array([kps_ref[i].pt for i in _q], dtype=np.float64)
                                    _dst       = np.array([kps_rel[i].pt for i in _t], dtype=np.float64)
                                    geo_errors = np.linalg.norm(_project_batch(_src, H_ref_to_rel) - _dst, axis=1)
                                else:
                                    geo_errors = np.array([], dtype=np.float64)

                                # ── Accumulate match pools for per-sequence mAP ────────
                                pool_tot_key = (max_kp, matcher, ratio_th, vis_filter)
                                if pool_tot_key not in match_pool_tot:
                                    match_pool_tot[pool_tot_key] = []
                                for m, err in zip(raw_matches, geo_errors):
                                    match_pool_tot[pool_tot_key].append((m.distance, float(err)))

                                for diff in diffs:
                                    pool_key = (max_kp, matcher, ratio_th, vis_filter, diff)
                                    if pool_key not in match_pool:
                                        match_pool[pool_key] = []
                                    for m, err in zip(raw_matches, geo_errors):
                                        match_pool[pool_key].append((m.distance, float(err)))

                                # ── Per-pair metrics ───────────────────────────────────
                                mma_kps = (
                                    {th: float(np.sum(geo_errors < th)) / n_ref for th in DISTANCE_THRESHOLDS}
                                    if n_ref > 0 else {th: 0.0 for th in DISTANCE_THRESHOLDS}
                                )
                                mma_matches = (
                                    {th: float(np.sum(geo_errors < th)) / n_matches for th in DISTANCE_THRESHOLDS}
                                    if n_matches > 0 else {th: 0.0 for th in DISTANCE_THRESHOLDS}
                                )

                                # ── Homography estimation ──────────────────────────────
                                hom_accs: dict[float, dict[int, float]] = {}
                                can_ransac = n_matches >= 4
                                if can_ransac:
                                    _q   = [m.query_idx for m in raw_matches]
                                    _t   = [m.train_idx for m in raw_matches]
                                    src  = np.float32([kps_ref[i].pt for i in _q])
                                    dst  = np.float32([kps_rel[i].pt for i in _t])
                                    for ransac_th in RANSAC_THRESHOLDS:
                                        H_est, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_th)
                                        if H_est is not None:
                                            corners_est = _project_batch(corners, H_est)
                                            mean_err    = float(np.mean(np.linalg.norm(corners_gt - corners_est, axis=1)))
                                            hom_accs[ransac_th] = {th: 1.0 if mean_err < th else 0.0
                                                                   for th in DISTANCE_THRESHOLDS}
                                        else:
                                            hom_accs[ransac_th] = {th: 0.0 for th in DISTANCE_THRESHOLDS}
                                else:
                                    for ransac_th in RANSAC_THRESHOLDS:
                                        hom_accs[ransac_th] = {th: 0.0 for th in DISTANCE_THRESHOLDS}

                                ratio_th_csv = ratio_th if ratio_th is not None else float("nan")

                                for ransac_th in RANSAC_THRESHOLDS:
                                    for diff in diffs:
                                        for dist_th in DISTANCE_THRESHOLDS:
                                            seq_rows.append({
                                                # Identity
                                                "method":                 combo_key,
                                                "tag":                    RUN_TAG,
                                                # Matching parameters
                                                "matcher":                matcher,
                                                "ratio_threshold":        ratio_th_csv,
                                                "ransac_threshold":       ransac_th,
                                                # Pipeline parameters
                                                "max_keypoints":          max_kp,
                                                "downsample_level":       ds_level,
                                                "initial_sigma":          init_sigma,
                                                "intrinsic_sigma":        intr_sigma,
                                                "apply_progressive_blur": prog_blur,
                                                "visibility_filter":      vis_filter,
                                                # Sequence
                                                "seq_name":               seq_name,
                                                "seq_id":                 seq_id,
                                                "seq_type":               seq_type,
                                                # Image pair
                                                "img_idx":                img_idx,
                                                "difficulty":             diff,
                                                "distance_threshold":     dist_th,
                                                # Metrics (mAP filled after full sequence)
                                                "mma_kps":                mma_kps[dist_th],
                                                "mma_matches":            mma_matches[dist_th],
                                                "rep":                    rep[dist_th],
                                                "hom_acc":                hom_accs[ransac_th][dist_th],
                                                "mAP":                    None,
                                                "mAP_tot":                None,
                                                # Counts
                                                "num_keypoints_ref":      n_ref,
                                                "num_keypoints_rel":      n_rel,
                                                "num_matches":            n_matches,
                                            })

                except Exception:
                    if SKIP_AT_ERROR:
                        with open("failed_combinations.txt", "a") as f:
                            f.write(f"{pair_label}\n")
                            f.write(traceback.format_exc() + "\n\n")
                    else:
                        raise

            # ── Compute per-sequence mAP and fill into rows ───────────────────────────
            mAP_cache:     dict[tuple, float] = {}
            mAP_tot_cache: dict[tuple, float] = {}

            for (max_kp, matcher, ratio_th, vis_filter, diff), pool in match_pool.items():
                for dist_th in DISTANCE_THRESHOLDS:
                    mAP_cache[(max_kp, matcher, ratio_th, vis_filter, diff, dist_th)] = compute_ap(pool, dist_th)

            for (max_kp, matcher, ratio_th, vis_filter), pool in match_pool_tot.items():
                for dist_th in DISTANCE_THRESHOLDS:
                    mAP_tot_cache[(max_kp, matcher, ratio_th, vis_filter, dist_th)] = compute_ap(pool, dist_th)

            for row in seq_rows:
                _rt = row["ratio_threshold"]
                _rt = None if (isinstance(_rt, float) and np.isnan(_rt)) else _rt
                _vf = row["visibility_filter"]
                row["mAP"] = mAP_cache.get(
                    (row["max_keypoints"], row["matcher"], _rt, _vf, row["difficulty"], row["distance_threshold"]),
                    float("nan"),
                )
                row["mAP_tot"] = mAP_tot_cache.get(
                    (row["max_keypoints"], row["matcher"], _rt, _vf, row["distance_threshold"]),
                    float("nan"),
                )

            # ── Write this sequence's rows to CSV ─────────────────────────────────────
            if seq_rows:
                df_out = pd.DataFrame(seq_rows)
                write_header = not os.path.isfile(RESULTS_FILE)
                df_out.to_csv(RESULTS_FILE, index=False, header=write_header, mode="a")

print("\nDone. Results written to", RESULTS_FILE)
