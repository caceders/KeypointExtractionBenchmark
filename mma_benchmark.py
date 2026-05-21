import warnings
import traceback
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from shi_tomasi_sift import ShiTomasiSift
from matchers import match_nn, match_mnn, match_keem, apply_ratio_uni, apply_ratio_fwd, apply_ratio_bi
from benchmark.utils import downsample
from benchmark.feature_extractor import FeatureExtractor

# ============================================================
# CONFIGURATION
# ============================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
HPATCHES_PATH = r"hpatches-sequences-release"
RESULTS_FILE  = "mma_results/RANSAC_confidence.csv"

# ── Run tag ───────────────────────────────────────────────────────────────────
RUN_TAG = "995"

# ── Feature combinations ──────────────────────────────────────────────────────
features2d = {
    # "SIFT":      cv2.SIFT_create(),
    # "ORB":       cv2.ORB_create(nfeatures=5000),
    # "BRISK":     cv2.BRISK_create(),
    # "AKAZE":     cv2.AKAZE_create(),
    # "GFTT":      cv2.GFTTDetector_create(maxCorners=5000),
    ## LOW THRESH
    "SIFT":        cv2.SIFT_create(contrastThreshold = 0.0001),
    #"ORB":         cv2.ORB_create(nfeatures=5000, edgeThreshold = 1, fastThreshold = 3),
    "BRISK":       cv2.BRISK_create(thresh = 1),
    "AKAZE":       cv2.AKAZE_create(threshold=0.000000001),
    "GFTT":        cv2.GFTTDetector_create(maxCorners=5000, qualityLevel = 0.0002),
}

ONLY_SELF             = True
ONLY_SELF_EXCEPTIONS  = [("GFTT", "SIFT")]
ONLY_USED_AS_DETECTOR = ["GFTT"]

# ── Evaluation thresholds ─────────────────────────────────────────────────────
DISTANCE_THRESHOLDS = list(range(1, 31))

# ── Matching parameters ───────────────────────────────────────────────────────
MAX_KEYPOINTS    = [500]
MATCHERS         = ["MNN", "NN"]  # "NN", "MNN", "KEEM"
RATIO_THRESHOLDS  = [0.8]  # applied to NN and MNN; ignored for KEEM
MNN_BIDIRECTIONAL = [True, False]  # True: bidirectional ratio test for MNN; False: unidirectional (same as NN)
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

RESULTS_FILE = f"mma_results/{RUN_NAME}.csv"
os.makedirs("mma_results", exist_ok=True)


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

# (matcher, ratio_th, bidirectional) — KEEM: no ratio test; MNN: sweeps MNN_BIDIRECTIONAL; NN: always unidirectional
_matching_configs: list[tuple[str, float | None, bool | None]] = []
for _m in MATCHERS:
    if _m == "KEEM":
        _matching_configs.append(("KEEM", None, None))
    elif _m == "MNN":
        for _r in RATIO_THRESHOLDS:
            for _bi in MNN_BIDIRECTIONAL:
                _matching_configs.append(("MNN", _r, _bi))
    else:
        for _r in RATIO_THRESHOLDS:
            _matching_configs.append((_m, _r, None))


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
    resp = np.array([kp.response for kp in kps], dtype=np.float32)
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
    """AP on matches ranked by ascending descriptor distance, labelled correct if geo_error < threshold."""
    if not match_pool:
        return 0.0
    pool      = np.asarray(match_pool, dtype=np.float64)   # (N, 2): [distance, error]
    y         = pool[:, 1] < threshold
    if not y.any():
        return 0.0
    order     = np.argsort(pool[:, 0])                     # ascending distance = descending score
    y_sorted  = y[order].astype(np.float64)
    n_pos     = y_sorted.sum()
    precision = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    return float((precision * y_sorted).sum() / n_pos)


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

warnings.filterwarnings("once", category=UserWarning)

_max_k = max(MAX_KEYPOINTS)

_ds_factor   = DOWNSAMPLE_FACTOR[0]
_intr_sigma  = INTRINSIC_SIGMA[0]
_prog_blur   = APPLY_PROGRESSIVE_BLUR[0]
_interp_type = DOWNSAMPLE_INTERPOLATION_TYPE[0]

for combo_key, extractor in tqdm(test_combinations.items(), desc="Methods", leave=True, position=0):

    for init_sigma in tqdm(INITIAL_SIGMAS, desc="Initial sigmas", leave=False, position=1):

        for ds_level in tqdm(DOWNSAMPLE_LEVELS, desc="Downsample levels", leave=False, position=2):

            ds_factor   = _ds_factor
            intr_sigma  = _intr_sigma
            prog_blur   = _prog_blur
            interp_type = _interp_type
            scale = ds_factor ** ds_level

            # ── Aggregate across all sequences ────────────────────────────────────────
            # key: (transformation, matcher, ratio_th_csv, ransac_th, max_kp, vis_filter, dist_th)
            agg_sums:  dict[tuple, dict[str, float]] = {}
            agg_count: dict[tuple, int]              = {}
            # key: (transformation, max_kp, matcher, ratio_th, vis_filter)
            agg_pool:  dict[tuple, list]             = {}

            for seq_id, (seq_name, seq_type, imgs, homos) in enumerate(tqdm(
                    sequences, leave=False, desc="Sequences", position=3)):

                h_ref, w_ref = imgs[0].shape[:2]
                img_ref_ds   = downsample(imgs[0], ds_level, ds_factor,
                                          intr_sigma, init_sigma, prog_blur, interp_type)

                # ── Detect + describe reference image ONCE for all pairs ──────────────────
                dtype        = np.float32 if extractor.distance_type == cv2.NORM_L2 else np.uint8
                kps_ref_base     = extractor.detect_keypoints(img_ref_ds)
                kps_ref_base     = _top_k_keypoints(kps_ref_base, _max_k)
                n_kps_ref_pool   = len(kps_ref_base)
                if kps_ref_base:
                    kps_ref_base, _dref = extractor.describe_keypoints(img_ref_ds, kps_ref_base)
                    descs_ref_np = np.array(_dref, dtype=dtype) if _dref else None
                    if scale != 1:
                        for kp in kps_ref_base:
                            kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                else:
                    descs_ref_np = None

                _need_inv_h = any(VISIBILITY_FILTERS)

                for rel_idx, (img_rel_orig, H_ref_to_rel) in enumerate(tqdm(
                        list(zip(imgs[1:], homos)), leave=False, desc="Image pairs", position=4)):

                    img_idx    = rel_idx + 2
                    pair_label = f"{combo_key} ds={ds_level} seq={seq_name} img={img_idx}"

                    try:
                        h_rel, w_rel = img_rel_orig.shape[:2]
                        img_rel_ds   = downsample(img_rel_orig, ds_level, ds_factor,
                                                  intr_sigma, init_sigma, prog_blur, interp_type)

                        # ── Detect + describe related image ────────────────────────────────
                        kps_rel_base     = extractor.detect_keypoints(img_rel_ds)
                        kps_rel_base     = _top_k_keypoints(kps_rel_base, _max_k)
                        n_kps_rel_pool   = len(kps_rel_base)
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

                                # ── Pre-extract keypoint coordinates once per (max_kp, vis_filter) ──
                                ref_pts_arr = np.array([kp.pt for kp in kps_ref], dtype=np.float64) if n_ref else None
                                rel_pts_arr = np.array([kp.pt for kp in kps_rel], dtype=np.float64) if n_rel else None

                                # ── Repeatability ─────────────────────────────────────────
                                if n_ref and n_rel:
                                    ref_proj_rel = _project_batch(ref_pts_arr, H_ref_to_rel)
                                    _d           = ref_proj_rel[:, None, :] - rel_pts_arr[None, :, :]
                                    sq_dist      = (_d * _d).sum(axis=2)
                                    sq_dref      = sq_dist.min(axis=1)
                                    sq_drel      = sq_dist.min(axis=0)
                                    rep = {th: float(np.sum(sq_dref < th * th) + np.sum(sq_drel < th * th)) / (n_ref + n_rel)
                                           for th in DISTANCE_THRESHOLDS}
                                else:
                                    rep = {th: 0.0 for th in DISTANCE_THRESHOLDS}

                                if have_descs and n_ref > 0 and n_rel > 0:
                                    desc_ref = descs_ref_vf[:n_ref]
                                    desc_rel = descs_rel_vf[:n_rel]
                                else:
                                    desc_ref = None
                                    desc_rel = None

                                _raw_match_cache: dict[str, list] = {}
                                for matcher, ratio_th, bidirectional in _matching_configs:
                                    # ── Match descriptors ──────────────────────────────────
                                    if desc_ref is not None:
                                        if matcher not in _raw_match_cache:
                                            if matcher == "NN":
                                                _raw_match_cache[matcher] = match_nn(desc_ref, desc_rel, extractor.distance_type)
                                            elif matcher == "MNN":
                                                _raw_match_cache[matcher] = match_mnn(desc_ref, desc_rel, extractor.distance_type)
                                            else:
                                                _raw_match_cache[matcher] = match_keem(desc_ref, desc_rel, extractor.distance_type)
                                        _raw = _raw_match_cache[matcher]
                                        if matcher == "NN":
                                            raw_matches = apply_ratio_uni(_raw, ratio_th) if ratio_th is not None else [m.best for m in _raw]
                                        elif matcher == "MNN":
                                            _apply = apply_ratio_bi if bidirectional else apply_ratio_fwd
                                            raw_matches = _apply(_raw, ratio_th) if ratio_th is not None else [m.best for m in _raw]
                                        else:
                                            raw_matches = _raw
                                    else:
                                        raw_matches = []

                                    n_matches = len(raw_matches)

                                    if n_matches > 0:
                                        q_arr      = np.array([m.query_idx for m in raw_matches], dtype=np.int32)
                                        t_arr      = np.array([m.train_idx for m in raw_matches], dtype=np.int32)
                                        _src       = ref_pts_arr[q_arr]   # float64, (N, 2)
                                        _dst       = rel_pts_arr[t_arr]   # float64, (N, 2)
                                        geo_errors = np.linalg.norm(_project_batch(_src, H_ref_to_rel) - _dst, axis=1)
                                        dists_arr  = np.array([m.distance for m in raw_matches], dtype=np.float64)
                                    else:
                                        geo_errors = np.array([], dtype=np.float64)
                                        dists_arr  = geo_errors

                                    # ── Accumulate match pool for mAP (per transformation) ─
                                    pool_key = (seq_type, max_kp, matcher, ratio_th, bidirectional, vis_filter)
                                    if pool_key not in agg_pool:
                                        agg_pool[pool_key] = []
                                    if n_matches > 0:
                                        agg_pool[pool_key].extend(zip(dists_arr.tolist(), geo_errors.tolist()))

                                    # ── Per-pair metrics ───────────────────────────────────
                                    mma_kp_ref = (
                                        {th: float(np.sum(geo_errors < th)) / n_ref for th in DISTANCE_THRESHOLDS}
                                        if n_ref > 0 else {th: 0.0 for th in DISTANCE_THRESHOLDS}
                                    )
                                    mma = (
                                        {th: float(np.sum(geo_errors < th)) / n_matches for th in DISTANCE_THRESHOLDS}
                                        if n_matches > 0 else {th: 0.0 for th in DISTANCE_THRESHOLDS}
                                    )

                                    # ── Homography estimation ──────────────────────────────
                                    hom_accs: dict[float, dict[int, float]] = {}
                                    can_ransac = n_matches >= 4
                                    if can_ransac:
                                        src = _src.astype(np.float32)
                                        dst = _dst.astype(np.float32)
                                        for ransac_th in RANSAC_THRESHOLDS:
                                            H_est, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_th, confidence = 0.995)
                                            if H_est is not None:
                                                corners_est = _project_batch(corners, H_est)
                                                diff_c      = corners_gt - corners_est
                                                mean_err    = float(np.sqrt((diff_c * diff_c).sum(axis=1)).mean())
                                                hom_accs[ransac_th] = {th: 1.0 if mean_err < th else 0.0
                                                                       for th in DISTANCE_THRESHOLDS}
                                            else:
                                                hom_accs[ransac_th] = {th: 0.0 for th in DISTANCE_THRESHOLDS}
                                    else:
                                        for ransac_th in RANSAC_THRESHOLDS:
                                            hom_accs[ransac_th] = {th: 0.0 for th in DISTANCE_THRESHOLDS}

                                    ratio_th_csv = ratio_th if ratio_th is not None else "-"

                                    # ── Accumulate into per-transformation aggregates ───────
                                    for ransac_th in RANSAC_THRESHOLDS:
                                        for dist_th in DISTANCE_THRESHOLDS:
                                            agg_key = (seq_type, matcher, ratio_th_csv, bidirectional, ransac_th, max_kp, vis_filter, dist_th)
                                            if agg_key not in agg_sums:
                                                agg_sums[agg_key]  = {
                                                    "mMA_kp_ref": 0.0, "mMA": 0.0,
                                                    "repeatability": 0.0, "homography_accuracy": 0.0,
                                                    "avg_num_matches": 0.0, "avg_num_keypoints": 0.0,
                                                    "num_keypoints_ref_detected": 0.0,
                                                    "num_keypoints_rel_detected": 0.0,
                                                    "avg_num_keypoints_detected": 0.0,
                                                }
                                                agg_count[agg_key] = 0
                                            s = agg_sums[agg_key]
                                            s["mMA_kp_ref"]                 += mma_kp_ref[dist_th]
                                            s["mMA"]                        += mma[dist_th]
                                            s["repeatability"]              += rep[dist_th]
                                            s["homography_accuracy"]        += hom_accs[ransac_th][dist_th]
                                            s["avg_num_matches"]            += n_matches
                                            s["avg_num_keypoints"]          += (n_ref + n_rel) / 2
                                            s["num_keypoints_ref_detected"] += n_kps_ref_pool
                                            s["num_keypoints_rel_detected"] += n_kps_rel_pool
                                            s["avg_num_keypoints_detected"] += (n_kps_ref_pool + n_kps_rel_pool) / 2
                                            agg_count[agg_key] += 1

                    except Exception:
                        if SKIP_AT_ERROR:
                            with open("failed_combinations.txt", "a") as f:
                                f.write(f"{pair_label}\n")
                                f.write(traceback.format_exc() + "\n\n")
                        else:
                            raise

            # ── Compute mAP and write aggregated rows ─────────────────────────────────
            mAP_cache: dict[tuple, float] = {}
            for (transformation, max_kp, matcher, ratio_th, bidirectional, vis_filter), pool in agg_pool.items():
                for dist_th in DISTANCE_THRESHOLDS:
                    mAP_cache[(transformation, max_kp, matcher, ratio_th, bidirectional, vis_filter, dist_th)] = compute_ap(pool, dist_th)

            rows = []
            for agg_key, s in agg_sums.items():
                transformation, matcher, ratio_th_csv, bidirectional, ransac_th, max_kp, vis_filter, dist_th = agg_key
                count = agg_count[agg_key]
                _rt   = None if ratio_th_csv == "-" else ratio_th_csv
                rows.append({
                    # Identity
                    "method":                 combo_key,
                    "tag":                    RUN_TAG,
                    # Matching parameters
                    "matcher":                matcher,
                    "ratio_threshold":        ratio_th_csv,
                    "mnn_bidirectional":      bidirectional if bidirectional is not None else "-",
                    "ransac_threshold":       ransac_th,
                    # Pipeline parameters
                    "max_keypoints":          max_kp,
                    "downsample_level":       ds_level,
                    "initial_sigma":          init_sigma,
                    "intrinsic_sigma":        intr_sigma,
                    "apply_progressive_blur": prog_blur,
                    "visibility_filter":      vis_filter,
                    # Transformation type
                    "transformation":         transformation,
                    # Threshold
                    "distance_threshold":     dist_th,
                    # Metrics
                    "mMA_kp_ref":             s["mMA_kp_ref"]          / count,
                    "mMA":                    s["mMA"]                  / count,
                    "repeatability":          s["repeatability"]        / count,
                    "homography_accuracy":    s["homography_accuracy"]  / count,
                    "mAP":                    mAP_cache.get((transformation, max_kp, matcher, _rt, bidirectional, vis_filter, dist_th), float("nan")),
                    # Counts
                    "avg_num_matches":            s["avg_num_matches"]            / count,
                    "avg_num_keypoints":          s["avg_num_keypoints"]          / count,
                    "num_keypoints_ref_detected": s["num_keypoints_ref_detected"] / count,
                    "num_keypoints_rel_detected": s["num_keypoints_rel_detected"] / count,
                    "avg_num_keypoints_detected": s["avg_num_keypoints_detected"] / count,
                })

            if rows:
                df_out = pd.DataFrame(rows)
                write_header = not os.path.isfile(RESULTS_FILE)
                df_out.to_csv(RESULTS_FILE, index=False, header=write_header, mode="a")

print("\nDone. Results written to", RESULTS_FILE)
