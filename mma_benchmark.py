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
RESULTS_FILE  = "results/mma_results_num_keypoints_scale_dependency.csv"

# ── Run tag ───────────────────────────────────────────────────────────────────
# Label for this entire benchmark run. All combinations share this tag.
# Use a different tag for each run you want to compare in display_mma.py.
RUN_TAG = "standard"

# ── Feature combinations ──────────────────────────────────────────────────────
features2d = {
    # "SIFT":      cv2.SIFT_create(),
    # "ORB":       cv2.ORB_create(nfeatures=2000),
    # "BRISK":     cv2.BRISK_create(),
    # "AKAZE":     cv2.AKAZE_create(),
    # "GFTT":      cv2.GFTTDetector_create(maxCorners=2000),
    "SIFT":  cv2.SIFT_create(contrastThreshold=0.004, edgeThreshold=100),
    "ORB":   cv2.ORB_create(fastThreshold=2, edgeThreshold=3, nfeatures=2000),
    "BRISK": cv2.BRISK_create(thresh=1),
    "AKAZE": cv2.AKAZE_create(threshold=0.0001),
    "GFTT":  cv2.GFTTDetector_create(qualityLevel=0.001, maxCorners=2000),
}

ONLY_SELF             = True
ONLY_SELF_EXCEPTIONS  = [("GFTT", "SIFT")]
ONLY_USED_AS_DETECTOR = ["GFTT"]

# ── Evaluation thresholds ─────────────────────────────────────────────────────
# Pixel-error thresholds used for MMA, repeatability, and homography accuracy.
THRESHOLDS = list(range(1, 11))

# ── Matching parameters ───────────────────────────────────────────────────────
# Each parameter is a list; all combinations are benchmarked and stored in the CSV.
MAX_FEATURES    = [100, 250, 500, 750, 1000, 1250, 1500]
USE_MNN         = [True]    # mutual nearest-neighbour filter on/off
RATIO_THRESHOLD = [0.8]     # Lowe's ratio test threshold
RANSAC_REPROJ   = [3.0]     # RANSAC reprojection error threshold (px)

# ── Downsampling parameters ───────────────────────────────────────────────────
DOWNSAMPLE_LEVELS             = [2]
DOWNSAMPLE_FACTOR             = [2]
DOWNSAMPLE_INTERPOLATION_TYPE = [None]
INITIAL_SIGMA                 = [1]
INTRINSIC_SIGMA               = [0.5]
APPLY_PROGRESSIVE_BLUR        = [False]

SKIP_AT_ERROR = True


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
        row: dict = {"scope": scope, "difficulty": difficulty, "threshold": threshold}
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

_max_k = max(MAX_FEATURES)

_ds_configs = list(itertools.product(
    DOWNSAMPLE_LEVELS,
    DOWNSAMPLE_FACTOR,
    INITIAL_SIGMA,
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
        for max_features in MAX_FEATURES:
            for use_mnn in USE_MNN:
                for ratio_th in RATIO_THRESHOLD:
                    for ransac_reproj in RANSAC_REPROJ:
                        sk = (max_features, use_mnn, ratio_th, ransac_reproj)
                        raw_by_sweep[sk] = []
                        match_cnt_by[sk] = []
        feature_cnts_by_mf = {mf: [] for mf in MAX_FEATURES}

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

                    zero_mma = {th: 0.0 for th in THRESHOLDS}
                    zero_rep = {th: 0.0 for th in THRESHOLDS}
                    zero_hom = {th: 0.0 for th in THRESHOLDS}

                    def _append_all_zeros(rep=zero_rep):
                        for max_features in MAX_FEATURES:
                            for use_mnn in USE_MNN:
                                for ratio_th in RATIO_THRESHOLD:
                                    for ransac_reproj in RANSAC_REPROJ:
                                        sk = (max_features, use_mnn, ratio_th, ransac_reproj)
                                        raw_by_sweep[sk].append(
                                            (seq_type, img_idx, zero_mma, zero_mma, rep, zero_hom)
                                        )
                                        match_cnt_by[sk].append(0)

                    # ── Detect keypoints in downsampled coords ────────────────
                    kps_ref_all = extractor.detect_keypoints(img_ref_ds)
                    kps_rel_all = extractor.detect_keypoints(img_rel_ds)

                    # Scale to original coords for geometric filtering
                    for kp in kps_ref_all:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                    for kp in kps_rel_all:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

                    # Keep only keypoints visible in the other image
                    kps_ref_all = [kp for kp in kps_ref_all
                                   if _in_bounds(_project(kp.pt, H_ref_to_rel), w_rel, h_rel)]
                    kps_rel_all = [kp for kp in kps_rel_all
                                   if _in_bounds(_project(kp.pt, H_rel_to_ref), w_ref, h_ref)]
                    kps_ref_all = sorted(kps_ref_all, key=lambda k: -k.response)[:_max_k]
                    kps_rel_all = sorted(kps_rel_all, key=lambda k: -k.response)[:_max_k]

                    if not kps_ref_all or not kps_rel_all:
                        _append_all_zeros()
                        continue

                    # Scale back to downsampled coords for description
                    for kp in kps_ref_all:
                        kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                    for kp in kps_rel_all:
                        kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

                    kps_ref_all, descs_ref_all = extractor.describe_keypoints(img_ref_ds, kps_ref_all)
                    kps_rel_all, descs_rel_all = extractor.describe_keypoints(img_rel_ds, kps_rel_all)

                    # Scale back to original coords for all downstream math
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

                    for max_features in MAX_FEATURES:
                        kps_ref  = kps_ref_all[:max_features]
                        kps_rel  = kps_rel_all[:max_features]
                        n_ref, n_rel = len(kps_ref), len(kps_rel)
                        desc_ref = desc_ref_full[:n_ref]
                        desc_rel = desc_rel_full[:n_rel]
                        feature_cnts_by_mf[max_features].extend([n_ref, n_rel])

                        if not kps_ref or not kps_rel:
                            for use_mnn in USE_MNN:
                                for ratio_th in RATIO_THRESHOLD:
                                    for ransac_reproj in RANSAC_REPROJ:
                                        sk = (max_features, use_mnn, ratio_th, ransac_reproj)
                                        raw_by_sweep[sk].append(
                                            (seq_type, img_idx, zero_mma, zero_rep, zero_hom)
                                        )
                                        match_cnt_by[sk].append(0)
                            continue

                        # ── Repeatability (descriptor-free, once per max_features) ──
                        ref_pts      = np.array([kp.pt for kp in kps_ref])
                        rel_pts      = np.array([kp.pt for kp in kps_rel])
                        ref_proj_rel = np.array([_project(kp.pt, H_ref_to_rel) for kp in kps_ref])
                        rel_proj_ref = np.array([_project(kp.pt, H_rel_to_ref) for kp in kps_rel])
                        dists_ref    = np.min(np.linalg.norm(ref_proj_rel[:, None] - rel_pts[None], axis=2), axis=1)
                        dists_rel    = np.min(np.linalg.norm(rel_proj_ref[:, None] - ref_pts[None], axis=2), axis=1)
                        rep = {th: float(np.sum(dists_ref < th) + np.sum(dists_rel < th)) / (n_ref + n_rel)
                               for th in THRESHOLDS}

                        # ── KNN matching (k=2 enables ratio test) ─────────────
                        k_nn    = min(2, n_rel)
                        knn_raw = bf.knnMatch(desc_ref, desc_rel, k=k_nn)

                        for use_mnn in USE_MNN:
                            # ── MNN filter ────────────────────────────────────
                            if use_mnn:
                                nn21    = {m.queryIdx: m.trainIdx for m in bf.match(desc_rel, desc_ref)}
                                knn_mnn = [pair for pair in knn_raw
                                           if nn21.get(pair[0].trainIdx) == pair[0].queryIdx]
                            else:
                                knn_mnn = list(knn_raw)

                            if not knn_mnn:
                                for ratio_th in RATIO_THRESHOLD:
                                    for ransac_reproj in RANSAC_REPROJ:
                                        sk = (max_features, use_mnn, ratio_th, ransac_reproj)
                                        raw_by_sweep[sk].append(
                                            (seq_type, img_idx, zero_mma, zero_mma, rep, zero_hom)
                                        )
                                        match_cnt_by[sk].append(0)
                                continue

                            # ── MMA ───────────────────────────────────────────
                            errors = np.array([
                                np.linalg.norm(
                                    np.array(_project(kps_ref[pair[0].queryIdx].pt, H_ref_to_rel))
                                    - np.array(kps_rel[pair[0].trainIdx].pt)
                                )
                                for pair in knn_mnn
                            ])
                            n_putative = len(errors)
                            mma_kps     = {th: float(np.sum(errors < th)) / n_ref      for th in THRESHOLDS}
                            mma_matches = {th: float(np.sum(errors < th)) / n_putative  for th in THRESHOLDS}

                            corners    = np.array([[0, 0], [w_ref-1, 0],
                                                   [w_ref-1, h_ref-1], [0, h_ref-1]], dtype=np.float64)
                            corners_gt = np.array([_project(pt, H_ref_to_rel) for pt in corners])

                            for ratio_th in RATIO_THRESHOLD:
                                # ── Lowe's ratio test ──────────────────────────
                                if ratio_th is not None and k_nn >= 2:
                                    matches = [pair[0] for pair in knn_mnn
                                               if len(pair) >= 2 and pair[0].distance < ratio_th * pair[1].distance]
                                    matches += [pair[0] for pair in knn_mnn if len(pair) == 1]
                                else:
                                    matches = [pair[0] for pair in knn_mnn]

                                can_ransac = len(matches) >= 4
                                if can_ransac:
                                    src = np.float32([kps_ref[m.queryIdx].pt for m in matches])
                                    dst = np.float32([kps_rel[m.trainIdx].pt for m in matches])

                                for ransac_reproj in RANSAC_REPROJ:
                                    # ── Homography estimation ──────────────────
                                    if can_ransac:
                                        H_est, _ = cv2.findHomography(dst, src, cv2.RANSAC, ransac_reproj)
                                        if H_est is not None:
                                            corners_est = np.array([_project(pt, np.linalg.inv(H_est))
                                                                    for pt in corners])
                                            mean_err    = float(np.mean(np.linalg.norm(
                                                corners_gt - corners_est, axis=1)))
                                            hom_acc = {th: 1.0 if mean_err < th else 0.0 for th in THRESHOLDS}
                                        else:
                                            hom_acc = zero_hom
                                    else:
                                        hom_acc = zero_hom

                                    sk = (max_features, use_mnn, ratio_th, ransac_reproj)
                                    raw_by_sweep[sk].append((seq_type, img_idx, mma_kps, mma_matches, rep, hom_acc))
                                    match_cnt_by[sk].append(len(matches))

                except Exception:
                    if SKIP_AT_ERROR:
                        with open("failed_combinations.txt", "a") as f:
                            f.write(f"{pair_label}\n")
                            f.write(traceback.format_exc() + "\n\n")
                    else:
                        raise

        # ── Write results to CSV ──────────────────────────────────────────────
        for max_features in MAX_FEATURES:
            for use_mnn in USE_MNN:
                for ratio_th in RATIO_THRESHOLD:
                    for ransac_reproj in RANSAC_REPROJ:
                        sk           = (max_features, use_mnn, ratio_th, ransac_reproj)
                        raw_pairs    = raw_by_sweep[sk]
                        match_counts = match_cnt_by[sk]
                        feat_counts  = feature_cnts_by_mf[max_features]

                        common = {
                            # ── Identity ───────────────────────────────────────────────
                            "combination":                   combo_key,
                            "tag":                           RUN_TAG,
                            # ── Downsampling ────────────────────────────────────────────
                            "downsample_level":              ds_level,
                            "downsample_factor":             ds_factor,
                            "initial_sigma":                 init_sigma,
                            "intrinsic_sigma":               intr_sigma,
                            "apply_progressive_blur":        prog_blur,
                            "downsample_interpolation_type": str(interp_type),
                            # ── Pipeline parameters ─────────────────────────────────────
                            "max_features":                  max_features,
                            "use_mnn":                       use_mnn,
                            "ratio_threshold":               ratio_th if ratio_th is not None else float("nan"),
                            "ransac_reproj":                 ransac_reproj,
                            # ── Run statistics ──────────────────────────────────────────
                            "avg_num_features":        float(np.mean(feat_counts))  if feat_counts  else 0.0,
                            "frac_below_max_features": float(np.mean([c < max_features for c in feat_counts])) if feat_counts else 0.0,
                            "avg_num_matches":         float(np.mean(match_counts)) if match_counts else 0.0,
                        }

                        metric_rows = aggregate(raw_pairs)
                        rows = [{**common, **row} for row in metric_rows]

                        df_out = pd.DataFrame(rows)
                        write_header = not os.path.isfile(RESULTS_FILE)
                        df_out.to_csv(RESULTS_FILE, index=False, header=write_header, mode="a")

print("\nDone. Results written to", RESULTS_FILE)
