import os
os.environ["BEARTYPE_IS_BEING_TYPE_CHECKED"] = "0"

from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import downsample
from tqdm import tqdm
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
HPATCHES_PATH = r"hpatches-sequences-release"
RESULTS_FILE  = "results/mma_results_kps_scale.csv"

PIXEL_THRESHOLDS  = list(range(1, 11))  # 1 … 10 px

# Matching pipeline toggles
USE_MNN             = True
ESTIMATE_HOMOGRAPHY = True   # run RANSAC after MMA to estimate homography
HOM_THRESHOLDS      = list(range(1, 6))   # corner-error thresholds for homography accuracy (px)

# Sweepable parameters — use a list or range to benchmark multiple values
MAX_FEATURES_SWEEP    = [100, 250, 500, 750, 1000, 1250, 1500]
RATIO_THRESHOLD_SWEEP = [0.8]
RANSAC_REPROJ_SWEEP   = [3.0]

# Scale
DOWNSAMPLE_LEVELS = [0, 1, 2]
INITIAL_SIGMA = 1
APPLY_PROGRESSIVE_BLUR = True #If off, only initial sigma is applied
INTRINSIC_SIGMA = 0.5
DOWNSAMPLE_FACTOR = 2
DOWNSAMPLE_INTERPOLATION_TYPE = None

SKIP_AT_ERROR = True

# HPatches has 6 images per sequence: image 1 (reference) + images 2–6 (related).
# Related-image index (0-based) → difficulty bucket  (buckets may overlap)
# Easy:   pairs (1,2) and (1,3)  → indices 0, 1
# Medium: pairs (1,4) and (1,5)  → indices 2, 3
# Hard:   pairs (1,5) and (1,6)  → indices 3, 4
EASY_INDICES   = [0, 1]
NORMAL_INDICES = [2, 3]
HARD_INDICES   = [3, 4]

# ============================================================
# FEATURE EXTRACTORS  (same pattern as main.py)
# ============================================================
features2d = {
    "SIFT": cv2.SIFT_create(),
    "ORB":  cv2.ORB_create(),
    "BRISK": cv2.BRISK_create(),
    "AKAZE": cv2.AKAZE_create(),
    "GFTT":  cv2.GFTTDetector_create(),
    # "SHIFT" : ShiTomasiSift(),
    "SIFT_low": cv2.SIFT_create(contrastThreshold = 0.001, edgeThreshold = 100),
    "ORB_low":  cv2.ORB_create(edgeThreshold = 1, nfeatures = 2000),
    "BRISK_low": cv2.BRISK_create(thresh = 1),
    "AKAZE_low": cv2.AKAZE_create(threshold = 0.00001),
    "GFTT_low":  cv2.GFTTDetector_create(qualityLevel = 0.0001, maxCorners = 2000),
    # "SHIFT_low" : ShiTomasiSift(quality_level = 0.000000001)
}

ONLY_SELF            = True
ONLY_SELF_EXCEPTIONS = [("GFTT", "SIFT"), ("GFTT_low", "SIFT_low")]
ONLY_USED_AS_DETECTOR = ["GFTT", "GFTT_low"]

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
# DATASET LOADING (with sequence type detection)
# ============================================================

def load_hpatches(path: str):
    """
    Returns list of (name, seq_type, images, homographies) sorted by name.
    seq_type: 'illumination' or 'viewpoint'.
    homographies: list of 5 matrices, each mapping related image → reference image 1.
    """
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
            continue  # skip non-HPatches folders

        imgs, homos = [], []
        for filename in sorted(os.listdir(subfolder)):
            fp = os.path.join(subfolder, filename)
            if filename.lower().endswith(".ppm"):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs.append(img)
            elif filename.startswith("H_"):
                H = np.loadtxt(fp)
                homos.append(np.linalg.inv(H))  # store as related → reference

        if imgs and homos:
            sequences.append((name, seq_type, imgs, homos))
    return sequences


sequences = load_hpatches(HPATCHES_PATH)
print(f"Loaded {len(sequences)} sequences "
      f"({sum(1 for s in sequences if s[1]=='illumination')} illumination, "
      f"{sum(1 for s in sequences if s[1]=='viewpoint')} viewpoint)")

# ============================================================
# MATCHING HELPERS
# ============================================================

def _project(pt, H):
    x, y = pt
    v = H @ np.array([x, y, 1.0])
    if abs(v[2]) < 1e-10:
        return float("inf"), float("inf")
    return v[0] / v[2], v[1] / v[2]


def _in_bounds(pt, w, h, margin=0):
    x, y = pt
    return margin <= x < (w - margin) and margin <= y < (h - margin)


IMG_INDICES = [2, 3, 4, 5, 6]   # actual image numbers (reference is image 1)

def _diffs_of(rel_idx):
    """Return all difficulty buckets an image-pair index belongs to (may overlap)."""
    diffs = []
    if rel_idx in EASY_INDICES:
        diffs.append("easy")
    if rel_idx in NORMAL_INDICES:
        diffs.append("normal")
    if rel_idx in HARD_INDICES:
        diffs.append("hard")
    return diffs


# ============================================================
# AGGREGATION HELPER
# ============================================================

def aggregate(raw):
    """
    raw: list of (seq_type, img_idx, mma_dict, rep_dict, hom_acc_dict)
          seq_type ∈ {'illumination', 'viewpoint'}
          img_idx  ∈ {2, 3, 4, 5, 6}

    Returns a list of row-dicts in long format. Each row represents one
    (scope, difficulty, metric, threshold) bucket and contains stats
    (mean, std, min, max, count) over all image pairs that fell into it.
    """

    # ── Pass 1: Collect raw values per (scope, difficulty, metric, threshold) ──
    # Each image pair contributes to its seq_type scope AND "overall",
    # and to each difficulty bucket it belongs to (easy/normal/hard) AND "all",
    # and to its specific per-image-index bucket (img2 … img6).
    buckets: dict[tuple, list[float]] = {}

    for seq_type, img_idx, mma_dict, rep_dict, hom_acc_dict in raw:
        scope_tag = "illumination" if seq_type == "illumination" else "viewpoint"
        diffs = _diffs_of(img_idx - 2)   # which named difficulty buckets this pair belongs to

        for scope in [scope_tag, "overall"]:
            # Named difficulty buckets (easy / normal / hard / all)
            for diff in diffs + ["all"]:
                for th, val in mma_dict.items():
                    buckets.setdefault((scope, diff, "mma", th), []).append(val)
                for th, val in rep_dict.items():
                    buckets.setdefault((scope, diff, "rep", th), []).append(val)
                for eps, val in hom_acc_dict.items():
                    buckets.setdefault((scope, diff, "hom_acc", eps), []).append(val)

            # Per-image-index bucket (img2, img3, img4, img5, img6)
            img_key = f"img{img_idx}"
            for th, val in mma_dict.items():
                buckets.setdefault((scope, img_key, "mma", th), []).append(val)
            for th, val in rep_dict.items():
                buckets.setdefault((scope, img_key, "rep", th), []).append(val)
            for eps, val in hom_acc_dict.items():
                buckets.setdefault((scope, img_key, "hom_acc", eps), []).append(val)

    # ── Pass 2: Compute stats for each bucket and emit one row per bucket ──────
    rows = []
    for (scope, difficulty, metric, threshold), vals in buckets.items():
        arr = np.array(vals, dtype=np.float64)
        rows.append({
            "scope":      scope,
            "difficulty": difficulty,
            "metric":     metric,
            "threshold":  threshold,
            "mean":       float(np.mean(arr)),
            "std":        float(np.std(arr)),
            "min":        float(np.min(arr)),
            "max":        float(np.max(arr)),
            "count":      len(arr),
        })
    return rows


# ============================================================
# MAIN LOOP  (optimized: detect+describe once per pair, sweep over cached data)
# ============================================================

warnings.filterwarnings("once", category=UserWarning)
os.makedirs("results", exist_ok=True)

# Detect+describe once at the largest max_features; inner loops slice the cached result.
_max_k = max(MAX_FEATURES_SWEEP)

for combo_key, extractor in tqdm(test_combinations.items(), desc="Combinations", position=0):

    for DOWNSAMPLE_LEVEL in tqdm(DOWNSAMPLE_LEVELS, desc="Downsample level", leave=False, position=1):

        # One accumulator list per sweep combination
        raw_by_sweep       = {}
        match_cnt_by       = {}
        for max_features in MAX_FEATURES_SWEEP:
            for ratio_th in RATIO_THRESHOLD_SWEEP:
                for ransac_reproj in RANSAC_REPROJ_SWEEP:
                    raw_by_sweep[(max_features, ratio_th, ransac_reproj)]  = []
                    match_cnt_by[(max_features, ratio_th, ransac_reproj)]  = []
        feature_cnts_by_mf = {mf: [] for mf in MAX_FEATURES_SWEEP}  # per-image counts (ref + rel)

        scale = DOWNSAMPLE_FACTOR ** DOWNSAMPLE_LEVEL

        for name, seq_type, imgs, homos in tqdm(sequences, leave=False, desc="Sequences", position=2):
            # Downsample ref image once per sequence; capture original dims for homography math
            h_ref, w_ref = imgs[0].shape[:2]
            img_ref_ds   = downsample(imgs[0], DOWNSAMPLE_LEVEL, DOWNSAMPLE_FACTOR,
                                      INTRINSIC_SIGMA, INITIAL_SIGMA,
                                      APPLY_PROGRESSIVE_BLUR, DOWNSAMPLE_INTERPOLATION_TYPE)

            for rel_idx, (img_rel_orig, H_rel_to_ref) in enumerate(tqdm(list(zip(imgs[1:], homos)), leave=False, desc="Image pairs", position=3)):
                img_idx    = rel_idx + 2
                pair_label = f"{combo_key} ds={DOWNSAMPLE_LEVEL} seq={name} img={img_idx}"
                try:
                    H_ref_to_rel = np.linalg.inv(H_rel_to_ref)
                    h_rel, w_rel = img_rel_orig.shape[:2]   # original dims for overlap check
                    img_rel_ds   = downsample(img_rel_orig, DOWNSAMPLE_LEVEL, DOWNSAMPLE_FACTOR,
                                              INTRINSIC_SIGMA, INITIAL_SIGMA,
                                              APPLY_PROGRESSIVE_BLUR, DOWNSAMPLE_INTERPOLATION_TYPE)

                    zero_mma     = {th:  0.0 for th  in PIXEL_THRESHOLDS}
                    zero_rep     = {th:  0.0 for th  in PIXEL_THRESHOLDS}
                    zero_hom_acc = {eps: 0.0 for eps in HOM_THRESHOLDS}

                    # ── Detect in downsampled image coords ─────────────────────
                    kps_ref_all = extractor.detect_keypoints(img_ref_ds)
                    kps_rel_all = extractor.detect_keypoints(img_rel_ds)

                    # Scale kp positions to original image coords (homographies live there)
                    for kp in kps_ref_all:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                    for kp in kps_rel_all:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

                    # Filter to overlap and sort in original coords
                    kps_ref_all = [kp for kp in kps_ref_all
                                   if _in_bounds(_project(kp.pt, H_ref_to_rel), w_rel, h_rel)]
                    kps_rel_all = [kp for kp in kps_rel_all
                                   if _in_bounds(_project(kp.pt, H_rel_to_ref), w_ref, h_ref)]
                    kps_ref_all = sorted(kps_ref_all, key=lambda k: -k.response)[:_max_k]
                    kps_rel_all = sorted(kps_rel_all, key=lambda k: -k.response)[:_max_k]

                    if not kps_ref_all or not kps_rel_all:
                        for max_features in MAX_FEATURES_SWEEP:
                            for ratio_th in RATIO_THRESHOLD_SWEEP:
                                for ransac_reproj in RANSAC_REPROJ_SWEEP:
                                    sk = (max_features, ratio_th, ransac_reproj)
                                    raw_by_sweep[sk].append((seq_type, img_idx, zero_mma, zero_rep, zero_hom_acc))
                                    match_cnt_by[sk].append(0)
                        continue

                    # Scale back to downsampled coords for description
                    for kp in kps_ref_all:
                        kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                    for kp in kps_rel_all:
                        kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)

                    # ── Describe in downsampled image coords ───────────────────
                    kps_ref_all, descs_ref_all = extractor.describe_keypoints(img_ref_ds, kps_ref_all)
                    kps_rel_all, descs_rel_all = extractor.describe_keypoints(img_rel_ds, kps_rel_all)

                    # Scale back to original coords for all downstream math
                    for kp in kps_ref_all:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
                    for kp in kps_rel_all:
                        kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

                    if not descs_ref_all or not descs_rel_all:
                        for max_features in MAX_FEATURES_SWEEP:
                            for ratio_th in RATIO_THRESHOLD_SWEEP:
                                for ransac_reproj in RANSAC_REPROJ_SWEEP:
                                    sk = (max_features, ratio_th, ransac_reproj)
                                    raw_by_sweep[sk].append((seq_type, img_idx, zero_mma, zero_rep, zero_hom_acc))
                                    match_cnt_by[sk].append(0)
                        continue

                    dtype         = np.float32 if extractor.distance_type == cv2.NORM_L2 else np.uint8
                    desc_ref_full = np.array(descs_ref_all, dtype=dtype)
                    desc_rel_full = np.array(descs_rel_all, dtype=dtype)
                    bf            = cv2.BFMatcher(extractor.distance_type, crossCheck=False)

                    for max_features in MAX_FEATURES_SWEEP:
                        # ── Slice to max_features (no re-detect / re-describe) ─
                        kps_ref  = kps_ref_all[:max_features]
                        kps_rel  = kps_rel_all[:max_features]
                        n_ref, n_rel = len(kps_ref), len(kps_rel)
                        desc_ref = desc_ref_full[:n_ref]
                        desc_rel = desc_rel_full[:n_rel]
                        feature_cnts_by_mf[max_features].extend([n_ref, n_rel])

                        if not kps_ref or not kps_rel:
                            for ratio_th in RATIO_THRESHOLD_SWEEP:
                                for ransac_reproj in RANSAC_REPROJ_SWEEP:
                                    sk = (max_features, ratio_th, ransac_reproj)
                                    raw_by_sweep[sk].append((seq_type, img_idx, zero_mma, zero_rep, zero_hom_acc))
                                    match_cnt_by[sk].append(0)
                            continue

                        # ── Repeatability (descriptor-free, once per max_features) ─
                        ref_pts      = np.array([kp.pt for kp in kps_ref])
                        rel_pts      = np.array([kp.pt for kp in kps_rel])
                        ref_proj_rel = np.array([_project(kp.pt, H_ref_to_rel) for kp in kps_ref])
                        rel_proj_ref = np.array([_project(kp.pt, H_rel_to_ref) for kp in kps_rel])
                        dists_ref    = np.min(np.linalg.norm(ref_proj_rel[:, None] - rel_pts[None], axis=2), axis=1)
                        dists_rel    = np.min(np.linalg.norm(rel_proj_ref[:, None] - ref_pts[None], axis=2), axis=1)
                        rep = {th: float(np.sum(dists_ref < th) + np.sum(dists_rel < th)) / (n_ref + n_rel)
                            for th in PIXEL_THRESHOLDS}

                        # ── KNN match + MNN filter (once per max_features) ───────────
                        k_nn    = min(2, n_rel)
                        knn_raw = bf.knnMatch(desc_ref, desc_rel, k=k_nn)

                        if USE_MNN:
                            nn21    = {m.queryIdx: m.trainIdx for m in bf.match(desc_rel, desc_ref)}
                            knn_mnn = [pair for pair in knn_raw
                                       if nn21.get(pair[0].trainIdx) == pair[0].queryIdx]
                        else:
                            knn_mnn = list(knn_raw)

                        if not knn_mnn:
                            for ratio_th in RATIO_THRESHOLD_SWEEP:
                                for ransac_reproj in RANSAC_REPROJ_SWEEP:
                                    sk = (max_features, ratio_th, ransac_reproj)
                                    raw_by_sweep[sk].append((seq_type, img_idx, zero_mma, rep, zero_hom_acc))
                                    match_cnt_by[sk].append(0)
                            continue

                        # ── MMA before ratio test (once per max_features) ─────────────
                        errors = np.array([
                            np.linalg.norm(
                                np.array(_project(kps_ref[pair[0].queryIdx].pt, H_ref_to_rel))
                                - np.array(kps_rel[pair[0].trainIdx].pt)
                            )
                            for pair in knn_mnn
                        ])
                        mma = {th: float(np.sum(errors < th)) / n_ref for th in PIXEL_THRESHOLDS}

                        # Pre-compute corners (shared across ratio_th and ransac_reproj sweeps)
                        corners    = np.array([[0, 0], [w_ref-1, 0],
                                               [w_ref-1, h_ref-1], [0, h_ref-1]], dtype=np.float64)
                        corners_gt = np.array([_project(pt, H_ref_to_rel) for pt in corners])

                        for ratio_th in RATIO_THRESHOLD_SWEEP:
                            # ── Apply ratio test to get matches for RANSAC ────────────
                            if ratio_th is not None and k_nn >= 2:
                                matches = [pair[0] for pair in knn_mnn
                                           if len(pair) >= 2 and pair[0].distance < ratio_th * pair[1].distance]
                                matches += [pair[0] for pair in knn_mnn if len(pair) == 1]
                            else:
                                matches = [pair[0] for pair in knn_mnn]

                            can_ransac = ESTIMATE_HOMOGRAPHY and len(matches) >= 4
                            if can_ransac:
                                src = np.float32([kps_ref[m.queryIdx].pt for m in matches])
                                dst = np.float32([kps_rel[m.trainIdx].pt for m in matches])

                            for ransac_reproj in RANSAC_REPROJ_SWEEP:
                                # ── RANSAC ────────────────────────────────────────────
                                hom_acc = zero_hom_acc
                                if can_ransac:
                                    H_est, _ = cv2.findHomography(dst, src, cv2.RANSAC, ransac_reproj)
                                    if H_est is not None:
                                        corners_est = np.array([_project(pt, np.linalg.inv(H_est)) for pt in corners])
                                        mean_err    = float(np.mean(np.linalg.norm(corners_gt - corners_est, axis=1)))
                                        hom_acc     = {eps: 1.0 if mean_err < eps else 0.0 for eps in HOM_THRESHOLDS}
                                sk = (max_features, ratio_th, ransac_reproj)
                                raw_by_sweep[sk].append((seq_type, img_idx, mma, rep, hom_acc))
                                match_cnt_by[sk].append(len(matches))

                except Exception:
                    if SKIP_AT_ERROR:
                        with open("failed_combinations.txt", "a") as f:
                            f.write(f"{pair_label}\n")
                            f.write(traceback.format_exc() + "\n\n")
                    else:
                        raise

        # ── Write CSV: one row per (sweep combination × scope × difficulty × metric × threshold) ──
        for max_features in MAX_FEATURES_SWEEP:
            for ratio_th in RATIO_THRESHOLD_SWEEP:
                for ransac_reproj in RANSAC_REPROJ_SWEEP:
                    sk           = (max_features, ratio_th, ransac_reproj)
                    raw_pairs    = raw_by_sweep[sk]
                    match_counts = match_cnt_by[sk]
                    feat_counts  = feature_cnts_by_mf[max_features]

                    label = (f"{combo_key} | mf={max_features} "
                             f"rt={'off' if ratio_th is None else ratio_th} "
                             f"rr={ransac_reproj}")
                    print(f"\n--- {label} ---")

                    # Columns shared by every row for this sweep combination
                    common = {
                        "combination":             combo_key,
                        "downsample_level":        DOWNSAMPLE_LEVEL,
                        "max_features":            max_features,
                        "ratio_threshold":         ratio_th if ratio_th is not None else float("nan"),
                        "ransac_reproj":           ransac_reproj,
                        "avg_num_features":        float(np.mean(feat_counts)) if feat_counts else 0.0,
                        "frac_below_max_features": float(np.mean([c < max_features for c in feat_counts])) if feat_counts else 0.0,
                        "avg_num_matches":         float(np.mean(match_counts)) if match_counts else 0.0,
                    }

                    # aggregate() returns one row per (scope, difficulty, metric, threshold)
                    metric_rows = aggregate(raw_pairs)
                    rows = [{**common, **row} for row in metric_rows]

                    print(f"  rows: {len(rows)}")
                    for k, v in common.items():
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

                    df_out = pd.DataFrame(rows)
                    write_header = not os.path.isfile(RESULTS_FILE)
                    df_out.to_csv(RESULTS_FILE, index=False, header=write_header, mode="a")

print("\nDone. Results written to", RESULTS_FILE)
