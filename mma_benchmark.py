import os
os.environ["BEARTYPE_IS_BEING_TYPE_CHECKED"] = "0"

from benchmark.feature_extractor import FeatureExtractor
from benchmark.utils import optional_try
from tqdm import tqdm
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
RESULTS_FILE  = "results/mma_results.csv"

MAX_FEATURES      = 1000          # top-k keypoints per image (in overlap region)
PIXEL_THRESHOLDS  = list(range(1, 11))  # 1 … 10 px

# Matching pipeline toggles
USE_MNN            = True
RATIO_THRESHOLD    = 0.9          # Lowe's ratio; set None to disable
ESTIMATE_HOMOGRAPHY = True        # run RANSAC after MMA to estimate homography
RANSAC_REPROJ      = 3.0
HOM_THRESHOLDS     = list(range(1, 6))   # corner-error thresholds for homography accuracy (px)

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
    "FAST":  cv2.FastFeatureDetector_create(),
    "GFTT":  cv2.GFTTDetector_create(),
    "SHIFT" : ShiTomasiSift()
}

ONLY_SELF             = True
ONLY_SELF_EXCEPTIONS  = [("GFTT", "SIFT"), ("GFTT2", "SIFT")]
ONLY_USED_AS_DETECTOR  = ["GFTT", "FAST2", "GFTT2"]
ONLY_USED_AS_DESCRIPTOR = ["FREAK", "SIFT_FAST2", "SIFT_GFTT2"]
BLACKLIST             = []
ALLOWED_DESCRIPTOR_FOR_DETECTOR = {
    "FAST2": "SIFT_FAST2",
    "GFTT2": "SIFT",
    "ORB":   "ORB",
    "SIFT":  "SIFT",
    "BRISK": "BRISK",
}
ALLOWED_DETECTOR_FOR_DESCRIPTOR = {
    "SIFT_FAST2": "FAST2",
}

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
        if (det_key, desc_key) in BLACKLIST:
            continue
        if det_key in ONLY_USED_AS_DESCRIPTOR:
            continue
        if desc_key in ONLY_USED_AS_DETECTOR:
            continue
        if det_key in ALLOWED_DESCRIPTOR_FOR_DETECTOR:
            if desc_key != ALLOWED_DESCRIPTOR_FOR_DETECTOR[det_key]:
                continue
        if desc_key in ALLOWED_DETECTOR_FOR_DESCRIPTOR:
            if det_key != ALLOWED_DETECTOR_FOR_DESCRIPTOR[desc_key]:
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
    return v[0] / v[2], v[1] / v[2]


def _in_bounds(pt, w, h, margin=0):
    x, y = pt
    return margin <= x < (w - margin) and margin <= y < (h - margin)


def compute_pair_metrics(
    img_ref, img_rel, H_rel_to_ref,
    extractor: FeatureExtractor,
    max_features, thresholds, hom_thresholds,
    use_mnn, ratio_th, estimate_homography, ransac_reproj,
):
    """
    Compute MMA, repeatability and (optionally) homography estimation accuracy for one image pair.

    MMA(th)      = |correct matches with reprojection error < th| / |keypoints in overlap|
    Rep(th)      = (# ref kps with a match in rel + # rel kps with a match in ref) / (N1 + N2)
                   where a "match" means the projected kp lands within th px of any kp in the other image
                   (SuperPoint eq. 13-14)
    hom_acc(eps) = 1 if mean corner reprojection error of estimated vs GT homography < eps

    Returns (mma_dict, rep_dict, hom_acc_dict, num_matches).
    hom_acc_dict is all zeros if estimate_homography is False or RANSAC fails.
    """
    H_ref_to_rel = np.linalg.inv(H_rel_to_ref)
    h_ref, w_ref = img_ref.shape[:2]
    h_rel, w_rel = img_rel.shape[:2]

    zero_mma     = {th: 0.0 for th in thresholds}
    zero_rep     = {th: 0.0 for th in thresholds}
    zero_hom_acc = {eps: 0.0 for eps in hom_thresholds}

    # --- Detect ---
    kps_ref = extractor.detect_keypoints(img_ref)
    kps_rel = extractor.detect_keypoints(img_rel)

    # --- Filter to common overlap region ---
    kps_ref = [kp for kp in kps_ref
               if _in_bounds(_project(kp.pt, H_ref_to_rel), w_rel, h_rel)]
    kps_rel = [kp for kp in kps_rel
               if _in_bounds(_project(kp.pt, H_rel_to_ref), w_ref, h_ref)]

    # --- Top-k by response ---
    kps_ref = sorted(kps_ref, key=lambda k: -k.response)[:max_features]
    kps_rel = sorted(kps_rel, key=lambda k: -k.response)[:max_features]

    if not kps_ref or not kps_rel:
        return zero_mma, zero_rep, zero_hom_acc, 0

    # --- Repeatability (SuperPoint eq. 13-14, descriptor-free) ---
    ref_pts      = np.array([kp.pt for kp in kps_ref])
    rel_pts      = np.array([kp.pt for kp in kps_rel])
    ref_proj_rel = np.array([_project(kp.pt, H_ref_to_rel) for kp in kps_ref])
    rel_proj_ref = np.array([_project(kp.pt, H_rel_to_ref) for kp in kps_rel])

    dists_ref = np.min(np.linalg.norm(ref_proj_rel[:, None] - rel_pts[None], axis=2), axis=1)
    dists_rel = np.min(np.linalg.norm(rel_proj_ref[:, None] - ref_pts[None], axis=2), axis=1)
    n1, n2 = len(kps_ref), len(kps_rel)
    rep = {th: float(np.sum(dists_ref < th) + np.sum(dists_rel < th)) / (n1 + n2)
           for th in thresholds}

    # --- Describe ---
    kps_ref, descs_ref = extractor.describe_keypoints(img_ref, kps_ref)
    kps_rel, descs_rel = extractor.describe_keypoints(img_rel, kps_rel)

    if not descs_ref or not descs_rel:
        return zero_mma, rep, zero_hom_acc, 0

    dtype = np.float32 if extractor.distance_type == cv2.NORM_L2 else np.uint8
    desc_ref = np.array(descs_ref, dtype=dtype)
    desc_rel = np.array(descs_rel, dtype=dtype)

    # --- Match ---
    bf = cv2.BFMatcher(extractor.distance_type, crossCheck=False)

    if ratio_th is not None:
        k = min(2, len(kps_rel))
        raw = bf.knnMatch(desc_ref, desc_rel, k=k)
        matches = []
        for pair in raw:
            if len(pair) == 2 and pair[0].distance < ratio_th * pair[1].distance:
                matches.append(pair[0])
            elif len(pair) == 1:
                matches.append(pair[0])
    else:
        matches = list(bf.match(desc_ref, desc_rel))

    if use_mnn and matches:
        nn21 = {m.queryIdx: m.trainIdx for m in bf.match(desc_rel, desc_ref)}
        matches = [m for m in matches if nn21.get(m.trainIdx) == m.queryIdx]

    if not matches:
        return zero_mma, rep, zero_hom_acc, 0

    # --- MMA (computed before RANSAC) ---
    errors = np.array([
        np.linalg.norm(
            np.array(_project(kps_ref[m.queryIdx].pt, H_ref_to_rel))
            - np.array(kps_rel[m.trainIdx].pt)
        )
        for m in matches
    ])
    n_kp = len(kps_ref)   # denominator = keypoints in overlap region
    mma = {th: float(np.sum(errors < th)) / n_kp for th in thresholds}

    # --- Homography estimation via RANSAC (SuperPoint / SiLK metric) ---
    hom_acc = zero_hom_acc
    if estimate_homography and len(matches) >= 4:
        src = np.float32([kps_ref[m.queryIdx].pt for m in matches])
        dst = np.float32([kps_rel[m.trainIdx].pt for m in matches])
        # estimate H mapping rel → ref (same convention as H_rel_to_ref)
        H_est, _ = cv2.findHomography(dst, src, cv2.RANSAC, ransac_reproj)
        if H_est is not None:
            # 4 corners of the reference image
            corners = np.array([[0, 0], [w_ref - 1, 0],
                                 [w_ref - 1, h_ref - 1], [0, h_ref - 1]], dtype=np.float64)
            # project corners ref → rel with GT and estimated homography
            corners_gt  = np.array([_project(c, H_ref_to_rel) for c in corners])
            H_ref_to_rel_est = np.linalg.inv(H_est)
            corners_est = np.array([_project(c, H_ref_to_rel_est) for c in corners])
            mean_err = float(np.mean(np.linalg.norm(corners_gt - corners_est, axis=1)))
            hom_acc = {eps: 1.0 if mean_err < eps else 0.0 for eps in hom_thresholds}

    return mma, rep, hom_acc, len(matches)


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

SCOPES       = ["overall", "illumination", "viewpoint"]
DIFFICULTIES = ["all", "easy", "normal", "hard"]

def aggregate(raw):
    """
    raw: list of (seq_type, img_idx, mma_dict, rep_dict, hom_acc_dict)
          seq_type ∈ {'illumination', 'viewpoint'}
          img_idx  ∈ {2, 3, 4, 5, 6}

    Returns flat dict with per-threshold and AUC columns for mma, rep, and hom_acc,
    broken down by scope × difficulty and per image index.
    """
    mma_diff = {}
    rep_diff = {}
    hom_diff = {}
    mma_img  = {}
    rep_img  = {}
    hom_img  = {}

    for scope in SCOPES:
        for diff in DIFFICULTIES:
            for th in PIXEL_THRESHOLDS:
                mma_diff[(scope, diff, th)] = []
                rep_diff[(scope, diff, th)] = []
            for eps in HOM_THRESHOLDS:
                hom_diff[(scope, diff, eps)] = []
        for img_idx in IMG_INDICES:
            for th in PIXEL_THRESHOLDS:
                mma_img[(scope, img_idx, th)] = []
                rep_img[(scope, img_idx, th)] = []
            for eps in HOM_THRESHOLDS:
                hom_img[(scope, img_idx, eps)] = []

    for seq_type, img_idx, mma_dict, rep_dict, hom_acc_dict in raw:
        scope_tag = "illumination" if seq_type == "illumination" else "viewpoint"
        diffs = _diffs_of(img_idx - 2)
        for scope in [scope_tag, "overall"]:
            for diff in diffs + ["all"]:
                for th, val in mma_dict.items():
                    mma_diff[(scope, diff, th)].append(val)
                for th, val in rep_dict.items():
                    rep_diff[(scope, diff, th)].append(val)
                for eps, val in hom_acc_dict.items():
                    hom_diff[(scope, diff, eps)].append(val)
            for th, val in mma_dict.items():
                mma_img[(scope, img_idx, th)].append(val)
            for th, val in rep_dict.items():
                rep_img[(scope, img_idx, th)].append(val)
            for eps, val in hom_acc_dict.items():
                hom_img[(scope, img_idx, eps)].append(val)

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    result = {}
    for scope in SCOPES:
        for diff in DIFFICULTIES:
            mma_th   = {th:  _mean(mma_diff[(scope, diff, th)])  for th  in PIXEL_THRESHOLDS}
            rep_th   = {th:  _mean(rep_diff[(scope, diff, th)])  for th  in PIXEL_THRESHOLDS}
            eps_vals = {eps: _mean(hom_diff[(scope, diff, eps)]) for eps in HOM_THRESHOLDS}
            result[f"mma_{scope}_{diff}_auc"]     = float(np.mean(list(mma_th.values())))
            result[f"rep_{scope}_{diff}_auc"]     = float(np.mean(list(rep_th.values())))
            result[f"hom_acc_{scope}_{diff}_auc"] = float(np.mean(list(eps_vals.values())))
            for th, v in mma_th.items():
                result[f"mma_{scope}_{diff}_th{th}"] = float(v)
            for th, v in rep_th.items():
                result[f"rep_{scope}_{diff}_th{th}"] = float(v)
            for eps, v in eps_vals.items():
                result[f"hom_acc_{scope}_{diff}_eps{eps}"] = float(v)

        for img_idx in IMG_INDICES:
            mma_th   = {th:  _mean(mma_img[(scope, img_idx, th)])  for th  in PIXEL_THRESHOLDS}
            rep_th   = {th:  _mean(rep_img[(scope, img_idx, th)])  for th  in PIXEL_THRESHOLDS}
            eps_vals = {eps: _mean(hom_img[(scope, img_idx, eps)]) for eps in HOM_THRESHOLDS}
            result[f"mma_{scope}_img{img_idx}_auc"]     = float(np.mean(list(mma_th.values())))
            result[f"rep_{scope}_img{img_idx}_auc"]     = float(np.mean(list(rep_th.values())))
            result[f"hom_acc_{scope}_img{img_idx}_auc"] = float(np.mean(list(eps_vals.values())))

    return result


# ============================================================
# MAIN LOOP
# ============================================================
warnings.filterwarnings("once", category=UserWarning)
os.makedirs("results", exist_ok=True)

for combo_key, extractor in tqdm(test_combinations.items(), desc="Combinations"):
    print(f"\n--- {combo_key} ---")
    with optional_try(SKIP_AT_ERROR, combo_key):
        raw_pairs = []    # (seq_type, difficulty, mma_dict, hom_acc_dict)
        match_counts = []

        for name, seq_type, imgs, homos in tqdm(sequences, leave=False, desc="Sequences"):
            img_ref = imgs[0]
            for rel_idx, (img_rel, H_rel_to_ref) in enumerate(zip(imgs[1:], homos)):
                img_idx = rel_idx + 2
                mma, rep, hom_acc, n_matches = compute_pair_metrics(
                    img_ref, img_rel, H_rel_to_ref,
                    extractor,
                    MAX_FEATURES, PIXEL_THRESHOLDS, HOM_THRESHOLDS,
                    USE_MNN, RATIO_THRESHOLD, ESTIMATE_HOMOGRAPHY, RANSAC_REPROJ,
                )
                raw_pairs.append((seq_type, img_idx, mma, rep, hom_acc))
                match_counts.append(n_matches)

        result = {"combination": combo_key,
                  "avg_num_matches": float(np.mean(match_counts)) if match_counts else 0.0}
        result.update(aggregate(raw_pairs))

        for k, v in result.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        df = pd.DataFrame([result])
        write_header = not os.path.isfile(RESULTS_FILE)
        df.to_csv(RESULTS_FILE, index=False, header=write_header, mode="a")

print("\nDone. Results written to", RESULTS_FILE)
