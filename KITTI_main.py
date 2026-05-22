import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from shi_tomasi_sift import ShiTomasiSift
from pathlib import Path
from matchers import match_nn, match_mnn, match_keem, apply_ratio_uni, apply_ratio_fwd, apply_ratio_bi
from benchmark.utils import downsample, optional_try, non_maximal_supression

# ============================================================
# CONFIGURATION
# ============================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = "./KITTI/data_odometry_gray/dataset"
#SEQUENCE = "01"
SEQUENCE = "00"

# ── Run tag ───────────────────────────────────────────────────────────────────
RUN_NAME = "kitti_ransasc_threshold_check"
RUN_TAG = "1000"

skip_at_error = True

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
ONLY_USED_AS_DESCRIPTOR = []
BLACKLIST = []
ALLOWED_DESCRIPTOR_FOR_DETECTOR = {
    "ORB":   "ORB",
    "SIFT":  "SIFT",
    "BRISK": "BRISK",
}
ALLOWED_DETECTOR_FOR_DESCRIPTOR = {}

# ── Active frames ─────────────────────────────────────────────────────────────
ACTIVE_FRAMES = (0, 1000)   # empty for full sequence

# ── Matching parameters ───────────────────────────────────────────────────────
MAX_KEYPOINTS    = [250,500,750,1000]
MATCHERS         = ["MNN", "NN"]   # "NN", "MNN"
RATIO_THRESHOLDS  = [0.6, 0.8, 1]   # applied to NN and MNN; ignored for KEEM
MNN_BIDIRECTIONAL = [True, False]  # True: bidirectional ratio test for MNN; False: unidirectional (same as NN)
RANSAC_THRESHOLDS   = [1, 3, 5, 10]
EPIPOLAR_THRESHOLDS = [1]

# ── Downsampling parameters ───────────────────────────────────────────────────
DOWNSAMPLE_LEVELS = [0, 1]
INITIAL_SIGMAS    = [0, 2]

apply_progressive_blur = False
intrinsic_gaussian_blur_sigma = 0.5
downsample_factor = 2
downsample_interpolation_type = None

# ── NMS ───────────────────────────────────────────────────────────────────────
APPLY_NMS = False
NMS_RADIUS = 1


BASE_OUT = Path("KITTI/results") / RUN_NAME
CSV_PATH = BASE_OUT / "results.csv"
TRAJ_DIR = BASE_OUT / "trajectories"
BASE_OUT.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BUILD TEST COMBINATIONS
# ============================================================

# ── FeatureExtractor helper ───────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, detect_fn, compute_fn, norm_type):
        self.detect_fn = detect_fn
        self.compute_fn = compute_fn
        self.norm = norm_type

    @staticmethod
    def from_opencv(detect_fn, compute_fn, norm_type):
        return FeatureExtractor(detect_fn, compute_fn, norm_type)

    def detect(self, img):
        return self.detect_fn(img)

    def compute(self, img, kps):
        return self.compute_fn(img, kps)


test_combinations: dict[str, FeatureExtractor] = {}
for detector_key in features2d:
    for descriptor_key in features2d:
        if ONLY_SELF and detector_key != descriptor_key and (detector_key, descriptor_key) not in ONLY_SELF_EXCEPTIONS:
            continue
        if (detector_key, descriptor_key) in BLACKLIST:
            continue
        if detector_key in ONLY_USED_AS_DESCRIPTOR:
            continue
        if descriptor_key in ONLY_USED_AS_DETECTOR:
            continue
        if detector_key in ALLOWED_DESCRIPTOR_FOR_DETECTOR:
            if descriptor_key != ALLOWED_DESCRIPTOR_FOR_DETECTOR[detector_key]:
                continue
        if descriptor_key in ALLOWED_DETECTOR_FOR_DESCRIPTOR:
            if detector_key != ALLOWED_DETECTOR_FOR_DESCRIPTOR[descriptor_key]:
                continue
        try:
            binary_descriptors = (
                cv2.ORB, cv2.BRISK, cv2.AKAZE,
                cv2.xfeatures2d.BriefDescriptorExtractor,
                cv2.xfeatures2d.FREAK,
                cv2.xfeatures2d.LATCH,
            )
        except AttributeError:
            binary_descriptors = (cv2.ORB, cv2.BRISK, cv2.AKAZE)
        distance_type = cv2.NORM_HAMMING if isinstance(features2d[descriptor_key], binary_descriptors) else cv2.NORM_L2
        test_combinations[f"{detector_key}+{descriptor_key}"] = FeatureExtractor.from_opencv(
            features2d[detector_key].detect,
            features2d[descriptor_key].compute,
            distance_type,
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
# MAIN VO LOOP
# ============================================================

def run_stereo_vo_multi(seq_root, extractor, downsample_level,
                        initial_gaussian_blur_sigma, full_configs):
    """
    Process all (max_kp, matcher, ratio_th, ransac_th, epipolar_th) configs in one
    pass over frames. Detects and describes each frame once; updates all VO states.
    Returns {config: (poses, stats)}.
    """
    P0, P1 = read_kitti_P0P1(seq_root / "calib.txt")
    K = P0[:, :3]

    left, right = load_stereo_images(seq_root)
    if ACTIVE_FRAMES:
        left  = left[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]
        right = right[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]

    _max_kp = max(mk for mk, *_ in full_configs)
    scale   = downsample_factor ** downsample_level
    max_kps = sorted(set(mk for mk, *_ in full_configs))

    def _read_ds(path):
        img = cv2.imread(str(path), 0)
        return downsample(img, downsample_level, downsample_factor,
                          intrinsic_gaussian_blur_sigma, initial_gaussian_blur_sigma,
                          apply_progressive_blur, downsample_interpolation_type)

    def _detect_describe(img):
        kps = extractor.detect(img)
        if APPLY_NMS:
            kps = non_maximal_supression(kps, NMS_RADIUS, _max_kp)
        else:
            if len(kps) > _max_kp:
                resp = np.array([kp.response for kp in kps], dtype=np.float32)
                part = np.argpartition(resp, -_max_kp)[-_max_kp:]
                kps  = [kps[i] for i in part[np.argsort(resp[part])[::-1]]]
            else:
                kps = sorted(kps, key=lambda x: x.response, reverse=True)
        if not kps:
            return [], None
        kps, descs = extractor.compute(img, kps)
        if scale != 1:
            for kp in kps:
                kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)
        return kps, descs

    # ── Initialise per-config VO state ────────────────────────────────────────
    def _empty_stats():
        return {"keypoints": [], "keypoints_detected": [], "temporal_matches": [], "stereo_matches": [],
                "triangulated": [], "temporal_tri_map_overlap": [],
                "pnp_inliers": [], "failures": 0}

    states = {cfg: {"poses": [np.eye(4)], "stats": _empty_stats(), "tri_map": {}}
              for cfg in full_configs}

    # previous frame: {max_kp: (kps, descs)}
    prev: dict[int, tuple] = {}

    # ── Frame 0: seed tri_maps ────────────────────────────────────────────────
    kpL_all, dL_all = _detect_describe(_read_ds(left[0]))
    kpR_all, dR_all = _detect_describe(_read_ds(right[0]))

    for max_kp in max_kps:
        kpL0 = kpL_all[:max_kp]
        dL0  = dL_all[:max_kp] if dL_all is not None else None
        kpR0 = kpR_all[:max_kp]
        dR0  = dR_all[:max_kp] if dR_all is not None else None
        prev[max_kp] = (kpL0, dL0)

        raw_stereo0:   dict[str, list] = {}
        stereo0_cache: dict[tuple, list] = {}
        for cfg in full_configs:
            mk, matcher, ratio_th, bidirectional, _, epipolar_th = cfg
            if mk != max_kp:
                continue
            if matcher not in raw_stereo0:
                have = dL0 is not None and dR0 is not None
                if matcher == "NN":
                    raw_stereo0[matcher] = match_nn(dL0, dR0, extractor.norm) if have else []
                elif matcher == "MNN":
                    raw_stereo0[matcher] = match_mnn(dL0, dR0, extractor.norm) if have else []
                else:
                    raw_stereo0[matcher] = match_keem(dL0, dR0, extractor.norm) if have else []
            key = (matcher, ratio_th, bidirectional)
            if key not in stereo0_cache:
                raw = raw_stereo0[matcher]
                if matcher == "NN":
                    stereo0_cache[key] = apply_ratio_uni(raw, ratio_th) if ratio_th is not None else [m.best for m in raw]
                elif matcher == "MNN":
                    _apply = apply_ratio_bi if bidirectional else apply_ratio_fwd
                    stereo0_cache[key] = _apply(raw, ratio_th) if ratio_th is not None else [m.best for m in raw]
                else:
                    stereo0_cache[key] = raw
            states[cfg]["tri_map"] = triangulate_stereo(
                kpL0, kpR0, stereo0_cache[key], P0, P1, epipolar_th)

    # ── Frame loop ────────────────────────────────────────────────────────────
    for i in tqdm(range(1, len(left)), leave=False, desc="Frames", position=3):
        kpL_all, dL_all = _detect_describe(_read_ds(left[i]))
        kpR_all, dR_all = _detect_describe(_read_ds(right[i]))

        for max_kp in max_kps:
            kpL = kpL_all[:max_kp]
            dL  = dL_all[:max_kp] if dL_all is not None else None
            kpR = kpR_all[:max_kp]
            dR  = dR_all[:max_kp] if dR_all is not None else None
            _, prev_dL = prev[max_kp]

            # Raw matches per matcher, ratio applied per (matcher, ratio_th)
            raw_temporal: dict[str, list] = {}
            raw_stereo:   dict[str, list] = {}
            temporal_cache: dict[tuple, list] = {}
            stereo_cache:   dict[tuple, list] = {}
            for cfg in full_configs:
                mk, matcher, ratio_th, bidirectional, *_ = cfg
                if mk != max_kp:
                    continue
                if matcher not in raw_temporal:
                    have_t = prev_dL is not None and dL is not None
                    have_s = dL is not None and dR is not None
                    if matcher == "NN":
                        raw_temporal[matcher] = match_nn(prev_dL, dL, extractor.norm) if have_t else []
                        raw_stereo[matcher]   = match_nn(dL, dR, extractor.norm) if have_s else []
                    elif matcher == "MNN":
                        raw_temporal[matcher] = match_mnn(prev_dL, dL, extractor.norm) if have_t else []
                        raw_stereo[matcher]   = match_mnn(dL, dR, extractor.norm) if have_s else []
                    else:
                        raw_temporal[matcher] = match_keem(prev_dL, dL, extractor.norm) if have_t else []
                        raw_stereo[matcher]   = match_keem(dL, dR, extractor.norm) if have_s else []
                key = (matcher, ratio_th, bidirectional)
                if key not in temporal_cache:
                    rt = raw_temporal[matcher]
                    rs = raw_stereo[matcher]
                    if matcher == "NN":
                        temporal_cache[key] = apply_ratio_uni(rt, ratio_th) if ratio_th is not None else [m.best for m in rt]
                        stereo_cache[key]   = apply_ratio_uni(rs, ratio_th) if ratio_th is not None else [m.best for m in rs]
                    elif matcher == "MNN":
                        _apply = apply_ratio_bi if bidirectional else apply_ratio_fwd
                        temporal_cache[key] = _apply(rt, ratio_th) if ratio_th is not None else [m.best for m in rt]
                        stereo_cache[key]   = _apply(rs, ratio_th) if ratio_th is not None else [m.best for m in rs]
                    else:
                        temporal_cache[key] = rt
                        stereo_cache[key]   = rs

            for cfg in full_configs:
                mk, matcher, ratio_th, bidirectional, ransac_th, epipolar_th = cfg
                if mk != max_kp:
                    continue
                state         = states[cfg]
                good_temporal = temporal_cache[(matcher, ratio_th, bidirectional)]
                good_stereo   = stereo_cache[(matcher, ratio_th, bidirectional)]

                state["stats"]["keypoints"].append((len(kpL) + len(kpR)) / 2)
                state["stats"]["keypoints_detected"].append((len(kpL_all) + len(kpR_all)) / 2)
                state["stats"]["temporal_matches"].append(len(good_temporal))
                state["stats"]["stereo_matches"].append(len(good_stereo))

                pts3d, pts2d, n_overlap = [], [], 0
                for m in good_temporal:
                    if m.query_idx in state["tri_map"]:
                        pts3d.append(state["tri_map"][m.query_idx])
                        pts2d.append(kpL[m.train_idx].pt)
                        n_overlap += 1
                state["stats"]["temporal_tri_map_overlap"].append(n_overlap)

                res = solve_pnp(pts3d, pts2d, K, ransac_th)
                if res is None:
                    state["stats"]["pnp_inliers"].append(0)
                    state["stats"]["failures"] += 1
                    state["poses"].append(state["poses"][-1].copy())
                else:
                    Rot, t, inl = res
                    state["stats"]["pnp_inliers"].append(len(inl))
                    T = build_T(Rot, t)
                    state["poses"].append(state["poses"][-1] @ T_inv(T))

                new_tri = triangulate_stereo(kpL, kpR, good_stereo, P0, P1, epipolar_th)
                state["tri_map"] = new_tri
                state["stats"]["triangulated"].append(len(new_tri))

            prev[max_kp] = (kpL, dL)

    return {cfg: (states[cfg]["poses"], states[cfg]["stats"]) for cfg in full_configs}


# ============================================================
# MAIN RUN
# ============================================================

def main():
    seq      = SEQUENCE
    seq_root = Path(DATA_ROOT) / "sequences" / seq
    gt_path  = Path(DATA_ROOT) / "poses" / f"{seq}.txt"
    gt_poses = read_gt_poses(gt_path)

    for name, extractor in tqdm(test_combinations.items(), leave=True, desc="Methods", position=0):
        for initial_gaussian_blur_sigma in tqdm(INITIAL_SIGMAS, leave=False, desc="Initial sigmas", position=1):
            for downsample_level in tqdm(DOWNSAMPLE_LEVELS, leave=False, desc="Downsample levels", position=2):

                full_configs = [
                    (max_kp, matcher, ratio_th, bidirectional, ransac_th, epipolar_th)
                    for max_kp      in MAX_KEYPOINTS
                    for matcher, ratio_th, bidirectional in _matching_configs
                    for ransac_th   in RANSAC_THRESHOLDS
                    for epipolar_th in EPIPOLAR_THRESHOLDS
                ]

                with optional_try(skip_at_error, f"{name}_{RUN_TAG}_{downsample_level}"):
                    all_results = run_stereo_vo_multi(
                        seq_root, extractor, downsample_level,
                        initial_gaussian_blur_sigma, full_configs,
                    )

                    _gt = gt_poses
                    if ACTIVE_FRAMES and _gt is not None:
                        _gt = _gt[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]

                    for cfg, (poses, stats) in all_results.items():
                        max_kp, matcher_name, ratio_th, bidirectional, ransac_th, epipolar_th = cfg

                        ratio_str = f"{ratio_th:.2f}" if ratio_th is not None else "none"
                        traj_path = (TRAJ_DIR /
                                     f"traj_{seq}_{name}_{RUN_TAG}_{matcher_name}"
                                     f"_{ratio_str}_{downsample_level}"
                                     f"_r{ransac_th}_e{epipolar_th}.txt")
                        save_trajectory_kitti(traj_path, poses)

                        if _gt is None:
                            ate_aligned = ate_strict = float("nan")
                            rpe1_trans = rpe1_rot = float("nan")
                            rpe10_trans = rpe10_rot = float("nan")
                            rpe1_trans_max = rpe1_rot_max = float("nan")
                            rpe10_trans_max = rpe10_rot_max = float("nan")
                            rpe1_trans_std = rpe1_rot_std = float("nan")
                            rpe10_trans_std = rpe10_rot_std = float("nan")
                        else:
                            ate_aligned = compute_ate_aligned(poses, _gt)
                            ate_strict  = compute_ate_strict(poses, _gt)
                            rpe1_trans,  rpe1_rot,  rpe1_trans_max,  rpe1_rot_max,  rpe1_trans_std,  rpe1_rot_std  = compute_rpe(poses, _gt, delta=1)
                            rpe10_trans, rpe10_rot, rpe10_trans_max, rpe10_rot_max, rpe10_trans_std, rpe10_rot_std = compute_rpe(poses, _gt, delta=10)

                        results = {
                            # Identity
                            "sequence":      seq,
                            "method":        name,
                            "tag":           RUN_TAG,
                            "active_frames": (f"{ACTIVE_FRAMES[0]}-{ACTIVE_FRAMES[1]}"
                                              if ACTIVE_FRAMES else f"0-{len(gt_poses)-1}"),
                            # Matching parameters
                            "matcher":            matcher_name,
                            "ratio_threshold":    ratio_th if ratio_th is not None else "-",
                            "mnn_bidirectional":  bidirectional if bidirectional is not None else "-",
                            "ransac_threshold":   ransac_th,
                            "epipolar_threshold": epipolar_th,
                            # Pipeline parameters
                            "downsample_level": downsample_level,
                            "initial_sigma":    initial_gaussian_blur_sigma,
                            "max_keypoints":    max_kp,
                            # Trajectory metrics
                            "ATE_RMSE_STRICT":   ate_strict,
                            "ATE_RMSE_ALIGNED":  ate_aligned,
                            "RPE1_trans_RMSE":   rpe1_trans,
                            "RPE1_rot_RMSE":     rpe1_rot,
                            "RPE1_trans_std":    rpe1_trans_std,
                            "RPE1_rot_std":      rpe1_rot_std,
                            "RPE10_trans_RMSE":  rpe10_trans,
                            "RPE10_rot_RMSE":    rpe10_rot,
                            "RPE10_trans_std":   rpe10_trans_std,
                            "RPE10_rot_std":     rpe10_rot_std,
                            "RPE1_trans_max":    rpe1_trans_max,
                            "RPE1_rot_max":      rpe1_rot_max,
                            "RPE10_trans_max":   rpe10_trans_max,
                            "RPE10_rot_max":     rpe10_rot_max,
                            # Run statistics
                            "avg_num_keypoints_detected":            float(np.mean(stats["keypoints_detected"])),
                            "avg_num_keypoints":                     float(np.mean(stats["keypoints"])),
                            "avg_num_temporal_matches":              float(np.mean(stats["temporal_matches"])),
                            "avg_num_stereo_matches":                float(np.mean(stats["stereo_matches"])),
                            "avg_num_triangulated_matches":          float(np.mean(stats["triangulated"])),
                            "avg_num_temporal_tri_map_overlap":      float(np.mean(stats["temporal_tri_map_overlap"])),
                            "avg_num_PnP_inliers":                   float(np.mean(stats["pnp_inliers"])),
                            "avg_num_dropped_temporal":              float(np.mean(stats["keypoints"])) - float(np.mean(stats["temporal_matches"])),
                            "avg_num_dropped_stereo":                float(np.mean(stats["keypoints"])) - float(np.mean(stats["stereo_matches"])),
                            "avg_num_dropped_stereo->tri":           float(np.mean(stats["stereo_matches"])) - float(np.mean(stats["triangulated"])),
                            "avg_num_dropped_temporal->tri_overlap": float(np.mean(stats["temporal_matches"])) - float(np.mean(stats["temporal_tri_map_overlap"])),
                            "avg_num_dropped_tri_overlap->PNP":      float(np.mean(stats["temporal_tri_map_overlap"])) - float(np.mean(stats["pnp_inliers"])),
                            "failures": int(stats["failures"]),
                        }
                        #print(f" method: {name} matcher: {matcher_name} bidirect: {bidirectional} ratio_thresh: {ratio_th} ate: {ate_strict} rpe: {rpe1_trans} pnp_inliers: {float(np.mean(stats["pnp_inliers"]))}")

                        df = pd.DataFrame(results, index=[0])
                        write_header = not CSV_PATH.exists()
                        df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)

    print(f"Results saved to {CSV_PATH}")


# ============================================================
# KITTI HELPERS
# ============================================================

def save_trajectory_kitti(path: Path, poses):
    with open(path, "w") as f:
        for T in poses:
            line = " ".join(f"{v:.6f}" for v in T[:3, :].reshape(-1))
            f.write(line + "\n")


def read_kitti_P0P1(calib_file):
    P = {}
    with open(calib_file, "r") as f:
        for line in f:
            key, _, vals = line.partition(":")
            vals = np.fromstring(vals, sep=" ")
            if key in ("P0", "P1"):
                P[key] = vals.reshape(3, 4)
    return P["P0"], P["P1"]


def load_stereo_images(seq_root):
    L = sorted((seq_root / "image_0").glob("*.png"))
    R = sorted((seq_root / "image_1").glob("*.png"))
    return L, R


def read_gt_poses(path):
    if not path.exists():
        return None
    out = []
    for line in open(path):
        v = list(map(float, line.split()))
        T = np.eye(4)
        T[:3, :4] = np.array(v).reshape(3, 4)
        out.append(T)
    return out


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti


def build_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


# ============================================================
# STEREO + PNP
# ============================================================

def triangulate_stereo(kL, kR, matches, P0, P1, epip_tol):
    ptsL = []; ptsR = []; idx = []
    for m in matches:
        l = kL[m.query_idx].pt
        r = kR[m.train_idx].pt
        if abs(l[1] - r[1]) > epip_tol:
            continue
        if l[0] - r[0] <= 0:
            continue
        idx.append(m.query_idx)
        ptsL.append(l)
        ptsR.append(r)

    if len(ptsL) < 6:
        return {}

    pL = np.float32(ptsL).T
    pR = np.float32(ptsR).T

    Xh = cv2.triangulatePoints(P0, P1, pL, pR)
    X  = (Xh[:3] / (Xh[3] + 1e-9)).T

    out = {}
    for i, xyz in zip(idx, X):
        if xyz[2] > 0:
            out[i] = xyz
    return out


def solve_pnp(X, pts2d, K, thresh):
    if len(X) < 6:
        return None
    X     = np.asarray(X).astype(np.float32)
    pts2d = np.asarray(pts2d).astype(np.float32)

    ok, r, t, inl = cv2.solvePnPRansac(
        X, pts2d, K, None,
        iterationsCount=1000,
        reprojectionError=thresh,
        confidence=0.999999,
    )
    if not ok or inl is None or len(inl) < 6:
        return None

    R, _ = cv2.Rodrigues(r)
    return R, t, inl.ravel()


# ============================================================
# TRAJECTORY METRICS
# ============================================================

def positions(poses):
    out = np.zeros((len(poses), 3))
    for i, T in enumerate(poses):
        out[i] = T[:3, 3]
    return out


def align_no_scale(est, gt):
    E = positions(est); G = positions(gt)
    muE = E.mean(0);    muG = G.mean(0)
    E0  = E - muE;      G0  = G - muG
    H   = E0.T @ G0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = muG[:, None] - R @ muE[:, None]
    return R, t


def compute_ate_aligned(est, gt):
    n = min(len(est), len(gt))
    est = est[:n]; gt = gt[:n]
    R, t = align_no_scale(est, gt)
    E = positions(est)
    E = (R @ E.T).T + t.ravel()
    G = positions(gt)
    err = np.linalg.norm(E - G, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def compute_ate_strict(est, gt):
    n = min(len(est), len(gt))
    E = positions(est[:n])
    G = positions(gt[:n])
    err = np.linalg.norm(E - G, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def relative_pose(T1, T2):
    return np.linalg.inv(T1) @ T2


def rotation_error_deg(R):
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def compute_rpe(est_poses, gt_poses, delta=1):
    trans_err = []
    rot_err   = []
    n = min(len(est_poses), len(gt_poses))
    for i in range(n - delta):
        T_est_rel = relative_pose(est_poses[i], est_poses[i + delta])
        T_gt_rel  = relative_pose(gt_poses[i],  gt_poses[i + delta])
        T_err     = relative_pose(T_gt_rel, T_est_rel)
        trans_err.append(np.linalg.norm(T_err[:3, 3]))
        rot_err.append(rotation_error_deg(T_err[:3, :3]))

    trans_rmse = np.sqrt(np.mean(np.square(trans_err)))
    rot_rmse   = np.sqrt(np.mean(np.square(rot_err)))
    return trans_rmse, rot_rmse, max(trans_err), max(rot_err), np.std(trans_err), np.std(rot_err)


if __name__ == "__main__":
    main()
