
import cv2
import numpy as np
from pathlib import Path
import csv
import math
from typing import Dict, List, Tuple, Optional
from shi_tomasi_sift import ShiTomasiSift
from benchmark.utils import downsample
from benchmark.feature_extractor import FeatureExtractor
import os
import pandas as pd

#########################################################
# ================= USER CONFIG =========================
#########################################################

DATA_ROOT = "./KITTI/data_odometry_gray/dataset"
#SEQUENCES = ["00", "01", "02", "03", "04", "05"]
SEQUENCES = ["00"]
RUN_NAME = "FRAME_TEST_LEFT_FLIP"
METHOD_SUFFIX = ""
BASE_OUT = Path("KITTI/results") / RUN_NAME
CSV_PATH = BASE_OUT / "results.csv"
TRAJ_DIR = BASE_OUT / "trajectories"
BASE_OUT.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_FRAMES = (0,250)  # or e.g. 500, 1000
MAX_FEATURES = 500
LOWE_RATIO = 0.75
PNP_REPROJ_THRESH = 2
EPIPOLAR_TOL = 1
RPE_DELTA = 1
initial_gaussian_blur_sigma = 1
downsample_iterations_nums = [0]
downsample_factor = 1.2
downsample_interpolation_type = cv2.INTER_LINEAR
downsample_gaussian_sigma = -1
 
#########################################################
# ===========  YOUR FEATURE COMBINATIONS  ===============
#########################################################


features2d = {
    #"AKAZE" : cv2.AKAZE_create(),
    "BRISK" : cv2.BRISK_create(),
    #"FAST" : cv2.FastFeatureDetector_create(),
    #"FAST2" : cv2.FastFeatureDetector_create(threshold = 15),
    #"GFTT" : cv2.GFTTDetector_create(),
    #"GFTT2" : cv2.GFTTDetector_create(blockSize = 6, qualityLevel = 0.005),
    #"ORB" : cv2.ORB_create(),
    "ORB_NO_PYRAMID" : cv2.ORB_create(nlevels=1),
    #"ORB_4_LAYERS" : cv2.ORB_create(nlevels=4),
    #"SIFT" : cv2.SIFT_create(),
    #"SIFT_LOW_THRESHOLD" : cv2.SIFT_create(contrastThreshold = 0.01, edgeThreshold = 100),
    #"SIFT_FAST2" : cv2.SIFT_create(sigma = 2.25),
    #"SIFT_GFTT2" : SIFT_GFTT2 = cv2.SIFT_create(),
    #"SIFT_SIG_3.5" : cv2.SIFT_create(sigma = 3.5),
    #"BRIEF" : cv2.xfeatures2d.BriefDescriptorExtractor_create(),
    #"SHIFT_3_octaves" : ShiTomasiSift(starting_level_scale_pyramid=0, num_octaves_in_scale_pyramid=3),
    #"SHIFT_NO_PYRAMID" : ShiTomasiSift(starting_level_scale_pyramid=0, num_octaves_in_scale_pyramid=1),
}

GFTT2_SCALE = 2
FAST2_SCALE = 1.5


ONLY_SELF = True #Forces no mixing
ONLY_SELF_EXCEPTIONS = [("GFTT", "SIFT"), ("GFTT2", "SIFT")]
ONLY_USED_AS_DETECTOR = ["GFTT", "FAST2", "GFTT2"]                     
ONLY_USED_AS_DESCRIPTOR = ["SIFT_FAST2", "SIFT_GFTT2"]                     
BLACKLIST = []                                 
ALLOWED_DESCRIPTOR_FOR_DETECTOR = {
    # "FAST": "SIFT",
    "FAST2": "SIFT_FAST2",
    # "GFTT": "SIFT",
    "GFTT2": "SIFT_GFTT2",
    "GFTT2": "SIFT",
    "ORB" : "ORB",
    "SIFT" : "SIFT",
    "BRISK": "BRISK",
    "SHIFT_5_octaves" : "SHIFT_5_octaves"
}   
ALLOWED_DETECTOR_FOR_DESCRIPTOR = {
    "SIFT_FAST2": "FAST2",
}


test_combinations: dict[str, FeatureExtractor] = {}
for detector_key in features2d.keys():
    for descriptor_key in features2d.keys():

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

        if descriptor_key in ALLOWED_DETECTOR_FOR_DESCRIPTOR :
            if detector_key != ALLOWED_DETECTOR_FOR_DESCRIPTOR[descriptor_key]:
                continue

        
        binary_descriptors = (
            cv2.ORB, cv2.BRISK, cv2.AKAZE, cv2.xfeatures2d.BriefDescriptorExtractor, cv2.xfeatures2d.FREAK, cv2.xfeatures2d.LATCH
        )

        if isinstance(features2d[descriptor_key], binary_descriptors):
            distance_type = cv2.NORM_HAMMING
        else:
            distance_type = cv2.NORM_L2


        test_combinations[detector_key + "+" + descriptor_key] = FeatureExtractor.from_opencv(features2d[detector_key].detect, features2d[descriptor_key].compute, distance_type)


#########################################################
# ==================== MAIN VO LOOP =====================
#########################################################

def run_stereo_vo(seq_root, name, extractor, downsample_iterations, timer=None):

    # Load calibration
    P0,P1 = read_kitti_P0P1(seq_root/"calib.txt")
    K = P0[:,:3]

    left,right = load_stereo_images(seq_root)
    left = left[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]
    right = right[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]

    # Extract first frame
    L0=cv2.imread(str(left[0]),0)
    R0=cv2.imread(str(right[0]),0)

    effective_downsample_factor = downsample_factor**downsample_iterations
    L0 = downsample(L0, effective_downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)
    R0 = downsample(R0, effective_downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)
    
    # for i in range(downsample_iterations):
    #     L0 = downsample(L0, downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)
    #     R0 = downsample(R0, downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)

    kL0 = extractor.detect_keypoints(L0)
    kR0 = extractor.detect_keypoints(R0)

    if name == "FAST2+SIFT_FAST2":
        for keypoint in kL0:
                keypoint.size = keypoint.size * FAST2_SCALE
        for keypoint in kR0:
                keypoint.size = keypoint.size * FAST2_SCALE
    if name == "GFTT2+SIFT_GFTT2":
        for keypoint in kL0:
                keypoint.size = keypoint.size * GFTT2_SCALE
        for keypoint in kR0:
                keypoint.size = keypoint.size * GFTT2_SCALE

    kL0 = sorted(kL0, key=lambda x:x.response, reverse=True)[:MAX_FEATURES]
    kR0 = sorted(kR0, key=lambda x:x.response, reverse=True)[:MAX_FEATURES]
    kL0, dL0 = extractor.describe_keypoints(L0, kL0)
    dL0 = np.array(dL0)
    kR0, dR0 = extractor.describe_keypoints(R0, kR0)
    dR0 = np.array(dR0)

    for keypoint in kL0:
        keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)
    for keypoint in kR0:
        keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)


    matcher = cv2.BFMatcher(extractor.distance_type)

    matchesLR0 = matcher.knnMatch(dL0, dR0, 2)
    good = [m for m,n in matchesLR0 if m.distance < LOWE_RATIO*n.distance]

    tri_prev = triangulate_stereo(kL0,kR0,good,P0,P1,EPIPOLAR_TOL)

    poses=[np.eye(4)]  # world_T_cam
    stats = {
        "keypoints" : [],
        "temporal_matches": [],
        "stereo_matches": [],
        "triangulated": [],
        "temporal_tri_map_overlap": [],
        "pnp_inliers": [],
        "failures": 0
    }

    kLp, dLp = kL0, dL0
    tri_map = tri_prev


    total_frames = len(left) - 1
    for i in range(1,len(left)):
        L=cv2.imread(str(left[i]),0)
        R=cv2.imread(str(right[i]),0)

        effective_downsample_factor = downsample_factor**downsample_iterations
        L = downsample(L, effective_downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)
        R = downsample(R, effective_downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)

        # for i in range(downsample_iterations):
        #     L = downsample(L, downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)
        #     R = downsample(R, downsample_factor, downsample_gaussian_sigma, downsample_interpolation_type)


        kpL = extractor.detect_keypoints(L)
        kpL = sorted(kpL, key=lambda x:x.response, reverse=True)[:MAX_FEATURES]
        kpR = extractor.detect_keypoints(R)
        kpR = sorted(kpR, key=lambda x:x.response, reverse=True)[:MAX_FEATURES]


        if name == "FAST2+SIFT_FAST2":
            for keypoint in kpL:
                    keypoint.size = keypoint.size * FAST2_SCALE
            for keypoint in kpR:
                    keypoint.size = keypoint.size * FAST2_SCALE
        if name == "GFTT2+SIFT_GFTT2":
            for keypoint in kpL:
                    keypoint.size = keypoint.size * GFTT2_SCALE
            for keypoint in kpR:
                    keypoint.size = keypoint.size * GFTT2_SCALE

        kpL, dL = extractor.describe_keypoints(L, kpL)
        dL = np.array(dL)
        kpR, dR = extractor.describe_keypoints(R, kpR)
        dR = np.array(dR)

        for keypoint in kpL:
            keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)
        for keypoint in kpR:
            keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)

        # temporal matches
        ml = matcher.knnMatch(dLp, dL, 2)
        good_temporal = [m for m,n in ml if m.distance < LOWE_RATIO*n.distance]

        stats["keypoints"].append((len(kpL)+len(kpR))/2)
        stats["temporal_matches"].append(len(good_temporal))
        stats["stereo_matches"].append(len(good))

        # Build 3D-2D pairs

        num_temporal_in_tri_map = 0
        pts3d=[]; pts2d=[]
        for m in good_temporal:
            if m.queryIdx in tri_map:
                pts3d.append(tri_map[m.queryIdx])
                pts2d.append(kpL[m.trainIdx].pt)
                num_temporal_in_tri_map += 1

        stats["temporal_tri_map_overlap"].append(num_temporal_in_tri_map)

        # Solve PnP
        effective_PNP_thresh = PNP_REPROJ_THRESH
        #effective_PNP_thresh = PNP_REPROJ_THRESH * (downsample_factor ** downsample_iterations)
        res = solve_pnp(pts3d, pts2d, K, effective_PNP_thresh)
        if res is None:
            stats["pnp_inliers"].append(0)
            stats["failures"]+=1
            poses.append(poses[-1].copy())
        else:
            Rot,t,inl = res
            stats["pnp_inliers"].append(len(inl))

            T = build_T(Rot,t)
            # world_T_cam_next = world_T_cam * inv(T_cam_next_cam)
            poses.append(poses[-1] @ T_inv(T))

        # recompute stereo for next step
        matchesLR = matcher.knnMatch(dL, dR, 2)
        good = [m for m,n in matchesLR if m.distance < LOWE_RATIO*n.distance]
        tri_map = triangulate_stereo(kpL,kpR,good,P0,P1,EPIPOLAR_TOL)
        stats["triangulated"].append(len(tri_map))

        kLp, dLp = kpL, dL
        if timer is not None:
            timer.check(total_frames - i, total_frames)

    return poses, stats


#########################################################
# ===================== MAIN RUN ========================
#########################################################

def main():
    for seq in SEQUENCES:
        seq_root = Path(DATA_ROOT) / "sequences" / seq
        gt_path = Path(DATA_ROOT) / "poses" / f"{seq}.txt"
        gt_poses = read_gt_poses(gt_path)
        print(f"=== Running sequence {seq} ===")
        for downsample_iterations in downsample_iterations_nums:
            print(f"Running test with {downsample_iterations} downsample iterations")
            for name, extractor in test_combinations.items():
                print(f" -> Testing {name}")

                poses, stats = run_stereo_vo(seq_root, name, extractor, downsample_iterations)
                gt_poses_trunc = gt_poses[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]

                traj_path = TRAJ_DIR / f"traj_{seq}_{name}_{METHOD_SUFFIX}_{downsample_iterations}.txt"
                save_trajectory_kitti(traj_path, poses)

                if gt_poses is None:
                    ate = float("nan")
                    rpe1_trans = rpe1_rot = float("nan")
                    rpe10_trans = rpe10_rot = float("nan")
                else:
                    ate = compute_ate(poses, gt_poses_trunc)

                    rpe1_trans, rpe1_rot, rpe1_trans_max, rpe1_rot_max = compute_rpe(poses, gt_poses_trunc, delta=1)
                    rpe10_trans, rpe10_rot, rpe10_trans_max, rpe10_rot_max = compute_rpe(poses, gt_poses_trunc, delta=10)


                # ---- build per-method result dict ----
                results = {
                    "sequence": seq,
                    "method": name + f"_{METHOD_SUFFIX}_{downsample_iterations}",
                    "active_frames" : f"{ACTIVE_FRAMES[0]}-{ACTIVE_FRAMES[1]}",
                    "ATE_RMSE": ate,
                    "RPE1_trans_RMSE": rpe1_trans,
                    "RPE1_rot_RMSE": rpe1_rot,
                    "RPE1_trans_max": rpe1_trans_max,
                    "RPE1_rot_max": rpe1_rot_max,
                    "RPE10_trans_RMSE": rpe10_trans,
                    "RPE10_rot_RMSE": rpe10_rot,
                    "RPE10_trans_max": rpe10_trans_max,
                    "RPE10_rot_max": rpe10_rot_max,
                    "keypoints": float(np.mean(stats["keypoints"])),
                    "temporal_matches_mean": float(np.mean(stats["temporal_matches"])),
                    "stereo_matches_mean": float(np.mean(stats["stereo_matches"])),
                    "triangulated_mean": float(np.mean(stats["triangulated"])),
                    "temporal_tri_map_overlap_mean": float(np.mean(stats["temporal_tri_map_overlap"])),
                    "PnP_inliers_mean": float(np.mean(stats["pnp_inliers"])),
                    "dropped_temporal": float(np.mean(stats["keypoints"]))- float(np.mean(stats["temporal_matches"])),
                    "dropped_stereo": float(np.mean(stats["keypoints"]))- float(np.mean(stats["stereo_matches"])),
                    "dropped_stereo->tri": float(np.mean(stats["stereo_matches"]))- float(np.mean(stats["triangulated"])),
                    "dropped_temporal->tri_map_overlap": float(np.mean(stats["temporal_matches"]))- float(np.mean(stats["temporal_tri_map_overlap"])),
                    "dropped_tri_map_overlap->PNP_inliers": float(np.mean(stats["temporal_tri_map_overlap"]))- float(np.mean(stats["pnp_inliers"])),
                    "failures": int(stats["failures"]),
                }

                # ---- print results immediately ----
                for k, v in results.items():
                    print(f"    {k}: {v}")

                # ---- append to CSV safely ----
                df = pd.DataFrame(results, index=[0])

                write_header = not CSV_PATH.exists()
                df.to_csv(
                    CSV_PATH,
                    mode="a",
                    header=write_header,
                    index=False,
                )


#########################################################
# ============== KITTI HELPERS ==========================
#########################################################

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
                P[key] = vals.reshape(3,4)
    return P["P0"], P["P1"]

def load_stereo_images(seq_root):
    L = sorted((seq_root/"image_0").glob("*.png"))
    R = sorted((seq_root/"image_1").glob("*.png"))
    return L, R

def read_gt_poses(path):
    if not path.exists(): return None
    out = []
    for line in open(path):
        v = list(map(float, line.split()))
        T = np.eye(4); T[:3,:4]=np.array(v).reshape(3,4)
        out.append(T)
    return out

def T_inv(T):
    R = T[:3,:3]
    t = T[:3,3:4]
    Ti = np.eye(4)
    Ti[:3,:3]=R.T
    Ti[:3,3:4]= -R.T@t
    return Ti

def build_T(R,t):
    T=np.eye(4); T[:3,:3]=R; T[:3,3:4]=t
    return T

#########################################################
# ============= Stereo + PnP Motion =====================
#########################################################

def triangulate_stereo(kL,kR,matches,P0,P1,epip_tol):
    ptsL=[]; ptsR=[]; idx=[]
    for m in matches:
        l=kL[m.queryIdx].pt; r=kR[m.trainIdx].pt
        if abs(l[1]-r[1])>epip_tol: continue
        if l[0]-r[0] <= 0: continue
        idx.append(m.queryIdx)
        ptsL.append(l); ptsR.append(r)

    if len(ptsL)<6: return {}

    pL=np.float32(ptsL).T
    pR=np.float32(ptsR).T

    Xh=cv2.triangulatePoints(P0,P1,pL,pR)
    X=(Xh[:3]/(Xh[3]+1e-9)).T

    out={}
    for i,xyz in zip(idx,X):
        if xyz[2]>0: out[i]=xyz
    return out

def solve_pnp(X, pts2d, K, thresh):
    if len(X)<6: return None
    X=np.asarray(X).astype(np.float32)
    pts2d=np.asarray(pts2d).astype(np.float32)

    ok,r,t,inl=cv2.solvePnPRansac(
        X,pts2d,K,None,
        iterationsCount=1000,
        reprojectionError=thresh,
        confidence=0.999)
    if not ok or inl is None or len(inl)<6: return None

    R,_=cv2.Rodrigues(r)
    return R,t,inl.ravel()


#########################################################
# ==================
#  TRAJECTORY METRICS
# ==================
#########################################################

def positions(poses):
    out=np.zeros((len(poses),3))
    for i,T in enumerate(poses):
        out[i]=T[:3,3]
    return out

def align_no_scale(est,gt):
    E=positions(est); G=positions(gt)
    muE=E.mean(0); muG=G.mean(0)
    E0=E-muE; G0=G-muG
    H=E0.T@G0
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T@U.T
    if np.linalg.det(R)<0: Vt[-1]*=-1; R=Vt.T@U.T
    t=muG[:,None]-R@muE[:,None]
    return R,t

def compute_ate(est,gt):
    n=min(len(est),len(gt))
    est=est[:n]; gt=gt[:n]
    R,t=align_no_scale(est,gt)
    E=positions(est); E=(R@E.T).T+t.ravel()
    G=positions(gt)
    err=np.linalg.norm(E-G,axis=1)
    return float(np.sqrt((err**2).mean()))


def relative_pose(T1, T2):
    """Compute relative transform T1^{-1} * T2."""
    return np.linalg.inv(T1) @ T2


def rotation_error_deg(R):
    """Rotation error (degrees) from a rotation matrix."""
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def compute_rpe(est_poses, gt_poses, delta=1):
    """
    Frame-based Relative Pose Error (RPE).
    Returns:
      - translation RMSE (meters)
      - rotation RMSE (degrees)
    """
    trans_err = []
    rot_err = []

    n = min(len(est_poses), len(gt_poses))
    for i in range(n - delta):
        T_est_rel = relative_pose(est_poses[i], est_poses[i + delta])
        T_gt_rel  = relative_pose(gt_poses[i], gt_poses[i + delta])

        T_err = relative_pose(T_gt_rel, T_est_rel)

        t_err = np.linalg.norm(T_err[:3, 3])
        r_err = rotation_error_deg(T_err[:3, :3])

        trans_err.append(t_err)
        rot_err.append(r_err)

    trans_rmse = np.sqrt(np.mean(np.square(trans_err)))
    rot_rmse   = np.sqrt(np.mean(np.square(rot_err)))

    return trans_rmse, rot_rmse, max(trans_err), max(rot_err)






def evaluate_kitti(extractor, timer=None):
    """
    Run stereo VO on KITTI sequences and return mean RPE translational RMSE (meters).
    Uses module-level config (DATA_ROOT, SEQUENCES, ACTIVE_FRAMES, etc.).
    """
    rpe_trans = []
    for seq in SEQUENCES:
        seq_root = Path(DATA_ROOT) / "sequences" / seq
        gt_path = Path(DATA_ROOT) / "poses" / f"{seq}.txt"
        gt_poses = read_gt_poses(gt_path)
        if gt_poses is None:
            continue
        poses, _ = run_stereo_vo(seq_root, "_optuna_", extractor, 0, timer=timer)
        gt_trunc = gt_poses[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]
        trans_rmse, _, _, _ = compute_rpe(poses, gt_trunc, delta=RPE_DELTA)
        rpe_trans.append(trans_rmse)
    return float(np.mean(rpe_trans)) if rpe_trans else float("nan")


if __name__ == "__main__":
    main()
