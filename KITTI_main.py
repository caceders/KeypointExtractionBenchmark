
import cv2
import numpy as np
from pathlib import Path
import csv
import math
from typing import Dict, List, Tuple, Optional
from shi_tomasi_sift import ShiTomasiSift
from benchmark.utils import downsample, visualize_matches_with_scale_change, optional_try, non_maximal_supression
import os
import pandas as pd


#########################################################
# ================= USER CONFIG =========================
#########################################################

DATA_ROOT = "./KITTI/data_odometry_gray/dataset"
#SEQUENCES = ["00", "01", "02", "03", "04", "05"]
SEQUENCES = ["00"]
RUN_NAME = "test"
METHOD_SUFFIX = "sig6"

ACTIVE_FRAMES = (0,1000)  #empty for full sequence
MAX_FEATURES = 500
LOWE_RATIO = 0.75
PNP_REPROJ_THRESH = 2
EPIPOLAR_TOL = 1
RPE_DELTA = 1
downsample_levels = [0]
initial_gaussian_blur_sigma = 0

apply_progressive_blur = False
intrinsic_gaussian_blur_sigma = 0.5
downsample_factor = 2
downsample_interpolation_type = None

APPLY_NMS = False
NMS_RADIUS = 1

skip_at_error = False

BASE_OUT = Path("KITTI/results") / RUN_NAME
CSV_PATH = BASE_OUT / "results.csv"
TRAJ_DIR = BASE_OUT / "trajectories"
BASE_OUT.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

 
#########################################################
# ===========  YOUR FEATURE COMBINATIONS  ===============
#########################################################


features2d = {
    #"AKAZE" : cv2.AKAZE_create(),
    #"BRISK" : cv2.BRISK_create(),
    #"GFTT" : cv2.GFTTDetector_create(),
    "ORB" : cv2.ORB_create(),
    #"ORB_NO_PYRAMID" : cv2.ORB_create(nlevels=1),
    #"SIFT" : cv2.SIFT_create(),
    #"SIFT_LOW_THRESHOLD" : cv2.SIFT_create(contrastThreshold = 0.01, edgeThreshold = 100),
    #"SHIFT" : ShiTomasiSift(starting_level_scale_pyramid=0, num_octaves_in_scale_pyramid=5),
    #"SHIFT_NO_PYRAMID" : ShiTomasiSift(starting_level_scale_pyramid=0, num_octaves_in_scale_pyramid=1),
}


ONLY_SELF = True #Forces no mixing
ONLY_SELF_EXCEPTIONS = [("GFTT", "SIFT")]
ONLY_USED_AS_DETECTOR = ["GFTT"]                     
ONLY_USED_AS_DESCRIPTOR = []                     
BLACKLIST = []                                 
ALLOWED_DESCRIPTOR_FOR_DETECTOR = {
    # "FAST": "SIFT",
    # "GFTT": "SIFT",
    "ORB" : "ORB",
    "SIFT" : "SIFT",
    "BRISK": "BRISK",
    "SHIFT_5_octaves" : "SHIFT_5_octaves"
}   
ALLOWED_DETECTOR_FOR_DESCRIPTOR = {

}

#########################################################
#   FeatureExtractor helper wrapper (simple version)
#########################################################

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

def run_stereo_vo(seq_root, name, extractor, downsample_level):

    effective_PNP_thresh = PNP_REPROJ_THRESH
    #effective_PNP_thresh = PNP_REPROJ_THRESH + downsample_level
    effective_EPIPOLAR_TOL = EPIPOLAR_TOL 
    #effective_EPIPOLAR_TOL = EPIPOLAR_TOL + downsample_level

    # Load calibration
    P0,P1 = read_kitti_P0P1(seq_root/"calib.txt")
    K = P0[:,:3]

    left,right = load_stereo_images(seq_root)
    if ACTIVE_FRAMES:
        left = left[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]
        right = right[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]

    # Extract first frame
    L0=cv2.imread(str(left[0]),0)
    R0=cv2.imread(str(right[0]),0)

    L0 = downsample(L0, downsample_level, downsample_factor, intrinsic_gaussian_blur_sigma, initial_gaussian_blur_sigma, apply_progressive_blur, downsample_interpolation_type)
    R0 = downsample(R0, downsample_level, downsample_factor, intrinsic_gaussian_blur_sigma, initial_gaussian_blur_sigma, apply_progressive_blur, downsample_interpolation_type)
    

    kL0 = extractor.detect(L0)
    kR0 = extractor.detect(R0)
    if APPLY_NMS:
        kL0 = non_maximal_supression(kL0, NMS_RADIUS, MAX_FEATURES*1.1)
        kR0 = non_maximal_supression(kR0, NMS_RADIUS, MAX_FEATURES*1.1)
    else:
        kL0 = sorted(kL0, key=lambda x:x.response, reverse=True)[:MAX_FEATURES*1.1]
        kR0 = sorted(kR0, key=lambda x:x.response, reverse=True)[:MAX_FEATURES*1.1]
    
    
    kL0, dL0 = extractor.compute(L0,kL0)[:MAX_FEATURES]
    kR0, dR0 = extractor.compute(R0,kR0)[:MAX_FEATURES]


    for keypoint in kL0:
        keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_level, keypoint.pt[1] * downsample_factor ** downsample_level)
    for keypoint in kR0:
        keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_level, keypoint.pt[1] * downsample_factor ** downsample_level)


    matcher = cv2.BFMatcher(extractor.norm)

    matchesLR0 = matcher.knnMatch(dL0, dR0, 2)
    good = [m for m,n in matchesLR0 if m.distance < LOWE_RATIO*n.distance]

    tri_prev = triangulate_stereo(kL0,kR0,good,P0,P1,effective_EPIPOLAR_TOL)

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


    for i in range(1,len(left)):
        L=cv2.imread(str(left[i]),0)
        R=cv2.imread(str(right[i]),0)

        L = downsample(L, downsample_level, downsample_factor, intrinsic_gaussian_blur_sigma, initial_gaussian_blur_sigma, apply_progressive_blur, downsample_interpolation_type)
        R = downsample(R, downsample_level, downsample_factor, intrinsic_gaussian_blur_sigma, initial_gaussian_blur_sigma, apply_progressive_blur, downsample_interpolation_type)

        kpL = extractor.detect(L)
        kpR = extractor.detect(R)
        if APPLY_NMS:
            kpL = non_maximal_supression(kpL,NMS_RADIUS, MAX_FEATURES*1.1)
            kpR = non_maximal_supression(kpR,NMS_RADIUS, MAX_FEATURES*1.1)
        else:
            kpL = sorted(kpL, key=lambda x:x.response, reverse=True)[:MAX_FEATURES*1,1]
            kpR = sorted(kpR, key=lambda x:x.response, reverse=True)[:MAX_FEATURES*1.1]

        kpL, dL = extractor.compute(L,kpL)[:MAX_FEATURES]
        kpR, dR = extractor.compute(R,kpR)[:MAX_FEATURES]

        for keypoint in kpL:
            keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_level, keypoint.pt[1] * downsample_factor ** downsample_level)
        for keypoint in kpR:
            keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_level, keypoint.pt[1] * downsample_factor ** downsample_level)

        # temporal matches
        ml = matcher.knnMatch(dLp, dL, 2)
        good_temporal = [m for m,n in ml if m.distance < LOWE_RATIO*n.distance]

        #visualize_matches_with_scale_change(13, np.stack([Lp, Lp, Lp], axis=2), np.stack([L, L, L], axis=2), np.eye(3), good_temporal)
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
        tri_map = triangulate_stereo(kpL,kpR,good,P0,P1,effective_EPIPOLAR_TOL)
        stats["triangulated"].append(len(tri_map))

        kLp, dLp = kpL, dL

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
        for downsample_level in downsample_levels:
            print(f"Running test with downsample level {downsample_level}")
            for name, extractor in test_combinations.items():
                with optional_try(skip_at_error, f"{name}_{METHOD_SUFFIX}_{downsample_level}"):
                    print(f" -> Testing {name}_{METHOD_SUFFIX}_{downsample_level}")

                    poses, stats = run_stereo_vo(seq_root, name, extractor, downsample_level)
                    if ACTIVE_FRAMES:
                        gt_poses = gt_poses[ACTIVE_FRAMES[0]:ACTIVE_FRAMES[1]]

                    traj_path = TRAJ_DIR / f"traj_{seq}_{name}_{METHOD_SUFFIX}_{downsample_level}.txt"
                    save_trajectory_kitti(traj_path, poses)

                    if gt_poses is None:
                        ate_aligned = float("nan")
                        ate_strict = float("nan")
                        rpe1_trans = rpe1_rot = float("nan")
                        rpe10_trans = rpe10_rot = float("nan")
                    else:
                        ate_aligned = compute_ate_aligned(poses, gt_poses)
                        ate_strict = compute_ate_strict(poses, gt_poses)
                        rpe1_trans, rpe1_rot, rpe1_trans_max, rpe1_rot_max, rpe1_trans_std, rpe1_rot_std = compute_rpe(poses, gt_poses, delta=1)
                        rpe10_trans, rpe10_rot, rpe10_trans_max, rpe10_rot_max, rpe10_trans_std, rpe10_rot_std = compute_rpe(poses, gt_poses, delta=10)


                    # ---- build per-method result dict ----
                    results = {
                        "sequence": seq,
                        "method": name + f"_{METHOD_SUFFIX}_{downsample_level}",
                        "active_frames" : f"{ACTIVE_FRAMES[0]}-{ACTIVE_FRAMES[1]}" if ACTIVE_FRAMES else f"0-{len(gt_poses)-1}",
                        "ATE_RMSE_STRICT": ate_strict,
                        "ATE_RMSE_ALIGNED": ate_aligned,
                        "RPE1_trans_RMSE": rpe1_trans,
                        "RPE1_rot_RMSE": rpe1_rot,
                        "RPE1_trans_std": rpe1_trans_std,
                        "RPE1_rot_std": rpe1_rot_std,
                        "RPE10_trans_std": rpe10_trans_std,
                        "RPE10_rot_std": rpe10_rot_std,
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

def compute_ate_aligned(est,gt):
    n=min(len(est),len(gt))
    est=est[:n]; gt=gt[:n]
    R,t=align_no_scale(est,gt)
    E=positions(est); E=(R@E.T).T+t.ravel()
    G=positions(gt)
    err=np.linalg.norm(E-G,axis=1)
    return float(np.sqrt((err**2).mean()))


def compute_ate_strict(est, gt):
    n = min(len(est), len(gt))
    E = positions(est[:n])
    G = positions(gt[:n])
    err = np.linalg.norm(E - G, axis=1)
    return float(np.sqrt(np.mean(err**2)))



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

    return trans_rmse, rot_rmse, max(trans_err), max(rot_err), np.std(trans_err), np.std(rot_err)






if __name__ == "__main__":
    main()
