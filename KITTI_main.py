
import cv2
import numpy as np
from pathlib import Path
import csv
import math
from typing import Dict, List, Tuple, Optional
from shi_tomasi_sift import ShiTomasiSift
from benchmark.utils import downsample

#########################################################
# ================= USER CONFIG =========================
#########################################################

DATA_ROOT = "./KITTI/data_odometry_gray/dataset"
#SEQUENCES = ["00", "01", "02", "03", "04", "05"]
SEQUENCES = ["00"]
RUN_NAME = "test_4_1.2"
BASE_OUT = Path("KITTI/results") / RUN_NAME
CSV_PATH = BASE_OUT / "results.csv"
TRAJ_DIR = BASE_OUT / "trajectories"
BASE_OUT.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
MAX_FEATURES = 500
LOWE_RATIO = 0.75
PNP_REPROJ_THRESH = 2.0
EPIPOLAR_TOL = 1.0
RPE_DELTA = 1
downsample_iterations = 4
downsample_factor = 1.2
downsample_interpolation_type = cv2.INTER_LINEAR
downsample_gaussian_sigma = -1
 
#########################################################
# ===========  YOUR FEATURE COMBINATIONS  ===============
#########################################################

# Insert your existing code here (unchanged):
AKAZE = cv2.AKAZE_create()
BRISK = cv2.BRISK_create()
FAST = cv2.FastFeatureDetector_create()
GFTT = cv2.GFTTDetector_create()
ORB = cv2.ORB_create()
ORB_NO_PYRAMID = cv2.ORB_create(nlevels = 1)
#SIFT = cv2.SIFT_create(contrastThreshold = 0.01, edgeThreshold = 100)
SIFT = cv2.SIFT_create()
SIFT_OPTIMAL = cv2.SIFT_create(sigma = 3.5)
BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
FAST2 = cv2.FastFeatureDetector_create(threshold = 15)
FAST2_SCALE = 1.5
GFTT2 = cv2.GFTTDetector_create(blockSize = 6, qualityLevel = 0.005)
GFTT2_SCALE = 2
SIFT_FAST2 = cv2.SIFT_create(sigma = 2.25)
SIFT_GFTT2 = cv2.SIFT_create()

features2d = {
    #"AKAZE" : AKAZE,
    "BRISK" : BRISK,
    #"FAST" : FAST,
    #"FAST2" : FAST2,
    #"GFTT" : GFTT,
    #"GFTT2" : GFTT2,
    "ORB" : ORB,
    "ORB_NO_PYRAMID" : ORB_NO_PYRAMID,
    "SIFT" : SIFT,
    #"SIFT_FAST2" : SIFT_FAST2,
    #"SIFT_GFTT2" : SIFT_GFTT2,
    #"SIFT_OPTIMAL" : SIFT_OPTIMAL,
    #"BRIEF" : BRIEF,
    "SHIFT_5_octaves" : ShiTomasiSift(starting_level_scale_pyramid=0, num_octaves_in_scale_pyramid=5),
    "SHIFT_NO_PYRAMID" : ShiTomasiSift(starting_level_scale_pyramid=0, num_octaves_in_scale_pyramid=1),
}


ONLY_SELF = True #Forces no mixing
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

        if ONLY_SELF and detector_key != descriptor_key:
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

        if descriptor_key in ["BRISK", "ORB", "AKAZE", "BRIEF", "FREAK", "LATCH"]:
            distance_type = cv2.NORM_HAMMING
        else:
            distance_type = cv2.NORM_L2

        test_combinations[detector_key + "+" + descriptor_key] = FeatureExtractor.from_opencv(features2d[detector_key].detect, features2d[descriptor_key].compute, distance_type)







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

    return trans_rmse, rot_rmse


#########################################################
# ==================== MAIN VO LOOP =====================
#########################################################

def run_stereo_vo(seq_root, name, extractor):

    # Load calibration
    P0,P1 = read_kitti_P0P1(seq_root/"calib.txt")
    K = P0[:,:3]

    left,right = load_stereo_images(seq_root)

    # Extract first frame
    L0=cv2.imread(str(left[0]),0)
    R0=cv2.imread(str(right[0]),0)

    for i in range(downsample_iterations):
        L0 = downsample(L0,downsample_factor,downsample_gaussian_sigma, downsample_interpolation_type)
        R0 = downsample(R0,downsample_factor,downsample_gaussian_sigma, downsample_interpolation_type)




    kL0 = extractor.detect(L0)
    kR0 = extractor.detect(R0)

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
    kL0, dL0 = extractor.compute(L0,kL0)
    kR0, dR0 = extractor.compute(R0,kR0)

    for keypoint in kL0:
        keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)
    for keypoint in kR0:
        keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)


    matcher = cv2.BFMatcher(extractor.norm)

    matchesLR0 = matcher.knnMatch(dL0, dR0, 2)
    good = [m for m,n in matchesLR0 if m.distance < LOWE_RATIO*n.distance]

    tri_prev = triangulate_stereo(kL0,kR0,good,P0,P1,EPIPOLAR_TOL)

    poses=[np.eye(4)]  # world_T_cam
    stats = {
        "stereo_matches": [],
        "triangulated": [],
        "temporal_matches": [],
        "pnp_inliers": [],
        "failures": 0
    }

    prevL, prevR = L0, R0
    kLp, dLp = kL0, dL0
    tri_map = tri_prev


    for i in range(1,len(left)):
        L=cv2.imread(str(left[i]),0)
        R=cv2.imread(str(right[i]),0)
        for i in range(downsample_iterations):
            L = downsample(L,downsample_factor,downsample_gaussian_sigma, downsample_interpolation_type)
            R = downsample(R,downsample_factor,downsample_gaussian_sigma, downsample_interpolation_type)

        kpL = extractor.detect(L)
        kpL = sorted(kpL, key=lambda x:x.response, reverse=True)[:MAX_FEATURES]
        kpR = extractor.detect(R)
        kpR = sorted(kpR, key=lambda x:x.response, reverse=True)[:MAX_FEATURES]


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

        kpL, dL = extractor.compute(L,kpL)
        kpR, dR = extractor.compute(R,kpR)

        for keypoint in kpL:
            keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)
        for keypoint in kpR:
            keypoint.pt = (keypoint.pt[0] * downsample_factor ** downsample_iterations, keypoint.pt[1] * downsample_factor ** downsample_iterations)

        # temporal matches
        ml = matcher.knnMatch(dLp, dL, 2)
        good_temporal = [m for m,n in ml if m.distance < LOWE_RATIO*n.distance]

        stats["temporal_matches"].append(len(good_temporal))
        stats["stereo_matches"].append(len(good))

        # Build 3D-2D pairs
        pts3d=[]; pts2d=[]
        for m in good_temporal:
            if m.queryIdx in tri_map:
                pts3d.append(tri_map[m.queryIdx])
                pts2d.append(kpL[m.trainIdx].pt)

        # Solve PnP
        res = solve_pnp(pts3d, pts2d, K, PNP_REPROJ_THRESH)
        if res is None:
            stats["pnp_inliers"].append(0)
            stats["failures"]+=1
            poses.append(poses[-1].copy())
        else:
            R,t,inl = res
            stats["pnp_inliers"].append(len(inl))

            T = build_T(R,t)
            # world_T_cam_next = world_T_cam * inv(T_cam_next_cam)
            poses.append(poses[-1] @ T_inv(T))

        # recompute stereo for next step
        matchesLR = matcher.knnMatch(dL, dR, 2)
        good = [m for m,n in matchesLR if m.distance < LOWE_RATIO*n.distance]
        tri_map = triangulate_stereo(kpL,kpR,good,P0,P1,EPIPOLAR_TOL)
        stats["triangulated"].append(len(tri_map))

        kLp, dLp = kpL, dL

    return poses, stats


#########################################################
# ===================== MAIN RUN ========================
#########################################################

def main():
    results = []

    for seq in SEQUENCES:
        seq_root = Path(DATA_ROOT)/"sequences"/seq
        gt_path = Path(DATA_ROOT)/"poses"/f"{seq}.txt"
        gt_poses = read_gt_poses(gt_path)



        print(f"=== Running sequence {seq} ===")

        for name, extractor in test_combinations.items():
            print(f" -> Testing {name}")

            poses, stats = run_stereo_vo(seq_root, name, extractor)

            traj_path = TRAJ_DIR / f"traj_{seq}_{name.replace('+','-')}.txt"
            save_trajectory_kitti(traj_path, poses)

            
            if gt_poses is None:
                ate = float("nan")
                rpe1_trans, rpe1_rot = float("nan")
                rpe10_trans, rpe10_rot = float("nan")
            else:
                ate = compute_ate(poses, gt_poses)
                
                rpe1_trans, rpe1_rot = compute_rpe(poses, gt_poses, delta=1)
                rpe10_trans, rpe10_rot = compute_rpe(poses, gt_poses, delta=10)




            results.append([
                seq, name,
                ate,
                rpe1_trans,
                rpe1_rot,
                rpe10_trans,
                rpe10_rot,
                np.mean(stats["pnp_inliers"]),
                np.mean(stats["temporal_matches"]),
                np.mean(stats["stereo_matches"]),
                np.mean(stats["triangulated"]),
                stats["failures"]
            ])



    # Save CSV

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
    

        w.writerow([
            "sequence", "method",
            "ATE_RMSE",
            "RPE1_trans_RMSE",
            "RPE1_rot_RMSE",
            "RPE10_trans_RMSE",
            "RPE10_rot_RMSE",
            "PnP_inliers_mean",
            "temporal_matches_mean",
            "stereo_matches_mean",
            "triangulated_mean",
            "failures"
        ])
        w.writerows(results)



    print(f"\nSaved results to {CSV_PATH}")


if __name__ == "__main__":
    main()
