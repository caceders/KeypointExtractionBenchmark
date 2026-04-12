import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

###########################################
# =============== CONFIG ==================
###########################################

RUN_DIR = Path("KITTI/results/test_4_1.2")
DATA_ROOT = Path("./KITTI/data_odometry_gray/dataset")

###########################################


def read_poses_kitti(path: Path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def positions(poses):
    P = np.zeros((len(poses), 3))
    for i, T in enumerate(poses):
        P[i] = T[:3, 3]
    return P


def align_se3(Pe, Pg):
    muE, muG = Pe.mean(0), Pg.mean(0)
    E0, G0 = Pe - muE, Pg - muG

    U, _, Vt = np.linalg.svd(E0.T @ G0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = muG - R @ muE
    return R, t


def strict_align_from_start(est_poses, gt_poses):
    T_align = gt_poses[0] @ np.linalg.inv(est_poses[0])
    return [T_align @ T for T in est_poses]


###########################################
# ============== LOAD RESULTS =============
###########################################

csv_path = RUN_DIR / "results.csv"
traj_dir = RUN_DIR / "trajectories"

if not csv_path.exists():
    raise FileNotFoundError(csv_path)

df = pd.read_csv(csv_path)


sequences = sorted(df["sequence"].unique())
methods = sorted(df["method"].unique())
multi_seq = len(sequences) > 1

###########################################
# ============ TRAJECTORY PLOTS ===========
###########################################

for seq in sequences:
    seq_str = f"{int(seq):02d}"
    gt_path = DATA_ROOT / "poses" / f"{seq_str}.txt"
    if not gt_path.exists():
        continue

    gt_poses = read_poses_kitti(gt_path)
    P_gt = positions(gt_poses)

    # -------- FREE ALIGNMENT --------
    plt.figure(figsize=(10, 7))
    plt.title(f"FREE alignment – sequence {seq_str}")
    plt.plot(P_gt[:, 0], P_gt[:, 2], "k-", lw=2.5, label="GT")
    plt.scatter(
        P_gt[0, 0], P_gt[0, 2],
        c="green", s=140, marker="o",
        edgecolors="black", zorder=6,
        label="GT start"
    )

    plt.scatter(
        P_gt[-1, 0], P_gt[-1, 2],
        c="red", s=180, marker="X",
        edgecolors="black", zorder=6,
        label="GT end"
)


    for method in methods:
        traj_path = traj_dir / f"traj_{seq_str}_{method.replace('+','-')}.txt"
        if not traj_path.exists():
            continue

        est_poses = read_poses_kitti(traj_path)
        n = min(len(est_poses), len(gt_poses))

        Pe = positions(est_poses[:n])
        Pg = P_gt[:n]

        R, t = align_se3(Pe, Pg)
        Pe_free = (R @ Pe.T).T + t

        plt.plot(Pe_free[:, 0], Pe_free[:, 2], lw=1.4, label=method)
        plt.scatter(Pe_free[0, 0], Pe_free[0, 2], c="green", s=60)
        plt.scatter(Pe_free[-1, 0], Pe_free[-1, 2], c="red", s=80)

    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show(block=False)

    # -------- STRICT ALIGNMENT --------
    plt.figure(figsize=(10, 7))
    plt.title(f"STRICT alignment – sequence {seq_str}")
    plt.plot(P_gt[:, 0], P_gt[:, 2], "k-", lw=2.5, label="GT")
    plt.scatter(
        P_gt[0, 0], P_gt[0, 2],
        c="green", s=140, marker="o",
        edgecolors="black", zorder=6,
        label="GT start"
    )

    plt.scatter(
        P_gt[-1, 0], P_gt[-1, 2],
        c="red", s=180, marker="X",
        edgecolors="black", zorder=6,
        label="GT end"
    )


    for method in methods:
        traj_path = traj_dir / f"traj_{seq_str}_{method.replace('+','-')}.txt"
        if not traj_path.exists():
            continue

        est_poses = read_poses_kitti(traj_path)
        n = min(len(est_poses), len(gt_poses))

        strict_poses = strict_align_from_start(est_poses[:n], gt_poses[:n])
        Pe_strict = positions(strict_poses)

        plt.plot(Pe_strict[:, 0], Pe_strict[:, 2], lw=1.4, label=method)
        plt.scatter(Pe_strict[0, 0], Pe_strict[0, 2], c="green", s=60)
        plt.scatter(Pe_strict[-1, 0], Pe_strict[-1, 2], c="red", s=80)

    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show(block=False)

###########################################
# ============ METRIC PLOTS ===============
###########################################

# Automatically detect numeric metrics
metric_cols = [
    c for c in df.columns
    if c not in ("sequence", "method")
    and np.issubdtype(df[c].dtype, np.number)
]

for metric in metric_cols:
    # ---- averaged over sequences ----
    plt.figure(figsize=(10, 5))
    means = df.groupby("method")[metric].mean()
    plt.bar(means.index, means.values)
    plt.title(f"{metric} (averaged over sequences)" if multi_seq else metric)
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show(block=False)

    # ---- big per-sequence plot ----
    if multi_seq:
        plt.figure(figsize=(12, 5))
        width = 0.8 / len(sequences)
        x = np.arange(len(methods))

        for i, seq in enumerate(sequences):
            vals = []
            for m in methods:
                row = df[(df["sequence"] == seq) & (df["method"] == m)]
                vals.append(row[metric].values[0] if len(row) else np.nan)

            plt.bar(x + i * width, vals, width, label=f"seq {seq}")

        plt.xticks(x + width * (len(sequences) - 1) / 2, methods, rotation=45)
        plt.title(f"{metric} (per sequence)")
        plt.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.show(block=False)

###########################################
# ============== SHOW ALL ================
###########################################

plt.show()