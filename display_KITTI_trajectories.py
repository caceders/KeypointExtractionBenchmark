import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from display_common import _distinct_colors, make_interactive_legend

# ============================================================
# CONFIG
# ============================================================
CSV_PATH  = Path("shared_results/KITTI/FINAL_baseline/results.csv")
TRAJ_DIR  = Path("shared_results/KITTI/FINAL_baseline/trajectories")
DATA_ROOT = Path("./KITTI/data_odometry_gray/dataset")

PLOTS = [
    {
        "sequences": "00",    # None = all available, or e.g. ["00", "05"]
        "alignment": "both",  # "free" | "strict" | "both"
        "select": {
            "method" : "BRISK+BRISK",
            "tag" : "default",
            "matcher" : "MNN",
            "ratio_threshold": 0.5,
            "mnn_bidirectional" : True,
            "ransac_threshold" : 20,
            "epipolar_threshold": 1,
            "downsample_level" : 2,
            "initial_sigma" : 4,
            "max_keypoints" : 250
        },
    },
    # {
    #     "sequences": "00",    # None = all available, or e.g. ["00", "05"]
    #     "alignment": "both",  # "free" | "strict" | "both"
    #     "select": {
    #         "method" : "BRISK+BRISK",
    #         "tag" : "no_scale",
    #         "matcher" : "NN",
    #         "ratio_threshold": 1.0,
    #         "ransac_threshold" : 3,
    #         "epipolar_threshold": 1,
    #         "downsample_level" : 1,
    #         "initial_sigma" : 2,
    #         "max_keypoints" : 1000
    #     },
    # },
]

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one trajectory figure per sequence.
#
# ── Keys ─────────────────────────────────────────────────────────────────────
#   sequences — list of sequence IDs (e.g. ["00", "05"]), a single string "00",
#               or None for all
#   alignment — "free" | "strict" | "both"
#   label     — column(s) to use for the legend label (default: "method")
#               e.g. ["method", "tag"] or ["method", "ransac_threshold"]
#   select    — dict mapping CSV column → value spec:
#                   "method": "ORB+ORB"              single value
#                   "method": ["ORB+ORB", "SIFT"]    list of values
#                   "downsample_level": 0             numeric filter
#
# ── CSV column reference ──────────────────────────────────────────────────────
#   Identity:    method, tag, matcher, ratio_threshold, mnn_bidirectional,
#                ransac_threshold, epipolar_threshold, downsample_level,
#                initial_sigma, max_keypoints
#   Trajectory:  ATE_RMSE_STRICT, ATE_RMSE_ALIGNED,
#                RPE1_trans_RMSE, RPE1_rot_RMSE, ...
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# IO
# ============================================================

def _read_poses_kitti(path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def _positions(poses):
    return np.array([T[:3, 3] for T in poses])


# ============================================================
# ALIGNMENT
# ============================================================

def _align_se3(Pe, Pg):
    muE, muG = Pe.mean(0), Pg.mean(0)
    E0, G0 = Pe - muE, Pg - muG
    U, _, Vt = np.linalg.svd(E0.T @ G0)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = muG - R @ muE
    return R, t


def _strict_align(est_poses, gt_poses):
    T_align = gt_poses[0] @ np.linalg.inv(est_poses[0])
    return [T_align @ T for T in est_poses]


# ============================================================
# HELPERS
# ============================================================

def _apply_select(df, select):
    dfs = df.copy()
    for col, spec in select.items():
        if not isinstance(spec, dict):
            spec = {"values": spec}
        values = spec.get("values")
        if values is None:
            continue
        if col not in dfs.columns:
            print(f"select: column '{col}' not in CSV — skipping.")
            continue
        if isinstance(values, (list, np.ndarray)):
            str_vals = [str(v) for v in values]
            dfs = dfs[dfs[col].isin(values) | dfs[col].astype(str).isin(str_vals)]
        else:
            dfs = dfs[(dfs[col] == values) | (dfs[col].astype(str) == str(values))]
    return dfs


def _num_str(v):
    """Format a number matching the trajectory filename convention (1.0 → '1', 0.25 → '0.25')."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(v)


def _build_traj_name(seq, row):
    ratio = row["ratio_threshold"]
    ratio_str = f"{float(ratio):.2f}" if str(ratio) not in ("-", "nan", "") else "none"
    return (
        f"traj_{seq}_{row['method']}_{row['tag']}_{row['matcher']}"
        f"_{ratio_str}_{_num_str(row['downsample_level'])}"
        f"_r{_num_str(row['ransac_threshold'])}"
        f"_e{_num_str(row['epipolar_threshold'])}.txt"
    )


def _make_label(row, label_cols):
    if isinstance(label_cols, str):
        label_cols = [label_cols]
    parts = [str(row[c]) for c in label_cols if c in row.index]
    return " / ".join(parts)


# ============================================================
# PLOTTING
# ============================================================

def _set_fig_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        try:
            fig.canvas.set_window_title(title)
        except Exception:
            pass


def _plot_trajectories(P_gt, trajs, title, colors):
    fig, ax = plt.subplots(figsize=(10, 7))
    _set_fig_title(fig, title)
    ax.set_title(title)

    gt_line, = ax.plot(P_gt[:, 0], P_gt[:, 2], "k-", lw=2.8, label="GT")
    gt_s = ax.scatter(P_gt[0, 0], P_gt[0, 2], s=140, c="green", edgecolors="black", zorder=5)
    gt_e = ax.scatter(P_gt[-1, 0], P_gt[-1, 2], s=180, c="red", marker="X", edgecolors="black", zorder=5)

    method_artists = {}
    for label, P in trajs.items():
        line, = ax.plot(P[:, 0], P[:, 2], lw=1.8, color=colors.get(label), label=label)
        ms = ax.scatter(P[0, 0], P[0, 2], c="green", s=60, zorder=4)
        me = ax.scatter(P[-1, 0], P[-1, 2], c="red", s=80, zorder=4)
        method_artists[label] = [line, ms, me]

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    leg = ax.legend(fontsize=9, frameon=True, fancybox=True)

    all_artists = {"GT": [gt_line, gt_s, gt_e], **method_artists}
    lined = {
        legline: all_artists[legtext.get_text()]
        for legline, legtext in zip(leg.get_lines(), leg.get_texts())
        if legtext.get_text() in all_artists
    }
    make_interactive_legend(fig, leg, lined)


# ============================================================
# RUN
# ============================================================

def run_traj_display(csv_path, traj_dir, data_root, plots):
    df = pd.read_csv(csv_path, na_values=[], keep_default_na=False)
    df["method"] = df["method"].astype(str).str.strip()

    gt_pose_dir = Path(data_root) / "poses"
    traj_dir    = Path(traj_dir)

    for cfg in plots:
        sequences  = cfg.get("sequences")
        alignment  = cfg.get("alignment", "both")
        select     = cfg.get("select", {})
        label_cols = cfg.get("label", "method")

        dfs = _apply_select(df, select)
        if dfs.empty:
            print("Trajectory plot: no data after filter — skipping.")
            continue

        # Build label → color map from filtered rows
        labels_ordered = []
        seen = set()
        for _, row in dfs.iterrows():
            lbl = _make_label(row, label_cols)
            if lbl not in seen:
                labels_ordered.append(lbl)
                seen.add(lbl)
        colors = dict(zip(labels_ordered, _distinct_colors(len(labels_ordered))))

        if sequences is None:
            seq_filter = None
        elif isinstance(sequences, str):
            seq_filter = {sequences.zfill(2)}
        else:
            seq_filter = {str(s).zfill(2) for s in sequences}

        for gt_file in sorted(gt_pose_dir.glob("*.txt")):
            seq = gt_file.stem

            if seq_filter is not None and seq not in seq_filter:
                continue

            gt_poses_all = _read_poses_kitti(gt_file)
            free_trajs, strict_trajs = {}, {}

            for _, row in dfs.iterrows():
                traj_path = traj_dir / _build_traj_name(seq, row)
                if not traj_path.exists():
                    continue

                label     = _make_label(row, label_cols)
                est_poses = _read_poses_kitti(traj_path)

                try:
                    start, end = map(int, str(row["active_frames"]).split("-"))
                    gt_poses = gt_poses_all[start:end + 1]
                except Exception:
                    gt_poses = gt_poses_all

                n = min(len(est_poses), len(gt_poses))
                Pe = _positions(est_poses[:n])
                Pg = _positions(gt_poses[:n])

                if alignment in ("free", "both"):
                    R, t = _align_se3(Pe, Pg)
                    free_trajs[label] = (R @ Pe.T).T + t

                if alignment in ("strict", "both"):
                    strict = _strict_align(est_poses[:n], gt_poses[:n])
                    strict_trajs[label] = _positions(strict)

            if not free_trajs and not strict_trajs:
                continue

            P_gt = _positions(gt_poses_all)

            if alignment in ("free", "both") and free_trajs:
                _plot_trajectories(P_gt, free_trajs, f"FREE – seq {seq}", colors)

            if alignment in ("strict", "both") and strict_trajs:
                _plot_trajectories(P_gt, strict_trajs, f"STRICT – seq {seq}", colors)

    plt.show()


run_traj_display(CSV_PATH, TRAJ_DIR, DATA_ROOT, PLOTS)
