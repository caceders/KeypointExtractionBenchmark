import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

###########################################
# =============== CONFIG ==================
###########################################

RUN_DIR = Path("KITTI/results/FRAME_TEST_LEFT")
if not RUN_DIR.exists():
    raise Exception("directory wrong")
TRAJ_DIR = RUN_DIR / "trajectories"
DATA_ROOT = Path("./KITTI/data_odometry_gray/dataset")
RESULTS_CSV = RUN_DIR / "results.csv"

FILTER_WHITELIST_MODE = False

SUFFIX_FILTER = [
    # "PNP6",
]

DOWNSAMPLE_FILTER = [
    #0,
]

METHOD_FILTER = [
    # "ORB+ORB"
]

ENABLE_SUFFIX_FILTER = True
ENABLE_DOWNSAMPLE_FILTER = True
ENABLE_METHOD_FILTER = True

###########################################
# ============ METHOD PARSING ==============
###########################################

def parse_method(method_name):
    """
    BASE_SUFFIX_DOWNSAMPLE
    """
    parts = method_name.split("_")
    method = "_".join(parts[0:-2])

    if len(parts) < 3:
        return method, None, None

    suffix = parts[-2]

    try:
        downsample = int(parts[-1])
    except ValueError:
        downsample = None

    return method, suffix, downsample


def passes_filter(method_name):
    method, suffix, downsample = parse_method(method_name)
    checks = []

    if ENABLE_SUFFIX_FILTER:
        checks.append(
            suffix in SUFFIX_FILTER if FILTER_WHITELIST_MODE else suffix not in SUFFIX_FILTER
        )

    if ENABLE_DOWNSAMPLE_FILTER:
        checks.append(
            downsample in DOWNSAMPLE_FILTER if FILTER_WHITELIST_MODE else downsample not in DOWNSAMPLE_FILTER
        )

    if ENABLE_METHOD_FILTER:
        checks.append(
            method in METHOD_FILTER if FILTER_WHITELIST_MODE else method not in METHOD_FILTER
        )

    return all(checks) if checks else True

###########################################
# ============== IO =======================
###########################################

def read_poses_kitti(path: Path):
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.eye(4)
            T[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def positions(poses):
    return np.array([T[:3, 3] for T in poses])

###########################################
# ============ ALIGNMENT ===================
###########################################

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
# ============ RESULTS METADATA ============
###########################################

def load_active_frame_ranges(csv_path):
    """
    Returns:
        dict[(sequence, method)] = (start_frame, end_frame)
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    ranges = {}

    if "active_frames" not in df.columns:
        return {}

    for _, r in df.iterrows():

        start, end = map(int, r["active_frames"].split("-"))
        ranges[(str(r["sequence"]), str(r["method"]))] = (start, end)


    return ranges

###########################################
# ============ PLOTTING ====================
###########################################

def plot_with_legend_toggle(P_gt, trajs, title, colors):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(title)

    # ---- GT ----
    ax.plot(P_gt[:, 0], P_gt[:, 2], "k-", lw=2.8, label="GT")
    ax.scatter(P_gt[0, 0], P_gt[0, 2],
               s=140, c="green", edgecolors="black", zorder=5)
    ax.scatter(P_gt[-1, 0], P_gt[-1, 2],
               s=180, c="red", marker="X", edgecolors="black", zorder=5)

    # ---- Method trajectories ----
    method_artists = {}

    for method, P in trajs.items():
        line, = ax.plot(
            P[:, 0], P[:, 2],
            lw=1.8, color=colors[method], label=method
        )
        start = ax.scatter(P[0, 0], P[0, 2], c="green", s=60, zorder=4)
        end = ax.scatter(P[-1, 0], P[-1, 2], c="red", s=80, zorder=4)

        method_artists[method] = (line, start, end)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # ---- Legend ----
    leg = ax.legend(fontsize=9, frameon=True, fancybox=True)
    legend_items = []

    for leg_line, leg_text in zip(leg.get_lines(), leg.get_texts()):
        method = leg_text.get_text()

        leg_line.set_picker(True)
        leg_line.set_pickradius(8)
        leg_text.set_picker(True)

        legend_items.append((leg_line, leg_text, method))

    # ---- Visibility helpers ----
    def set_method_visibility(method, visible):
        for artist in method_artists[method]:
            artist.set_visible(visible)

        for l, t, m in legend_items:
            if m == method:
                alpha = 1.0 if visible else 0.25
                l.set_alpha(alpha)
                t.set_alpha(alpha)
                t.set_fontweight("bold" if visible else "normal")

    def hide_all_except(method):
        for m in method_artists:
            set_method_visibility(m, m == method)

    # ---- Pick handler ----
    def on_pick(event):
        mouse_button = event.mouseevent.button
        is_double = event.mouseevent.dblclick

        for l, t, method in legend_items:
            if event.artist in (l, t):
                currently_visible = method_artists[method][0].get_visible()

                # Right-click OR double-click → isolate
                if mouse_button == 3 or is_double:
                    hide_all_except(method)
                else:
                    set_method_visibility(method, not currently_visible)

                fig.canvas.draw_idle()
                return

    fig.canvas.mpl_connect("pick_event", on_pick)

###########################################
# ================ MAIN ===================
###########################################

active_ranges = load_active_frame_ranges(RESULTS_CSV)

gt_pose_dir = DATA_ROOT / "poses"

methods = sorted({
    p.name.split("_", 2)[2].replace(".txt", "")
    for p in TRAJ_DIR.glob("traj_*_*.txt")
})

colors = {m: plt.cm.tab20(i / max(1, len(methods)))
          for i, m in enumerate(methods)}

for gt_file in sorted(gt_pose_dir.glob("*.txt")):
    seq = gt_file.stem
    gt_poses_all = read_poses_kitti(gt_file)

    free_trajs = {}
    strict_trajs = {}

    for method in methods:
        if not passes_filter(method):
            continue

        traj_path = TRAJ_DIR / f"traj_{seq}_{method}.txt"
        if not traj_path.exists():
            continue

        est_poses = read_poses_kitti(traj_path)

        # --- active frame lookup ---
        
        start, end = active_ranges.get(
            (str(int(seq)), method)
        )
        gt_poses = gt_poses_all[start:end + 1]
        n = min(len(est_poses), len(gt_poses))

        Pe = positions(est_poses[:n])
        Pg = positions(gt_poses[:n])

        R, t = align_se3(Pe, Pg)
        free_trajs[method] = (R @ Pe.T).T + t

        strict = strict_align_from_start(est_poses[:n], gt_poses[:n])
        strict_trajs[method] = positions(strict)
    if not free_trajs:
        continue

    plot_with_legend_toggle(
        positions(gt_poses),
        free_trajs,
        f"FREE alignment – sequence {seq}",
        colors
    )

    plot_with_legend_toggle(
        positions(gt_poses),
        strict_trajs,
        f"STRICT alignment – sequence {seq}",
        colors
    )

plt.show()