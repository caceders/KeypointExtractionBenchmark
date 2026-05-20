from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "KITTI/results/optimize/results.csv"

PLOTS = [

    {
        "y":        "RPE1_trans_RMSE",
        "x":        ["method", "tag"],
        "select": {
        },
    },
    {
        "y":        "ATE_RMSE_STRICT",
        "x":        ["method", "tag"],
        "select": {
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one matplotlib figure.
#
# ── Axes ──────────────────────────────────────────────────────────────────────
#   x        — CSV column for the x-axis
#   y        — CSV column to use as the y-value
#   lines    — CSV column(s) whose unique values become separate lines (omit → bar chart)
#   subplots — CSV column that creates subplot panels (default: None → one panel)
#
# ── Select ────────────────────────────────────────────────────────────────────
#   select — dict mapping column → spec, processed in key order:
#                 {"values": [...]}                  filter to these values only
#                 {"fn": "mean"}                     collapse all values with fn
#                 {"fn": "auc", "range": [lo, hi]}   collapse values in range
#                 {"values": [...], "fn": "mean"}    filter then collapse
#              fn options: "auc" (= mean) | "mean" | "std" | "min" | "max"
#
# ── CSV column reference ──────────────────────────────────────────────────────
#   Identity:    method, tag, sequence, downsample_level, initial_sigma,
#                max_keypoints, matcher, ratio_threshold, ransac_threshold,
#                epipolar_threshold
#   Trajectory:  ATE_RMSE_STRICT, ATE_RMSE_ALIGNED,
#                RPE1_trans_RMSE, RPE1_rot_RMSE, RPE10_trans_RMSE, RPE10_rot_RMSE
#   Counts:      avg_num_keypoints, avg_num_temporal_matches,
#                avg_num_stereo_matches, avg_num_PnP_inliers, failures
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS)
