from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "KITTI/results/kitti_ransasc_threshold_check/results.csv"

PLOTS = [

    {
        "y":        "ATE_RMSE_STRICT",
        "x":        ["ransac_threshold"],
        "lines":    ["tag"],
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
        },
    },

        {
        "y":        "RPE1_trans_RMSE",
        "x":        ["ransac_threshold"],
        "lines":    ["tag"],
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one matplotlib figure.
#
# ── Axes ──────────────────────────────────────────────────────────────────────
#   x        — CSV column(s) for the x-axis (str or list of str)
#   y        — CSV column name (str) OR a lambda df -> Series for derived metrics:
#                  lambda df: df["RPE1_trans_RMSE"] + df["RPE1_rot_RMSE"]
#              Set y_label when using a lambda (default label is "derived").
#   y_label  — override the y-axis label (optional)
#   lines    — CSV column(s) whose unique values become separate lines (omit → bar chart)
#   subplots — CSV column that creates subplot panels (default: None → one panel)
#   title    — override the auto-generated figure title (optional)
#
# ── Select ────────────────────────────────────────────────────────────────────
#   select — dict mapping column → spec, processed in key order:
#                 {"values": [...]}                  filter to these values only
#                 {"fn": "mean"}                     collapse all values with fn
#                 {"fn": "auc", "range": [lo, hi]}   AUC over values in range
#                 {"values": [...], "fn": "mean"}    filter then collapse
#              fn options: "auc" (= mean) | "mean" | "std" | "min" | "max"
#
# ── CSV column reference ──────────────────────────────────────────────────────
#   Identity:    method, tag, sequence, downsample_level, initial_sigma,
#                intrinsic_sigma, max_keypoints, matcher, ratio_threshold,
#                ransac_threshold, epipolar_threshold
#   Trajectory:  ATE_RMSE_STRICT, ATE_RMSE_ALIGNED,
#                RPE1_trans_RMSE,  RPE1_rot_RMSE,  RPE1_trans_std,  RPE1_rot_std,
#                RPE10_trans_RMSE, RPE10_rot_RMSE, RPE10_trans_std, RPE10_rot_std,
#                RPE1_trans_max,   RPE1_rot_max,   RPE10_trans_max, RPE10_rot_max
#   Counts:      avg_num_keypoints_detected, avg_num_keypoints,
#                avg_num_temporal_matches, avg_num_stereo_matches,
#                avg_num_triangulated_matches, avg_num_PnP_inliers, failures
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS)
