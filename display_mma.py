import numpy as np
from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "mma_results/ransac_thresholds.csv"

PLOTS = [

    {
        "y":        "mAP",
        "x":        "ransac_threshold",
        "lines":    "tag",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold" : {"values" : np.arange(0,10), "fn" : "auc"},
            
        },
    },
    {
        "y":        "mMA",
        "x":        "ransac_threshold",
        "lines":    "tag",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold" : {"values" : np.arange(0,10), "fn" : "auc"},
            
        },
    },
    {
        "y":        "homography_accuracy",
        "x":        "ransac_threshold",
        "lines":    ["tag", "max_keypoints"],
        "subplots": "method",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold" : {"values" : np.arange(0,6), "fn" : "auc"},
            "matcher" : "NN",
            "downsample_level" : "0",
            "initial_sigma" : "0",
            "ratio_threshold" : 0.2,
        },
    },
    {
        "y":        "homography_success_rate",
        "x":        "ransac_threshold",
        "lines":    "tag",
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
#                  lambda df: df["mMA"] * df["repeatability"]
#                  lambda df: (df["mMA_kp_ref"] + df["mMA"]) / 2
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
#   Identity:    method, tag, matcher, ratio_threshold, ransac_threshold,
#                max_keypoints, downsample_level, initial_sigma, intrinsic_sigma,
#                apply_progressive_blur, visibility_filter
#   Grouping:    transformation, distance_threshold
#   Metrics:     mMA_kp_ref, mMA, repeatability, homography_accuracy, mAP
#   Counts:      avg_num_keypoints, avg_num_keypoints_detected,
#                num_keypoints_ref_detected, num_keypoints_rel_detected,
#                avg_num_matches, num_keypoints_ref, num_keypoints_rel
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS)
