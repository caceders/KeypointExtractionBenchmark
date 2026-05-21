import numpy as np
from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "mma_results/optimize.csv"

PLOTS = [

    {
        "title" : "mAP * mma_matches * hom_acc",
        "y_label" : "mAP * mma_matches * hom_acc",
        "y":        lambda df: df["mAP"] * df["mma_matches"] * df["hom_acc"],
        "x":        ["method", "tag"],
        "select": {
        },
    },
    {
        "y":        "hom_acc",
        "x":        ["method", "tag"],
        "select": {
        },
    },
    {
        "y":        "mma_matches",
        "x":        ["method", "tag"],
        "select": {
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
