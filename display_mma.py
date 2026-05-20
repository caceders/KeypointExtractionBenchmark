import numpy as np
from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "mma_results/optimize.csv"

PLOTS = [

    {
        "y":        "mAP_tot",
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
#   Identity:    method, tag, matcher, ratio_threshold, ransac_threshold,
#                max_keypoints, downsample_level, initial_sigma, intrinsic_sigma,
#                apply_progressive_blur, visibility_filter
#   Sequence:    seq_name, seq_id, seq_type
#   Image pair:  img_idx, difficulty, distance_threshold
#   Metrics:     mma_kps, mma_matches, rep, hom_acc, mAP, mAP_tot
#   Counts:      num_keypoints_ref, num_keypoints_rel, num_matches
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS)
