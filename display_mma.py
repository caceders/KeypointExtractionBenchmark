import numpy as np
from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "mma_results/FINAL_low_threshold.csv"

PLOTS = [


        {
        "y":        "mAP",
        "x":        "max_keypoints",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "matcher": "default",
            
        },
    },

        {
        "y":        "mAP",
        "x":        "initial_sigma",
        "lines": ["method", "tag"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "matcher": "default",
            
        },
    },

        {
        "y":        "rep_mean",
        "x":        "initial_sigma",
        "lines": ["method", "tag"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold": {"values": np.arange(0,11), "fn": "auc"},
            
        },
    },

    {
        "y":        "mma_matches_mean",
        "x":        "max_keypoints",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold": {"values": np.arange(0,11), "fn": "auc"},
            
        },
    },

        {
        "y":        "mma_matches_mean",
        "x":        "initial_sigma",
        "lines": ["method", "tag"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold": {"values": np.arange(0,11), "fn": "auc"},
            
        },
    },

        {
        "y":        "hom_acc",
        "x":        "max_keypoints",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold": {"values": np.arange(0,21), "fn": "auc"},
            
        },
    },

            {
        "y":        "mma_matches_mean",
        "x":        "distance_threshold",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
        },
    },

        {
        "y":        "hom_acc",
        "x":        "distance_threshold",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
        },
    },

        {
        "y":        "hom_acc",
        "x":        "initial_sigma",
        "lines": ["method", "tag"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "distance_threshold": {"values": np.arange(0,21), "fn": "auc"},
            
        },
    },

    {
        "y":        "avg_num_matches",
        "x":        "max_keypoints",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
        },
    },

        {
        "y":        "avg_num_matches",
        "x":        "initial_sigma",
        "lines": ["method", "tag"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
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
