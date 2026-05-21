from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "results/referance.csv"

PLOTS = [

    {
        "y":        "Matching distance mAP",
        "x":        "max_keypoints",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "matcher": "default",
            
        },
    },

        {
        "y":        "Matching distance mAP",
        "x":        "initial_sigma",
        "lines": ["method", "tag"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "matcher": "default",
            
        },
    },

    {
        "y":        "avg_num_keypoints",
        "x":        "max_keypoints",
        "lines": ["method", "tag", "initial_sigma"],
        "subplots": "downsample_level",
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            
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
        "y":        "avg_num_keypoints",
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
#   Identity:    method, tag, downsample_level, max_keypoints,
#                matcher, ratio_threshold, ransac_threshold
#   Counts:      avg_num_keypoints, avg_num_matches
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS)
