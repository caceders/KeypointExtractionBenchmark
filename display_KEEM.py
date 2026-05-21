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
#   x        — CSV column(s) for the x-axis (str or list of str)
#   y        — CSV column name (str) OR a lambda df -> Series for derived metrics:
#                  lambda df: df["mMA"] * df["repeatability"]
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
#   Identity:    method, tag, downsample_level, max_keypoints,
#                matcher, ratio_threshold, ransac_threshold
#   Counts:      avg_num_keypoints, avg_num_matches
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS)
