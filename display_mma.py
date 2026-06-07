import numpy as np
from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "shared_results/modern/FINAL_baseline_complete.csv"

UNITS = {
    "Distance error threshold": "px",
    "Gaussian blur": "σ",
}

PLOTS = [
    {
        "y":        "mHA",
        "x":        "Distance error threshold",
        "lines" : ["Method", "Invariance configuration"],
        "subplots" : "Matching algorithm",
        "select": {
            # "Method" : ,
            # "Invariance configuration" : ,
            # "Matching algorithm" : ,
            # "Ratio test threshold" : ,
            # "Ratio test directionality" : ,
            # "RANSAC threshold" : ,
            "Visibility filter" : False,
            # "Downsample level" : ,
            # "Gaussian blur" : ,
            # "Max features" : ,
            "Distance error threshold" : np.arange(0,10),
        },
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one matplotlib figure.
#
# ── Axes ──────────────────────────────────────────────────────────────────────
#   x        — CSV column(s) for the x-axis (str or list of str)
#   y        — CSV column name (str) OR a lambda df -> Series for derived metrics:
#                  lambda df: df["mMA"] * df["Repeatability"]
#                  lambda df: (df["mMA kp ref"] + df["mMA"]) / 2
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
#   Identity:    Method, Invariance configuration, Matching algorithm,
#                Ratio test threshold, Ratio test directionality, RANSAC threshold,
#                Max features, Downsample level, Gaussian blur, Visibility filter
#   Grouping:    Image transformation type, Distance error threshold
#   Metrics:     mMA kp ref, mMA, Repeatability, mHA, mAP
#   Counts:      Average number of features, Avg num matches
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS, units=UNITS)
