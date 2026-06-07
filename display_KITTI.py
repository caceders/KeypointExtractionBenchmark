from display_common import run_display

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "shared_results/KITTI/FINAL_baseline_complete/results.csv"

UNITS = {
    "Gaussian blur": "σ",
    "RANSAC threshold": "px",
    "ATE": "m",
    "RPE - translational": "m",
    "RPE - rotational": "°",
}


PLOTS = [
    {
        "y":        "RPE - translational",
        "x":        "Method",
        #"lines" : "Method",
        #"subplots" : "Downsample level",
        "select": {
            "Method" : ["SHIFT", "SIFT", "ORB", "BRISK", "AKAZE"],
            "Invariance configuration" : "Both",
            "Matching algorithm" : "MNN",
            "Ratio test threshold" : 1,
            "Ratio test directionality" : "Unidirectional",
            "RANSAC threshold" : 3,
            "Downsample level" : 0,
            "Gaussian blur" : 0,
            "Max features" : 1000,
            "Epipolar threshold" : 1,
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one matplotlib figure.
#
# ── Axes ──────────────────────────────────────────────────────────────────────
#   x        — CSV column(s) for the x-axis (str or list of str)
#   y        — CSV column name (str) OR a lambda df -> Series for derived metrics:
#                  lambda df: df["RPE - translational"] + df["RPE - rotational"]
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
#   Identity:    Method, Invariance configuration, Active frames, Matching algorithm,
#                Ratio test threshold, Ratio test directionality, RANSAC threshold,
#                Epipolar threshold, Downsample level, Gaussian blur, Max features
#   Trajectory:  ATE, ATE RMSE ALIGNED,
#                RPE - translational, RPE - rotational,
#                RPE1 trans std, RPE1 rot std,
#                RPE10 trans RMSE, RPE10 rot RMSE, RPE10 trans std, RPE10 rot std,
#                RPE1 trans max, RPE1 rot max, RPE10 trans max, RPE10 rot max
#   Counts:      Average number of features, Avg num temporal matches,
#                Avg num stereo matches, Avg num triangulated matches,
#                Avg num PnP inliers, Failures
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================
# RUN
# ============================================================
run_display(CSV_PATH, PLOTS, units=UNITS)
