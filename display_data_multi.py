
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# ====================== CONFIG ======================
CSV_PATH = "output_sigma_2.csv"  # <<-- your file

# --- Toggle between views ---
# False => Task view: Verification / Matching / Retrieval
# True  => Matching-only view: Distance / Average Response / Distinctiveness
USE_MATCHING_ONLY_VIEW = False

# Coloring:
#   "detector"   -> consistent color per detector
#   "descriptor" -> consistent color per descriptor_name
#   "rank"       -> gradient by row index (keeps CSV order)
COLOR_BY = "detector"  # "detector" | "descriptor" | "rank"

# Keep original CSV order (no sorting for display)
KEEP_INPUT_ORDER = True

# Percent labels:
#   If max value ≤ 1.2, treat values as fractions [0..1] and show value*100%;
#   otherwise assume values are already percent-like (0..100).
SHOW_PERCENT_LABELS = True

# Show only first K rows (None = all)
TOP_K = None  # e.g., 60

# Right-side label position (axes coords)
RIGHT_LABEL_X_AXES = 1.02  # move further out if needed

# Optional faint background shading by contiguous runs of group (detector/descriptor)
SHADE_SECTIONS = False

# ---------------- Blacklists (sets) ----------------
# Case-insensitive for names; numbers match trailing descriptor number
BLACKLIST_DETECTORS   = set({"SIFT 3.2"})       # e.g., {"MSER","FAST"}
BLACKLIST_DESCRIPTORS = set({"DAISY", "FREAK"})       # e.g., {"DAISY","SIFT"}
BLACKLIST_NUMBERS     = set()       # e.g., {64, 32.0, 0.06125}
BLACKLIST_NUMBERS_EPSILON = None    # e.g., 1e-9 (None = exact match)

DEBUG = False  # print blacklisted rows

# Visuals
FIG_DPI = 140
FIG_WIDTH = 12
BAR_ALPHA = 0.95
FONT_FAMILY_YTICKS = "monospace"
TEXT_SIZE = 9
LEFT_PAD_FRAC = 0.02
ROW_HEIGHT = 0.18
CMAP_DET  = plt.cm.get_cmap("tab20")
CMAP_DESC = plt.cm.get_cmap("tab20")
CMAP_RANK = plt.cm.get_cmap("turbo")
# ===================================================


# -------------------------- Load & rename --------------------------
df = pd.read_csv(CSV_PATH)

# Explicit rename map (from your header) to clean snake_case
orig_lower = {c.lower().strip(): c for c in df.columns}
rename_map_lower_to_new = {
    "combination": "combination",
    "speed": "speed",
    "repeatability mean": "repeatability_mean",
    "repeatability std": "repeatability_std",
    "total num matches": "total_num_matches",
    "number possible correct matches": "number_possible_correct_matches",
    "total correct matches": "total_correct_matches",
    "ratio correct/total matches": "ratio_correct_over_total_matches",
    "ratio correct/possible correct matches": "ratio_correct_over_possible_correct_matches",
    "size mean": "size_mean",
    "size std": "size_std",
    "size normalized std": "size_normalized_std",
    "size min": "size_min",
    "size max": "size_max",
    "size unique count": "size_unique_count",
    "size correct: avg": "size_correct_avg",
    "size correct/all ratio": "size_correct_over_all_ratio",
    "total num keypoints": "total_num_keypoints",
    "correct matches per sequence: avg": "correct_matches_per_sequence_avg",
    "correct matches per sequence: std": "correct_matches_per_sequence_std",
    "distance correct/all ratio": "distance_correct_over_all_ratio",
    "response correct/all ratio": "response_correct_over_all_ratio",
    "distinctiveness all: avg": "distinctiveness_all_avg",
    "distinctiveness correct: avg": "distinctiveness_correct_avg",
    "distinctiveness correct/all ratio": "distinctiveness_correct_over_all_ratio",
    "match rank: avg": "match_rank_avg",
    "match rank: std": "match_rank_std",
    "match rank correct: avg": "match_rank_correct_avg",
    "match rank correct: std": "match_rank_correct_std",
    "ratio rank >25 / all": "ratio_rank_gt25_over_all",
    "ratio rank >25 correct / rank >25": "ratio_rank_gt25_correct_over_rank_gt25",
    "matching distance map": "matching_distance_map",
    "matching average_response map": "matching_average_response_map",
    "matching distinctiveness map": "matching_distinctiveness_map",
    "verification distance ap": "verification_distance_ap",
    "verification average_response ap": "verification_average_response_ap",
    "verification distinctiveness ap": "verification_distinctiveness_ap",
    "retrieval distance map": "retrieval_distance_map",
    "retrieval average_response map": "retrieval_average_response_map",
    "retrieval distinctiveness map": "retrieval_distinctiveness_map",
    "distance-distinctiveness correlation": "distance_distinctiveness_correlation",
}
rename_actual = {orig_lower[k]: v for k, v in rename_map_lower_to_new.items() if k in orig_lower}
df = df.rename(columns=rename_actual)

if "combination" not in df.columns:
    raise ValueError("CSV must contain a 'combination' column.")

# --------------------- Parse combination into parts ---------------------
comb = df["combination"].astype(str).str.strip()
parts = comb.str.split("+", n=1, expand=True)
if parts.shape[1] < 2:
    raise ValueError("Each 'combination' must contain one '+', e.g. 'AKAZE+AKAZE 0.030625'.")
df["detector"]        = parts[0].fillna("").str.strip()
df["descriptor_full"] = parts[1].fillna("").str.strip()

# Extract descriptor_name and trailing number (optional) from descriptor_full
desc_extract = df["descriptor_full"].str.extract(r"^(.*?)(?:\s+([+-]?\d+(?:\.\d+)?))?$")
df["descriptor_name"] = desc_extract[0].fillna("").str.strip()
df["descriptor_num"]  = pd.to_numeric(desc_extract[1], errors="coerce")

# --------------------------- Blacklist ---------------------------
if BLACKLIST_DETECTORS or BLACKLIST_DESCRIPTORS or BLACKLIST_NUMBERS:
    det_mask  = df["detector"].str.lower().isin({s.lower() for s in BLACKLIST_DETECTORS})
    desc_mask = df["descriptor_name"].str.lower().isin({s.lower() for s in BLACKLIST_DESCRIPTORS})
    if len(BLACKLIST_NUMBERS) == 0:
        num_mask = pd.Series(False, index=df.index)
    else:
        if BLACKLIST_NUMBERS_EPSILON is None:
            num_mask = df["descriptor_num"].isin(list(BLACKLIST_NUMBERS))
        else:
            eps = float(BLACKLIST_NUMBERS_EPSILON)
            num_mask = pd.Series(False, index=df.index)
            for b in BLACKLIST_NUMBERS:
                try:
                    bval = float(b)
                except Exception:
                    continue
                num_mask = num_mask | (df["descriptor_num"].notna() & (np.abs(df["descriptor_num"] - bval) <= eps))
    drop_mask = det_mask | desc_mask | num_mask
    if DEBUG:
        before = len(df)
        dropped = df.loc[drop_mask, ["combination","detector","descriptor_name","descriptor_num"]]
        print(f"[Blacklist] Dropping {len(dropped)} of {before} rows.")
        if not dropped.empty:
            print(dropped.head(10))
    df = df.loc[~drop_mask].reset_index(drop=True)

if len(df) == 0:
    raise ValueError("All rows removed by blacklist; nothing to plot.")

# ---------------------- Color maps (stable) ---------------------
all_detectors   = sorted(df["detector"].dropna().astype(str).unique(), key=str.lower)
all_descriptors = sorted(df["descriptor_name"].dropna().astype(str).unique(), key=str.lower)
det_color_map  = {d: CMAP_DET(i % CMAP_DET.N)   for i, d in enumerate(all_detectors)}
desc_color_map = {d: CMAP_DESC(i % CMAP_DESC.N) for i, d in enumerate(all_descriptors)}

# ---------------------- Label alignment (global) ----------------------
lefts_all  = df["detector"].astype(str).tolist()
rights_all = df["descriptor_full"].astype(str).tolist()
MAX_LEFT   = max((len(s) for s in lefts_all),  default=0)
MAX_RIGHT  = max((len(s) for s in rights_all), default=0)
def aligned_label(det, desc_full):
    return det.ljust(MAX_LEFT) + " + " + desc_full.ljust(MAX_RIGHT)

# ---- Pretty x-label formatter to force 'AP'/'mAP' casing ----
def pretty_metric_label(metric_col: str) -> str:
    s = metric_col.replace("_", " ")
    s = re.sub(r"\bmap\b", "mAP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bap\b", "AP", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmatching\b", "Matching", s, flags=re.IGNORECASE)
    s = re.sub(r"\bverification\b", "Verification", s, flags=re.IGNORECASE)
    s = re.sub(r"\bretrieval\b", "Retrieval", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdistance\b", "Distance", s, flags=re.IGNORECASE)
    s = re.sub(r"\baverage response\b", "Average Response", s, flags=re.IGNORECASE)
    s = re.sub(r"\bdistinctiveness\b", "Distinctiveness", s, flags=re.IGNORECASE)
    return s

# ---------------------- Build the two views ----------------------
PANEL_SPECS_TASK = [
    ("Matching",     "matching_distance_map"),
    ("Verification", "verification_distance_ap"),
    ("Retrieval",    "retrieval_distance_map"),
]

PANEL_SPECS_MATCHING_ONLY = [
    ("Matching (Distance)",        "matching_distance_map"),
    ("Matching (Average Response)","matching_average_response_map"),
    ("Matching (Distinctiveness)", "matching_distinctiveness_map"),
]

# Higher-is-better flags (default True if missing)
HIGHER_IS_BETTER = {
    "verification_distance_ap": True,
    "verification_average_response_ap": True,
    "verification_distinctiveness_ap": True,
    "matching_distance_map": True,
    "matching_average_response_map": True,
    "matching_distinctiveness_map": True,
    "retrieval_distance_map": True,
    "retrieval_average_response_map": True,
    "retrieval_distinctiveness_map": True,
}

PANEL_SPECS = PANEL_SPECS_MATCHING_ONLY if USE_MATCHING_ONLY_VIEW else PANEL_SPECS_TASK

# ----------------------------- PLOTTING -----------------------------
n_rows_total = len(df)
fig_h = max(6, (TOP_K if TOP_K is not None else n_rows_total) * ROW_HEIGHT)
fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(FIG_WIDTH, fig_h), dpi=FIG_DPI, sharey=True)
if len(PANEL_SPECS) == 1:
    axes = [axes]

for ax, (title, metric_col) in zip(axes, PANEL_SPECS):
    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric_col}' not found in CSV (after renaming).")

    # Keep input order (no sorting)
    df_panel = df.copy().reset_index(drop=True)
    if TOP_K is not None:
        df_panel = df_panel.head(int(TOP_K)).copy()

    # Colors
    if COLOR_BY == "descriptor":
        group_key = "descriptor_name"; color_map = desc_color_map
        groups = df_panel[group_key].astype(str).tolist()
        bar_colors = [color_map.get(g, (0.6,0.6,0.6,1)) for g in groups]
    elif COLOR_BY == "detector":
        group_key = "detector";        color_map = det_color_map
        groups = df_panel[group_key].astype(str).tolist()
        bar_colors = [color_map.get(g, (0.6,0.6,0.6,1)) for g in groups]
    else:  # "rank" gradient by row index (input order)
        group_key = None
        n = len(df_panel)
        bar_colors = [CMAP_RANK(1.0 - (i / max(n-1, 1))) for i in range(n)]

    # Labels
    y_labels = [aligned_label(d, f) for d, f in zip(df_panel["detector"].astype(str),
                                                    df_panel["descriptor_full"].astype(str))]

    # Values & percents
    vals = df_panel[metric_col].astype(float).values
    vmax = float(np.nanmax(vals)) if np.any(~np.isnan(vals)) else 1.0
    if SHOW_PERCENT_LABELS:
        perc_vals = (vals * 100.0) if vmax <= 1.2 else vals
    else:
        perc_vals = None

    # Per-metric ranks (best = 1; ties share min rank)
    higher = HIGHER_IS_BETTER.get(metric_col, True)
    ranks = pd.Series(vals).rank(method="min", ascending=not higher).astype(int).values

    y_pos = np.arange(len(df_panel))

    # Bars
    ax.barh(y_pos, vals, color=bar_colors, alpha=BAR_ALPHA)

    # Axis limits (small left pad)
    x_right = max(vmax * 1.10, 1e-9)
    ax.set_xlim(-x_right * LEFT_PAD_FRAC, x_right)

    # Right-side labels: uniform position for all rows
    if SHOW_PERCENT_LABELS:
        for y, p, r in zip(y_pos, perc_vals, ranks):
            ax.text(RIGHT_LABEL_X_AXES, y, f"{p:.0f}%   {r}",
                    transform=ax.get_yaxis_transform(),  # x in axes coords, y in data coords
                    va="center", ha="left", fontsize=TEXT_SIZE, color="black")

    # Optional shaded sections (contiguous runs)
    if SHADE_SECTIONS and group_key is not None and len(df_panel) > 0:
        seq = df_panel[group_key].astype(str).tolist()
        runs = []
        start = 0
        curr = seq[0]
        for i, v in enumerate(seq[1:], start=1):
            if v != curr:
                runs.append((start, i, curr))
                start = i
                curr  = v
        runs.append((start, len(seq), curr))
        for i, (rstart, rend, grp) in enumerate(runs):
            ax.axhspan(rstart - 0.5, rend - 0.5, color=color_map.get(grp, (0.85,0.85,0.85,1)), alpha=0.08, lw=0)
            if i > 0:
                ax.axhline(rstart - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Titles/axes
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(pretty_metric_label(metric_col))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()
