
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# ---------------------------- CONFIG ----------------------------
CSV_PATH = "output_size_scaling_distance_10_1.csv"

# SORT_MODE:
#   "alphabetical_by_detector"   -> detector, then descriptor name, then descriptor number
#   "alphabetical_by_descriptor" -> descriptor name, then descriptor number, then detector
#   "metric"                     -> per-metric sort; ties respect detector-first order
SORT_MODE = "alphabetical_by_detector"   # "alphabetical_by_detector" | "alphabetical_by_descriptor" | "metric"
METRIC_ASCENDING = False                   # used only when SORT_MODE == "metric"

# Descriptor number ordering when present
NUMBER_DESCENDING = True   # True => 32 above 4; False => 4 above 32
PERCENTAGE = True
# Color/section grouping:
#   "auto":       descriptor when SORT_MODE is alphabetical_by_descriptor, else detector
#   "detector":   always color/section by detector
#   "descriptor": always color/section by descriptor name
SECTION_COLOR_MODE = "auto"
SHADE_SECTIONS = True

# ---------------- Blacklists (REQUIRED as sets) ----------------
# Detector blacklist (case-insensitive match)
BLACKLIST_DETECTORS   = {"MSER", "AGAST", "SIFT 3.2"}         # e.g., {"MSER", "FAST"}
# Descriptor NAME blacklist (case-insensitive; number ignored)
BLACKLIST_DESCRIPTORS = {"DAISY"}        # e.g., {"DAISY", "SIFT"}
# Trailing NUMBER blacklist (numeric match against the parsed number)
BLACKLIST_NUMBERS     = set({64,32, 0.030625})            # e.g., {64, 32.0, 0.06125}

# Optional tolerance when matching BLACKLIST_NUMBERS (None = exact match)
BLACKLIST_NUMBERS_EPSILON = None  # e.g., 1e-9 for float tolerance; None for exact match

# Case-insensitive for detector/descriptor name
BLACKLIST_CASE_INSENSITIVE = True
DEBUG = False  # print how many rows were blacklisted

# Visuals
FIG_DPI = 120
FONT_FAMILY_YTICKS = "monospace"
EXCLUDE_DESCRIPTOR_NUM_FROM_PLOTS = True  # exclude the parsed descriptor number column from metrics
# ---------------------------------------------------------------

# -------- Validate blacklist types (require sets) --------
def _require_set(name, val):
    if not isinstance(val, (set, frozenset)):
        raise TypeError(f"{name} must be a set, e.g., {{'MSER','FAST'}}. Got: {type(val).__name__}")

_require_set("BLACKLIST_DETECTORS", BLACKLIST_DETECTORS)
_require_set("BLACKLIST_DESCRIPTORS", BLACKLIST_DESCRIPTORS)
_require_set("BLACKLIST_NUMBERS", BLACKLIST_NUMBERS)

# -------------------------- Load & Parse -------------------------
df = pd.read_csv(CSV_PATH)

# Ensure combination exists and is clean string
if "combination" not in df.columns:
    df = df.copy()
    df["combination"] = df.index.astype(str)
else:
    df = df.copy()
    df["combination"] = df["combination"].astype(str).str.strip()

# Split at FIRST '+': Detector | DescriptorFull
parts = df["combination"].str.split("+", n=1, expand=True)
if parts.shape[1] < 2:
    raise ValueError("Each 'combination' must contain one '+', e.g. 'AKAZE+AKAZE 0.030625'.")

df["detector"]        = parts[0].fillna("").str.strip()
df["descriptor_full"] = parts[1].fillna("").str.strip()

# Extract descriptor_name and descriptor_num (number at end after whitespace is optional)
# Examples:
#  "AKAZE 0.030625" -> name="AKAZE", num=0.030625
#  "DAISY"          -> name="DAISY", num=NaN
desc_extract = df["descriptor_full"].str.extract(r"^(.*?)(?:\s+([+-]?\d+(?:\.\d+)?))?$")
df["descriptor_name"] = desc_extract[0].fillna("").str.strip()
df["descriptor_num"]  = pd.to_numeric(desc_extract[1], errors="coerce")

# --------------------------- Blacklist ---------------------------
# Detector / descriptor name (case-insensitive by default)
if BLACKLIST_CASE_INSENSITIVE:
    det_mask  = df["detector"].str.lower().isin({s.lower() for s in BLACKLIST_DETECTORS})
    desc_mask = df["descriptor_name"].str.lower().isin({s.lower() for s in BLACKLIST_DESCRIPTORS})
else:
    det_mask  = df["detector"].isin(BLACKLIST_DETECTORS)
    desc_mask = df["descriptor_name"].isin(BLACKLIST_DESCRIPTORS)

# Numbers: exact or epsilon match
if len(BLACKLIST_NUMBERS) == 0:
    num_mask = pd.Series(False, index=df.index)
else:
    # Build mask against numeric column df["descriptor_num"]
    if BLACKLIST_NUMBERS_EPSILON is None:
        # Exact membership (NaN never matches)
        num_mask = df["descriptor_num"].isin(list(BLACKLIST_NUMBERS))
    else:
        eps = float(BLACKLIST_NUMBERS_EPSILON)
        num_mask = pd.Series(False, index=df.index)
        # For each banned number, mark rows where |descriptor_num - banned| <= eps
        for b in BLACKLIST_NUMBERS:
            # Skip non-numeric entries
            try:
                bval = float(b)
            except Exception:
                continue
            num_mask = num_mask | (df["descriptor_num"].notna() & (np.abs(df["descriptor_num"] - bval) <= eps))




# Combine masks and drop
drop_mask = det_mask | desc_mask | num_mask
if DEBUG:
    before = len(df)
    dropped = df.loc[drop_mask, ["combination","detector","descriptor_name","descriptor_num"]]
    print(f"[Blacklist] Dropping {len(dropped)} of {before} rows.")
    if not dropped.empty:
        print(dropped.head(10))

df = df.loc[~drop_mask].reset_index(drop=True)

# ---------------------- Metrics to plot -------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if EXCLUDE_DESCRIPTOR_NUM_FROM_PLOTS and "descriptor_num" in numeric_cols:
    numeric_cols.remove("descriptor_num")

if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found to plot (after applying blacklist and exclusions).")

# ---------------------- Build sort baselines --------------------
# Vectorized keys (temporary columns for stable mergesort)
df_work = df.assign(
    __det_key  = df["detector"].fillna("").str.lower(),
    __desc_key = df["descriptor_name"].fillna("").str.lower(),
    __comb_key = df["combination"].fillna("").str.lower()
)

# Numeric key for descriptor number (desc/asc + sentinel for NaN)
if NUMBER_DESCENDING:
    desc_num_key = np.where(df["descriptor_num"].notna(), -df["descriptor_num"].values, -np.inf)
else:
    desc_num_key = np.where(df["descriptor_num"].notna(), df["descriptor_num"].values, np.inf)
df_work["__desc_num_key"] = desc_num_key

# Detector-first baseline
df_alpha_by_detector = df_work.sort_values(
    by=["__det_key", "__desc_key", "__desc_num_key", "__comb_key"],
    kind="mergesort"
).drop(columns=["__det_key","__desc_key","__desc_num_key","__comb_key"]).reset_index(drop=True)

# Descriptor-first baseline
df_alpha_by_descriptor = df_work.sort_values(
    by=["__desc_key", "__desc_num_key", "__det_key", "__comb_key"],
    kind="mergesort"
).drop(columns=["__det_key","__desc_key","__desc_num_key","__comb_key"]).reset_index(drop=True)

# ---------------------- Color maps (stable) ---------------------
all_detectors   = sorted(df["detector"].dropna().astype(str).unique(), key=str.lower)
all_descriptors = sorted(df["descriptor_name"].dropna().astype(str).unique(), key=str.lower)
cmap_det  = plt.cm.get_cmap("tab20")
cmap_desc = plt.cm.get_cmap("tab20")
det_color_map  = {d: cmap_det(i % cmap_det.N) for i, d in enumerate(all_detectors)}
desc_color_map = {d: cmap_desc(i % cmap_desc.N) for i, d in enumerate(all_descriptors)}

# ---------------------- Plotting ----------------------
smode = SORT_MODE.lower()

for col in numeric_cols:
    # Choose row order
    if smode == "alphabetical_by_detector":
        df_sorted = df_alpha_by_detector.copy()
        sort_label = "alphabetical by DETECTOR (name → descriptor → number)"
    elif smode == "alphabetical_by_descriptor":
        df_sorted = df_alpha_by_descriptor.copy()
        sort_label = "alphabetical by DESCRIPTOR (name → number → detector)"
    elif smode == "metric":
        # Stable metric sort; ties keep detector-first order
        df_sorted = df_alpha_by_detector.sort_values(by=col, ascending=METRIC_ASCENDING, kind="mergesort").reset_index(drop=True)
        sort_label = f"by metric '{col}' ({'asc' if METRIC_ASCENDING else 'desc'})"
    else:
        raise ValueError("SORT_MODE must be 'alphabetical_by_detector', 'alphabetical_by_descriptor', or 'metric'.")

    # Grouping key for colors/sections
    if SECTION_COLOR_MODE == "descriptor":
        group_key = "descriptor_name"; legend_title = "Descriptor"; color_map = desc_color_map
    elif SECTION_COLOR_MODE == "detector":
        group_key = "detector";        legend_title = "Detector";  color_map = det_color_map
    else:  # "auto"
        if smode == "alphabetical_by_descriptor":
            group_key = "descriptor_name"; legend_title = "Descriptor"; color_map = desc_color_map
        else:
            group_key = "detector";        legend_title = "Detector";  color_map = det_color_map

    # Colors per bar (extend map for unseen groups if needed)
    group_vals = df_sorted[group_key].astype(str).tolist()
    if any(g not in color_map for g in group_vals):
        base = cmap_desc if group_key == "descriptor_name" else cmap_det
        existing = set(color_map.keys())
        missing = sorted({g for g in group_vals if g not in existing}, key=str.lower)
        start = len(existing)
        for j, g in enumerate(missing, start=start):
            color_map[g] = base(j % base.N)
    bar_colors = [color_map[g] for g in group_vals]

    # Labels: aligned "Detector + DescriptorFull"
    split_two = [s.split("+", 1) for s in (df_sorted["detector"].astype(str) + "+" + df_sorted["descriptor_full"].astype(str))]
    max_left  = max(len(p[0]) for p in split_two) if split_two else 0
    max_right = max(len(p[1]) for p in split_two) if split_two else 0
    y_labels = [p[0].ljust(max_left) + " + " + p[1].ljust(max_right) for p in split_two]

    y_pos = np.arange(len(df_sorted))
    fig_h = max(6, len(df_sorted) * 0.14)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=FIG_DPI)
    try:
        fig.canvas.manager.set_window_title(str(col))
    except Exception:
        pass

    ax.barh(y_pos, df_sorted[col].values, color=bar_colors)

    # Shaded contiguous runs by chosen grouping key
    if SHADE_SECTIONS and len(df_sorted) > 0:
        seq = df_sorted[group_key].astype(str).tolist()
        runs = []
        start = 0
        curr = seq[0]
        for i, v in enumerate(seq[1:], start=1):
            if v != curr:
                runs.append((start, i, curr))
                start = i
                curr  = v
        runs.append((start, len(seq), curr))

        x_right = ax.get_xlim()[1]
        if x_right == 0:
            x_right = max(float(df_sorted[col].max()), 1.0)

        for i, (rstart, rend, grp) in enumerate(runs):
            ax.axhspan(rstart - 0.5, rend - 0.5, color=color_map[grp], alpha=0.08, lw=0)
            if i > 0:
                ax.axhline(rstart - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            mid = (rstart + rend - 1) / 2.0
            ax.text(x_right * 1.01, mid, grp, va="center", ha="left",
                    fontsize=11, fontweight="bold", color=color_map[grp])

    # ---------------------- Header/X-label formatting (capitalization & units) ----------------------
    # If your column name has a unit like "matching distance [px]",
    #   - Title: show without the unit, capitalize first letter -> "Matching distance"
    #   - X-label (text under): show with the unit, capitalize first letter -> "Matching distance [px]"
    import re as _re
    _m = _re.match(r"^(.*?)(\s*\[[^\]]*\])\s*$", str(col))  # capture name + optional [unit]
    if _m:
        _name = _m.group(1).strip()
        _unit = _m.group(2).strip()
    else:
        _name = str(col).strip()
        _unit = ""

    # Capitalize first letter
    _name_cap = (_name[:1].upper() + _name[1:]) if _name else _name

    # Apply to title (no unit) and xlabel (with unit if present)
    ax.set_title(f"{_name_cap} by combination — grouped & colored by {legend_title}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{_name_cap}{(' ' + _unit) if _unit else ''}")
    # ------------------------------------------------------------------------------------------------

    ax.set_ylabel("combination")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)

    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Legend removed earlier
    plt.tight_layout()

plt.show()
