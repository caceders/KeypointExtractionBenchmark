
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------- CONFIG ----------------------------
CSV_PATH = "output_size_scaling.csv"

# SORT_MODE options:
#   "alphabetical_by_detector"   -> natural sort: detector primary, then descriptor, then rest
#   "alphabetical_by_descriptor" -> natural sort: descriptor primary, then detector, then rest
#   "metric"                     -> per-metric sort, ties keep alphabetical_by_detector order
SORT_MODE = "alphabetical_by_detector"   # "alphabetical_by_detector" | "alphabetical_by_descriptor" | "metric"
METRIC_ASCENDING = True  # used only when SORT_MODE == "metric"

# Colors & section shading mode:
#   "auto":       'descriptor' when SORT_MODE is alphabetical_by_descriptor, otherwise 'detector'
#   "detector":   always color/section by detector
#   "descriptor": always color/section by descriptor
SECTION_COLOR_MODE = "auto"
SHADE_SECTIONS = True

# Blacklists: can be a string (e.g., "MSER") or an iterable (e.g., {"MSER", "FAST"})
BLACKLIST_DETECTORS = "MSER"       # examples: "MSER"  or {"MSER", "FAST"}
BLACKLIST_DESCRIPTORS = "DAISY"    # examples: "DAISY" or {"DAISY", "SIFT"}
BLACKLIST_CASE_INSENSITIVE = True
DEBUG = True  # print how many rows are filtered by blacklist
# Visuals
FIG_DPI = 120
FONT_FAMILY_YTICKS = "monospace"
# ---------------------------------------------------------------

# ---- small helpers (used >1 time) ----
def _as_token_set(x):
    """Normalize config blacklist to a set of full tokens (not characters)."""
    if x is None:
        return set()
    if isinstance(x, str):
        return {x}
    return set(x)

def _token_key(s: str):
    """Numeric-first compare: numbers sort numerically, else case-insensitive strings."""
    s = "" if s is None else str(s).strip()
    try:
        return (0, float(s))
    except ValueError:
        return (1, s.lower())

def _natural_combo_key(comb: str):
    """Natural key across all '+' tokens for deterministic tie-breaking."""
    parts = str(comb).split("+")
    return tuple(_token_key(p) for p in parts)
# ---------------------------------------

# Load data
df = pd.read_csv(CSV_PATH)

# Ensure 'combination' exists and is str
if "combination" not in df.columns:
    df = df.copy()
    df["combination"] = df.index.astype(str)
else:
    df = df.copy()
    df["combination"] = df["combination"].astype(str)

# Parse detector (token 0) and descriptor (token 1)
tokens = df["combination"].str.split("+", n=2, expand=True)
df["detector"]   = tokens[0].fillna("").str.strip()
df["descriptor"] = tokens[1].fillna("").str.strip()

# ---- blacklist filtering (robust to strings vs iterables) ----
bl_det = _as_token_set(BLACKLIST_DETECTORS)
bl_desc = _as_token_set(BLACKLIST_DESCRIPTORS)

if BLACKLIST_CASE_INSENSITIVE:
    bl_det_norm = {s.lower().strip() for s in bl_det}
    bl_desc_norm = {s.lower().strip() for s in bl_desc}
    mask_keep = (~df["detector"].str.lower().isin(bl_det_norm)) & (~df["descriptor"].str.lower().isin(bl_desc_norm))
else:
    bl_det_norm = bl_det
    bl_desc_norm = bl_desc
    mask_keep = (~df["detector"].isin(bl_det_norm)) & (~df["descriptor"].isin(bl_desc_norm))

if DEBUG:
    before = len(df)
    drop_mask = ~mask_keep
    dropped = df.loc[drop_mask, ["combination", "detector", "descriptor"]]
    print(f"[Blacklist] Dropping {len(dropped)} of {before} rows.")
    if len(dropped) > 0:
        print(dropped.head(min(5, len(dropped))))

df = df.loc[mask_keep].reset_index(drop=True)

# Numeric columns to plot
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found to plot.")

# ---- build global alphabetical baselines ----
# A) detector-first baseline
order_alpha_by_detector = sorted(
    range(len(df)),
    key=lambda i: (
        _token_key(df.iloc[i]["detector"]),
        _token_key(df.iloc[i]["descriptor"]),
        _natural_combo_key(df.iloc[i]["combination"])
    )
)
df_alpha_by_detector = df.iloc[order_alpha_by_detector].reset_index(drop=True)

# B) descriptor-first baseline
order_alpha_by_descriptor = sorted(
    range(len(df)),
    key=lambda i: (
        _token_key(df.iloc[i]["descriptor"]),
        _token_key(df.iloc[i]["detector"]),
        _natural_combo_key(df.iloc[i]["combination"])
    )
)
df_alpha_by_descriptor = df.iloc[order_alpha_by_descriptor].reset_index(drop=True)

# ---- build stable color maps for both grouping options (detector & descriptor) ----
all_detectors = sorted(df["detector"].dropna().astype(str).unique().tolist(), key=str.lower)
all_descriptors = sorted(df["descriptor"].dropna().astype(str).unique().tolist(), key=str.lower)
cmap_det = plt.cm.get_cmap("tab20")
cmap_desc = plt.cm.get_cmap("tab20")  # can use tab20b/c if you want a different palette
det_color_map = {d: cmap_det(i % cmap_det.N) for i, d in enumerate(all_detectors)}
desc_color_map = {d: cmap_desc(i % cmap_desc.N) for i, d in enumerate(all_descriptors)}

# ---- plotting ----
smode = SORT_MODE.lower()

for col in numeric_cols:
    # Choose order
    if smode == "alphabetical_by_detector":
        df_sorted = df_alpha_by_detector.copy()
        sort_label = "alphabetical by DETECTOR (natural)"
    elif smode == "alphabetical_by_descriptor":
        df_sorted = df_alpha_by_descriptor.copy()
        sort_label = "alphabetical by DESCRIPTOR (natural)"
    elif smode == "metric":
        # Stable: ties respect detector-first alphabetical order
        df_sorted = df_alpha_by_detector.sort_values(by=col, ascending=METRIC_ASCENDING, kind="mergesort").reset_index(drop=True)
        sort_label = f"by metric '{col}' ({'asc' if METRIC_ASCENDING else 'desc'})"
    else:
        raise ValueError("SORT_MODE must be 'alphabetical_by_detector', 'alphabetical_by_descriptor', or 'metric'.")

    # Decide grouping key *inside* the loop to avoid stale state
    if SECTION_COLOR_MODE == "descriptor":
        group_key = "descriptor"
    elif SECTION_COLOR_MODE == "detector":
        group_key = "detector"
    else:  # "auto"
        group_key = "descriptor" if smode == "alphabetical_by_descriptor" else "detector"

    # Pick the appropriate color map
    if group_key == "detector":
        color_map = det_color_map
        legend_title = "Detector"
    else:
        color_map = desc_color_map
        legend_title = "Descriptor"

    # Compute bar colors from the current df_sorted and group_key
    group_vals = df_sorted[group_key].astype(str).tolist()
    # If a value isn't in the color_map (e.g., rare case after filtering), add it on the fly
    if any(g not in color_map for g in group_vals):
        # extend with new colors deterministically
        existing = set(color_map.keys())
        missing = [g for g in group_vals if g not in existing]
        base_map = cmap_det if group_key == "detector" else cmap_desc
        start_idx = len(existing)
        for j, g in enumerate(sorted(set(missing), key=str.lower), start=start_idx):
            color_map[g] = base_map(j % base_map.N)

    bar_colors = [color_map[g] for g in group_vals]

    # Y-axis labels aligned on first '+', detector-first text
    raw = df_sorted["combination"].astype(str).tolist()
    split_labels = [lbl.split("+", 1) for lbl in raw]
    left_parts  = [p[0] for p in split_labels]
    right_parts = [p[1] if len(p) > 1 else "" for p in split_labels]
    max_left  = max(len(s) for s in left_parts) if left_parts else 0
    max_right = max(len(s) for s in right_parts) if right_parts else 0
    y_labels = [
        left.ljust(max_left) + " + " + right.ljust(max_right)
        for left, right in zip(left_parts, right_parts)
    ]

    y_pos = np.arange(len(df_sorted))
    fig_h = max(6, len(df_sorted) * 0.14)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=FIG_DPI)
    try:
        fig.canvas.manager.set_window_title(str(col))
    except Exception:
        pass

    ax.barh(y_pos, df_sorted[col].values, color=bar_colors)

    # Shaded sections follow the same grouping key as colors
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

    ax.set_title(f"{col} by combination — grouped & colored by {legend_title}\nSort: {sort_label}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("combination")
    ax.set_xlabel(str(col))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)

    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Legend based on groups visible in this plot (keeps legend clean)
    visible_groups = [g for g in sorted(set(group_vals), key=str.lower)]
    legend_handles = [mpatches.Patch(color=color_map[g], label=g) for g in visible_groups]
    ax.legend(handles=legend_handles, title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()

plt.show()
