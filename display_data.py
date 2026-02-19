import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

CSV_PATH = "output_bigger_500_10_1000.csv"
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
BLACKLIST_DETECTORS = {"MSER", "AGAST", "SIFT 3.2"}
BLACKLIST_DESCRIPTORS = {"DAISY"}
BLACKLIST_NUMBERS = set({64, 32, 0.030625})
BLACKLIST_NUMBERS_EPSILON = None
BLACKLIST_CASE_INSENSITIVE = True
DEBUG = False
FIG_DPI = 120
FONT_FAMILY_YTICKS = "monospace"
EXCLUDE_DESCRIPTOR_NUM_FROM_PLOTS = True

def _require_set(name, val):
    if not isinstance(val, (set, frozenset)):
        raise TypeError(f"{name} must be a set")
_require_set("BLACKLIST_DETECTORS", BLACKLIST_DETECTORS)
_require_set("BLACKLIST_DESCRIPTORS", BLACKLIST_DESCRIPTORS)
_require_set("BLACKLIST_NUMBERS", BLACKLIST_NUMBERS)

# ---------- LOAD CSV ----------
df = pd.read_csv(CSV_PATH)
if "combination" not in df.columns:
    df["combination"] = df.index.astype(str)
else:
    df["combination"] = df["combination"].astype(str).str.strip()

parts = df["combination"].str.split("+", n=1, expand=True)
df["detector"] = parts[0].fillna("").str.strip()
df["descriptor_full"] = parts[1].fillna("").str.strip()

desc_extract = df["descriptor_full"].str.extract(r"^(.*?)(?:\s+([+-]?\d+(?:\.\d+)?))?$")
df["descriptor_name"] = desc_extract[0].fillna("").str.strip()
df["descriptor_num"] = pd.to_numeric(desc_extract[1], errors="coerce")

# ---------- BLACKLIST ----------
if BLACKLIST_CASE_INSENSITIVE:
    det_mask = df["detector"].str.lower().isin({d.lower() for d in BLACKLIST_DETECTORS})
    desc_mask = df["descriptor_name"].str.lower().isin({d.lower() for d in BLACKLIST_DESCRIPTORS})
else:
    det_mask = df["detector"].isin(BLACKLIST_DETECTORS)
    desc_mask = df["descriptor_name"].isin(BLACKLIST_DESCRIPTORS)

if len(BLACKLIST_NUMBERS) == 0:
    num_mask = pd.Series(False, index=df.index)
else:
    if BLACKLIST_NUMBERS_EPSILON is None:
        num_mask = df["descriptor_num"].isin(list(BLACKLIST_NUMBERS))
    else:
        eps = float(BLACKLIST_NUMBERS_EPSILON)
        num_mask = pd.Series(False, index=df.index)
        for b in BLACKLIST_NUMBERS:
            try: bval = float(b)
            except: continue
            num_mask |= (df["descriptor_num"].notna() & (np.abs(df["descriptor_num"] - bval) <= eps))

drop_mask = det_mask | desc_mask | num_mask
df = df.loc[~drop_mask].reset_index(drop=True)

# ---------- FIND ILLUM/VIEWPOINT COLUMN PAIRS ----------
illum_cols = [c for c in df.columns if c.lower().endswith("illumination")]
view_cols  = [c for c in df.columns if c.lower().endswith("viewpoint")]

pairs = []
for illum in illum_cols:
    base = illum.rsplit(" ", 1)[0]
    view = base + " viewpoint"
    if view in view_cols:
        pairs.append((base, illum, view))

# Create averaged columns
for base, illum, view in pairs:
    df[base] = (df[illum] + df[view]) / 2
    df[f"{base}__illum"] = df[illum]
    df[f"{base}__view"]  = df[view]

# ---------- NUMERIC COLUMNS (AVERAGED ONLY) ----------
numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if not c.endswith("__illum")
    and not c.endswith("__view")
    and not c.lower().endswith("illumination")
    and not c.lower().endswith("viewpoint")
]

if EXCLUDE_DESCRIPTOR_NUM_FROM_PLOTS and "descriptor_num" in numeric_cols:
    numeric_cols.remove("descriptor_num")

if len(numeric_cols) == 0:
    raise ValueError("No numeric columns to plot.")

# ---------- SORTING BASELINES ----------
df_work = df.assign(
    __det_key=df["detector"].fillna("").str.lower(),
    __desc_key=df["descriptor_name"].fillna("").str.lower(),
    __comb_key=df["combination"].fillna("").str.lower()
)

desc_num_key = np.where(df["descriptor_num"].notna(),
                        -df["descriptor_num"].values if NUMBER_DESCENDING else df["descriptor_num"].values,
                        -np.inf if NUMBER_DESCENDING else np.inf)

df_work["__desc_num_key"] = desc_num_key

df_alpha_by_detector = df_work.sort_values(
    by=["__det_key","__desc_key","__desc_num_key","__comb_key"],
    kind="mergesort"
).drop(columns=["__det_key","__desc_key","__desc_num_key","__comb_key"]).reset_index(drop=True)

df_alpha_by_descriptor = df_work.sort_values(
    by=["__desc_key","__desc_num_key","__det_key","__comb_key"],
    kind="mergesort"
).drop(columns=["__det_key","__desc_key","__desc_num_key","__comb_key"]).reset_index(drop=True)

# ---------- COLORS ----------
all_detectors = sorted(df["detector"].dropna().astype(str).unique(), key=str.lower)
all_descriptors = sorted(df["descriptor_name"].dropna().astype(str).unique(), key=str.lower)
cmap_det = plt.cm.get_cmap("tab20")
cmap_desc = plt.cm.get_cmap("tab20")
det_color_map = {d: cmap_det(i % cmap_det.N) for i, d in enumerate(all_detectors)}
desc_color_map = {d: cmap_desc(i % cmap_desc.N) for i, d in enumerate(all_descriptors)}

# ---------- PLOTTING ----------
smode = SORT_MODE.lower()

for col in numeric_cols:
    if smode == "alphabetical_by_detector":
        df_sorted = df_alpha_by_detector.copy()
        legend_title = "Detector"
        group_key = "detector" if SECTION_COLOR_MODE != "descriptor" else "descriptor_name"
    elif smode == "alphabetical_by_descriptor":
        df_sorted = df_alpha_by_descriptor.copy()
        legend_title = "Descriptor"
        group_key = "descriptor_name" if SECTION_COLOR_MODE != "detector" else "detector"
    elif smode == "metric":
        df_sorted = df_alpha_by_detector.sort_values(by=col, ascending=METRIC_ASCENDING, kind="mergesort").reset_index(drop=True)
        group_key = "detector"
    else:
        raise ValueError("Invalid SORT_MODE")

    # Choose color map
    color_map = det_color_map if group_key == "detector" else desc_color_map

    group_vals = df_sorted[group_key].astype(str).tolist()
    bar_colors = [color_map[g] for g in group_vals]

    # Y labels
    split = [s.split("+",1) for s in (df_sorted["detector"].astype(str)+"+"+df_sorted["descriptor_full"].astype(str))]
    maxL = max(len(p[0]) for p in split)
    maxR = max(len(p[1]) for p in split)
    y_labels = [p[0].ljust(maxL)+" + "+p[1].ljust(maxR) for p in split]

    y_pos = np.arange(len(df_sorted))
    fig_h = max(6, len(df_sorted)*0.14)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=FIG_DPI)
    try: fig.canvas.manager.set_window_title(str(col))
    except: pass

    ax.barh(y_pos, df_sorted[col].values, color=bar_colors)

    # ---- DRAW < AND X ----
    for idx, row in df_sorted.iterrows():
        illum = row.get(f"{col}__illum", None)
        view  = row.get(f"{col}__view", None)

        if illum is not None:
            ax.text(illum, idx, "<", va="center", ha="center",
                    color="black", fontweight="bold")

        if view is not None:
            ax.text(view, idx, "X", va="center", ha="center",
                    color="black", fontweight="bold")

    # ---- SHADE SECTIONS ----
    if SHADE_SECTIONS and len(df_sorted) > 0:
        seq = df_sorted[group_key].astype(str).tolist()
        runs = []
        start = 0
        curr = seq[0]
        for i,v in enumerate(seq[1:], start=1):
            if v != curr:
                runs.append((start,i,curr)); start=i; curr=v
        runs.append((start,len(seq),curr))

        x_right = ax.get_xlim()[1] or float(df_sorted[col].max())
        for i,(r0,r1,g) in enumerate(runs):
            ax.axhspan(r0-0.5, r1-0.5, color=color_map[g], alpha=0.08, lw=0)
            if i>0:
                ax.axhline(r0-0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            mid = (r0+r1-1)/2
            ax.text(x_right*1.01, mid, g, va="center", ha="left",
                    fontsize=11, fontweight="bold", color=color_map[g])

    # ---- TITLE / LABELS ----
    m = re.match(r"^(.*?)(\s*\[[^\]]*\])\s*$", str(col))
    if m:
        name = m.group(1).strip()
        unit = m.group(2).strip()
    else:
        name = str(col).strip()
        unit = ""
    name_cap = name[:1].upper()+name[1:]

    ax.set_title(f"{name_cap} — grouped by {legend_title}", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"{name_cap}{(' '+unit) if unit else ''}")
    ax.set_ylabel("combination")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    # ---- LEGEND ----
    illum_patch = mpatches.Patch(color='none', label='<  illumination')
    view_patch  = mpatches.Patch(color='none', label='X  viewpoint')
    ax.legend(handles=[illum_patch, view_patch], loc="lower right")

    plt.tight_layout()

plt.show()