import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import json

# ---------------------------- CONFIG ----------------------------
CSV_PATH = "big_downsampling_test.csv"
# SORT_MODE:
#   "alphabetical_by_detector"   -> detector, then descriptor name, then descriptor number
#   "alphabetical_by_descriptor" -> descriptor name, then descriptor number, then detector
#   "metric"                     -> per-metric sort; ties respect detector-first order
SORT_MODE = "alphabetical_by_detector"
METRIC_ASCENDING = False
NUMBER_DESCENDING = True
# Color/section grouping:
#   "auto":       descriptor when SORT_MODE is alphabetical_by_descriptor, else detector 
#   "detector":   always color/section by detector
#   "descriptor": always color/section by descriptor name
SECTION_COLOR_MODE = "auto"
SHADE_SECTIONS = True
BLACKLIST_DETECTORS   = {"MSER", "AGAST", "SIFT 3.2"}
BLACKLIST_DESCRIPTORS = {"DAISY"}
BLACKLIST_NUMBERS     = {64, 32, 0.030625}
BLACKLIST_NUMBERS_EPSILON = None
BLACKLIST_CASE_INSENSITIVE = True 
EXCLUDE_DESCRIPTOR_NUM_FROM_PLOTS = True
FIG_DPI = 120
FONT_FAMILY_YTICKS = "monospace"
 
# ----------------------------------------------------------------
# Helper to set figure window title robustly
def set_fig_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except:
        try:
            fig.canvas.set_window_title(title)
        except:
            pass

# ----------------------------------------------------------------
# Load CSV
df = pd.read_csv(CSV_PATH)
df["combination"] = df.get("combination", df.index).astype(str).str.strip()

# Split combination
parts = df["combination"].str.split("+", n=1, expand=True)
df["detector"] = parts[0].str.strip()
df["descriptor_full"] = parts[1].str.strip()

# Extract descriptor name + number
desc_ex = df["descriptor_full"].str.extract(r"^(.*?)(?:\s+(\d+(?:\.\d+)?))?$")
df["descriptor_name"] = desc_ex[0].str.strip()
df["descriptor_num"]  = pd.to_numeric(desc_ex[1], errors="coerce")

# ----------------------------------------------------------------
# Blacklist
if BLACKLIST_DESCRIPTORS:
    df = df.loc[~df["descriptor_name"].str.lower().isin({d.lower() for d in BLACKLIST_DESCRIPTORS})]
if BLACKLIST_DETECTORS:
    df = df.loc[~df["detector"].str.lower().isin({d.lower() for d in BLACKLIST_DETECTORS})]
if BLACKLIST_NUMBERS:
    if BLACKLIST_NUMBERS_EPSILON is None:
        df = df.loc[~df["descriptor_num"].isin(list(BLACKLIST_NUMBERS))]
    else:
        eps = float(BLACKLIST_NUMBERS_EPSILON)
        bad = []
        for b in BLACKLIST_NUMBERS:
            try: bval = float(b)
            except: continue
            bad.append(np.abs(df["descriptor_num"] - bval) <= eps)
        if bad:
            mask = bad[0]
            for m in bad[1:]: mask |= m
            df = df.loc[~mask]

df = df.reset_index(drop=True)

# ----------------------------------------------------------------
# Detect illumination/viewpoint metric pairs
illum_cols=[c for c in df.columns if c.lower().endswith("illumination")]
view_cols =[c for c in df.columns if c.lower().endswith("viewpoint")]

pairs=[]
for illum in illum_cols:
    base = illum.rsplit(" ",1)[0]
    view = base + " viewpoint"
    if view in view_cols:
        pairs.append((base, illum, view))

# Average illumination/viewpoint into a new metric
for base, illum, view in pairs:
    df[base] = (df[illum] + df[view]) / 2
    df[f"{base}__illum"] = df[illum]
    df[f"{base}__view"]  = df[view]

# ----------------------------------------------------------------
# EXCLUDE octave_stats from main plots
def is_octave_col(c):
    return "octave" in c.lower()

numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if not c.endswith("__illum")
    and not c.endswith("__view")
    and not c.lower().endswith("illumination")
    and not c.lower().endswith("viewpoint")
    and not is_octave_col(c)
]

if EXCLUDE_DESCRIPTOR_NUM_FROM_PLOTS and "descriptor_num" in numeric_cols:
    numeric_cols.remove("descriptor_num")

# ----------------------------------------------------------------
# ORDERING LOGIC (unchanged)
dfw=df.assign(
    __det_key=df["detector"].str.lower(),
    __desc_key=df["descriptor_name"].str.lower(),
    __comb_key=df["combination"].str.lower()
)

desc_num_key=np.where(
    df["descriptor_num"].notna(),
    -df["descriptor_num"] if NUMBER_DESCENDING else df["descriptor_num"],
    -np.inf if NUMBER_DESCENDING else np.inf
)
dfw["__desc_num_key"]=desc_num_key


SORT_HELPERS = ["__det_key", "__desc_key", "__desc_num_key", "__comb_key"]

df_by_det = dfw.sort_values(
    by=SORT_HELPERS,
    kind="mergesort"
).drop(columns=SORT_HELPERS).reset_index(drop=True)

df_by_desc = dfw.sort_values(
    by=["__desc_key", "__desc_num_key", "__det_key", "__comb_key"],
    kind="mergesort"
).drop(columns=SORT_HELPERS).reset_index(drop=True)


# ----------------------------------------------------------------
# COLOR MAPS
all_det=sorted(df["detector"].unique(), key=str.lower)
all_des=sorted(df["descriptor_name"].unique(), key=str.lower)
cmap_det=plt.cm.get_cmap("tab20")
cmap_desc=plt.cm.get_cmap("tab20")
det_color={d:cmap_det(i%20) for i,d in enumerate(all_det)}
des_color={d:cmap_desc(i%20) for i,d in enumerate(all_des)}

# ----------------------------------------------------------------
# MAIN METRIC PLOTS
for col in numeric_cols:

    if SORT_MODE=="alphabetical_by_detector":
        df_sorted=df_by_det.copy()
        group_key="detector" if SECTION_COLOR_MODE!="descriptor" else "descriptor_name"
    elif SORT_MODE=="alphabetical_by_descriptor":
        df_sorted=df_by_desc.copy()
        group_key="descriptor_name" if SECTION_COLOR_MODE!="detector" else "detector"
    else:  # metric
        df_sorted=df_by_det.sort_values(by=col, ascending=METRIC_ASCENDING, kind="mergesort")
        group_key="detector"

    color_map = det_color if group_key=="detector" else des_color
    bar_colors=[color_map[g] for g in df_sorted[group_key]]

    # Labels
    split=[s.split("+",1) for s in (df_sorted["detector"]+"+"+df_sorted["descriptor_full"])]
    maxL=max(len(p[0]) for p in split); maxR=max(len(p[1]) for p in split)
    y_labels=[p[0].ljust(maxL)+" + "+p[1].ljust(maxR) for p in split]
    y_pos=np.arange(len(df_sorted))

    fig_h=max(6, len(df_sorted)*0.14)
    fig,ax=plt.subplots(figsize=(10,fig_h),dpi=FIG_DPI)
    set_fig_title(fig, col)

    ax.barh(y_pos, df_sorted[col], color=bar_colors)

    # < and X markers
    for i,row in df_sorted.iterrows():
        illum=row.get(f"{col}__illum")
        view =row.get(f"{col}__view")
        if illum is not None:
            ax.text(illum, i, "<", ha="center", va="center", fontweight="bold")
        if view is not None:
            ax.text(view, i, "X", ha="center", va="center", fontweight="bold")

    # Shading (optional)
    if SHADE_SECTIONS:
        seq=df_sorted[group_key].tolist()
        runs=[]; start=0; curr=seq[0]
        for j,v in enumerate(seq[1:],1):
            if v!=curr: runs.append((start,j,curr)); start=j; curr=v
        runs.append((start,len(seq),curr))
        xright=ax.get_xlim()[1] or float(df_sorted[col].max())
        for k,(a,b,g) in enumerate(runs):
            ax.axhspan(a-0.5, b-0.5, color=color_map[g], alpha=0.08, lw=0)
            if k>0: ax.axhline(a-0.5, color="gray", ls="--", alpha=0.7)
            mid=(a+b-1)/2
            ax.text(xright*1.01, mid, g, ha="left", va="center",
                    fontweight="bold", color=color_map[g])

    ax.set_title(col, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color='none',label='< illumination'),
        mpatches.Patch(color='none',label='X viewpoint')
    ], loc="lower right")

    plt.tight_layout()

# ===================================================================
#              OCTAVE PLOTTING (JSON-BASED)
# ===================================================================

# ===================================================================
#             OCTAVE PLOTTING — AUTODETECT ALL METRICS
# ===================================================================

import json

# Load JSON dictionary
df["octave_stats"] = df["octave_stats"].apply(json.loads)

# ---------------------------------------------------------------
# Collect all metric names that appear in any octave_stats
all_octave_metrics = set()
for stats in df["octave_stats"]:
    for oc, m in stats.items():
        for k in m:
            all_octave_metrics.add(k)

all_octave_metrics = sorted(all_octave_metrics)  # e.g. ["keypoints", "response", "min_response", ...]

# ---------------------------------------------------------------
# Helper to expand df rows for a given metric
def explode_octave_metric(metric_name):
    """
    Build a long-format DF for this metric.
    Columns:
        combination, detector, descriptor_full, octave_idx, value
    """
    rows = []
    for _, row in df.iterrows():
        stats = row["octave_stats"]
        for oc, metrics in stats.items():
            if metric_name not in metrics:
                continue
            val = metrics[metric_name]
            if val is None:
                continue
            rows.append({
                "combination": row["combination"],
                "detector": row["detector"],
                "descriptor_full": row["descriptor_full"],
                "octave_idx": int(oc),
                "value": float(val)
            })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# Order mapping for consistent method order
order_map = {name: i for i, name in enumerate(df_by_det["combination"])}

# ---------------------------------------------------------------
# Produce one plot per octave metric
for metric_name in all_octave_metrics:

    df_exp = explode_octave_metric(metric_name)
    if df_exp.empty:
        continue

    # Sort rows first by method order, then by octave index
    df_exp["_order"] = df_exp["combination"].map(order_map)
    df_exp = df_exp.dropna(subset=["_order"])
    df_exp = df_exp.sort_values(["_order", "octave_idx"]).reset_index(drop=True)

    # Build labels
    y_labels = [
        f"{r['detector']} + {r['descriptor_full']} — octave {r['octave_idx']}"
        for _, r in df_exp.iterrows()
    ]
    y_pos = np.arange(len(df_exp))

    # Colors by octave
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(oc % 10) for oc in df_exp["octave_idx"]]

    fig_h = max(6, len(df_exp) * 0.14)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=FIG_DPI)
    set_fig_title(fig, f"{metric_name} per octave")

    # Bar plot
    ax.barh(y_pos, df_exp["value"], color=colors)

    ax.set_title(f"{metric_name} per octave", fontsize=12, fontweight="bold")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Combination")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Legend by octave index
    legend_handles = [
        mpatches.Patch(color=cmap(o % 10), label=f"octave {o}")
        for o in sorted(df_exp["octave_idx"].unique())
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()

# ----------------------------------------------------------------
plt.show()