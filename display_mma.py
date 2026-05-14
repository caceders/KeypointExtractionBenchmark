import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
CSV_PATH         = "results/mma_results.csv"
PIXEL_THRESHOLDS = list(range(1, 11))
HOM_THRESHOLDS   = [1, 3, 5]
IMG_INDICES      = [2, 3, 4, 5, 6]

# Bar-chart sort / display
# SORT_MODE: "alphabetical_by_detector" | "alphabetical_by_descriptor" | "metric"
SORT_MODE          = "alphabetical_by_detector"
METRIC_ASCENDING   = False
NUMBER_DESCENDING  = True
SECTION_COLOR_MODE = "auto"   # "auto" | "detector" | "descriptor"
SHADE_SECTIONS     = True
FIG_DPI            = 120
FONT_FAMILY_YTICKS = "monospace"

# Bar-chart columns to plot.
# None → auto: only show _auc summary columns + avg_num_matches
# Set to a list to override, e.g. ["mma_overall_all_auc", "mma_illum_all_auc"]
BAR_CHART_COLS = None

# Method filter
FILTER_WHITELIST_MODE = False
METHOD_FILTER         = []    # e.g. ["ORB+ORB"] to exclude
ENABLE_METHOD_FILTER  = False

# MMA curve plots: which (scope, difficulty) panels to show.
# One figure per unique scope; difficulties become subplots within it.
# scope: "overall" | "illumination" | "viewpoint"
# diff:  "all" | "easy" | "normal" | "hard"
CURVE_CATEGORIES = [
    ("overall", "all"),
    ("overall", "easy"),
    ("overall", "normal"),
    ("overall", "hard"),
    ("illumination",   "all"),
    ("viewpoint",    "all"),
]

# Repeatability curve plots (same structure as MMA, uses rep_*_th{th} columns)
REP_CURVE_CATEGORIES = [
    ("overall", "all"),
    ("overall", "easy"),
    ("overall", "normal"),
    ("overall", "hard"),
    ("illumination", "all"),
    ("viewpoint",    "all"),
]

# Homography accuracy curve plots (same structure, uses hom_acc_*_eps{eps} columns)
HOM_CURVE_CATEGORIES = [
    ("overall", "all"),
    ("overall", "easy"),
    ("overall", "normal"),
    ("overall", "hard"),
    ("illumination",   "all"),
    ("viewpoint",    "all"),
]

# Per-image lineplots: which scopes to show (x-axis = image index 2-6)
IMG_CURVE_SCOPES = ["overall", "illumination", "viewpoint"]

# ============================================================
# HELPERS
# ============================================================

def set_fig_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        try:
            fig.canvas.set_window_title(title)
        except Exception:
            pass


def passes_filter(combo):
    if not ENABLE_METHOD_FILTER:
        return True
    in_filter = combo in METHOD_FILTER
    return in_filter if FILTER_WHITELIST_MODE else not in_filter


def bar_chart(df_sorted, col, group_key, color_map, fig_dpi):
    split  = [s.split("+", 1) for s in df_sorted["detector"] + "+" + df_sorted["descriptor_full"]]
    maxL   = max(len(p[0]) for p in split)
    maxR   = max(len(p[1]) for p in split)
    labels = [p[0].ljust(maxL) + " + " + p[1].ljust(maxR) for p in split]
    y_pos  = np.arange(len(df_sorted))
    colors = [color_map[g] for g in df_sorted[group_key]]

    fig_h = max(6, len(df_sorted) * 0.18)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=fig_dpi)
    set_fig_title(fig, col)
    ax.barh(y_pos, df_sorted[col], color=colors)

    if SHADE_SECTIONS:
        seq = df_sorted[group_key].tolist()
        runs, start, curr = [], 0, seq[0]
        for j, v in enumerate(seq[1:], 1):
            if v != curr:
                runs.append((start, j, curr)); start = j; curr = v
        runs.append((start, len(seq), curr))
        xright = ax.get_xlim()[1] or float(df_sorted[col].max() or 1.0)
        for k, (a, b, g) in enumerate(runs):
            ax.axhspan(a - 0.5, b - 0.5, color=color_map[g], alpha=0.08, lw=0)
            if k > 0:
                ax.axhline(a - 0.5, color="gray", ls="--", alpha=0.7)
            ax.text(xright * 1.01, (a + b - 1) / 2, g,
                    ha="left", va="center", fontweight="bold", color=color_map[g])

    ax.set_title(col, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontfamily=FONT_FAMILY_YTICKS)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()


def make_interactive_legend(fig, leg, lined):
    """
    Wire up a legend so clicking a legend line (or its text label)
    toggles the visibility of all associated data lines.

    lined: dict  {legend_line_artist → [data_line, ...]}
    """
    # Also map legend text items so clicking the label works too
    text_to_lines = {}
    for legline, legtext, data_lines in zip(
        leg.get_lines(), leg.get_texts(), lined.values()
    ):
        legline.set_picker(6)
        legtext.set_picker(6)
        text_to_lines[legtext] = (legline, data_lines)

    def on_pick(event):
        artist = event.artist
        # Resolve to (legline, data_lines)
        if artist in lined:
            legline, data_lines = artist, lined[artist]
        elif artist in text_to_lines:
            legline, data_lines = text_to_lines[artist]
        else:
            return
        vis = not data_lines[0].get_visible()
        for line in data_lines:
            line.set_visible(vis)
        legline.set_alpha(1.0 if vis else 0.2)
        event.artist.get_figure().canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)


# ============================================================
# LOAD CSV
# ============================================================
df = pd.read_csv(CSV_PATH)
df["combination"] = df["combination"].astype(str).str.strip()
df = df[df["combination"].apply(passes_filter)].reset_index(drop=True)

parts = df["combination"].str.split("+", n=1, expand=True)
df["detector"]        = parts[0].str.strip()
df["descriptor_full"] = parts[1].str.strip()

desc_ex = df["descriptor_full"].str.extract(r"^(.*?)(?:\s+(\d+(?:\.\d+)?))?$")
df["descriptor_name"] = desc_ex[0].str.strip()
df["descriptor_num"]  = pd.to_numeric(desc_ex[1], errors="coerce")

# ============================================================
# SORTING
# ============================================================
dfw = df.assign(
    __det_key  = df["detector"].str.lower(),
    __desc_key = df["descriptor_name"].str.lower(),
    __comb_key = df["combination"].str.lower(),
)
desc_num_key = np.where(
    df["descriptor_num"].notna(),
    -df["descriptor_num"] if NUMBER_DESCENDING else df["descriptor_num"],
    -np.inf if NUMBER_DESCENDING else np.inf,
)
dfw["__desc_num_key"] = desc_num_key
SORT_HELPERS = ["__det_key", "__desc_key", "__desc_num_key", "__comb_key"]

df_by_det  = dfw.sort_values(by=SORT_HELPERS, kind="mergesort").drop(columns=SORT_HELPERS).reset_index(drop=True)
df_by_desc = dfw.sort_values(
    by=["__desc_key", "__desc_num_key", "__det_key", "__comb_key"], kind="mergesort"
).drop(columns=SORT_HELPERS).reset_index(drop=True)

# ============================================================
# COLOR MAPS
# ============================================================
all_det  = sorted(df["detector"].unique(), key=str.lower)
all_des  = sorted(df["descriptor_name"].unique(), key=str.lower)
cmap20   = plt.cm.get_cmap("tab20")
det_color  = {d: cmap20(i % 20) for i, d in enumerate(all_det)}
des_color  = {d: cmap20(i % 20) for i, d in enumerate(all_des)}
combo_color = {row["combination"]: cmap20(i % 20) for i, (_, row) in enumerate(df.iterrows())}

# ============================================================
# BAR CHARTS  (AUC columns only)
# ============================================================
extra_cols = [c for c in ["avg_num_matches"] if c in df.columns]

if BAR_CHART_COLS is not None:
    bar_cols = [c for c in BAR_CHART_COLS if c in df.columns]
else:
    bar_cols = extra_cols   # _auc columns are already covered by curve/line plots

for col in bar_cols:
    if SORT_MODE == "alphabetical_by_detector":
        df_sorted = df_by_det.copy()
        group_key = "detector" if SECTION_COLOR_MODE != "descriptor" else "descriptor_name"
    elif SORT_MODE == "alphabetical_by_descriptor":
        df_sorted = df_by_desc.copy()
        group_key = "descriptor_name" if SECTION_COLOR_MODE != "detector" else "detector"
    else:
        df_sorted = df_by_det.sort_values(by=col, ascending=METRIC_ASCENDING, kind="mergesort")
        group_key = "detector"

    color_map = det_color if group_key == "detector" else des_color
    bar_chart(df_sorted, col, group_key, color_map, FIG_DPI)

# ============================================================
# MMA CURVE PLOTS  (one figure per scope, subplots per difficulty)
#
# Click a legend line or its label to toggle that method on/off.
# ============================================================

# Group configured categories by scope
scope_to_diffs = defaultdict(list)
for scope, diff in CURVE_CATEGORIES:
    if diff not in scope_to_diffs[scope]:
        scope_to_diffs[scope].append(diff)

for scope, diffs in scope_to_diffs.items():
    # Check at least one category has data
    valid_diffs = [
        d for d in diffs
        if all(f"mma_{scope}_{d}_th{th}" in df.columns for th in PIXEL_THRESHOLDS)
    ]
    if not valid_diffs:
        continue

    n       = len(valid_diffs)
    ncols   = min(n, 2)
    nrows   = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                              dpi=FIG_DPI, squeeze=False)

    title = f"MMA curves — {scope}"
    set_fig_title(fig, title)
    fig.suptitle(title, fontweight="bold")

    # combo → list of ALL line objects across every subplot
    combo_all_lines: dict[str, list] = {row["combination"]: [] for _, row in df.iterrows()}

    for panel_idx, diff in enumerate(valid_diffs):
        row_i, col_i = divmod(panel_idx, ncols)
        ax = axes[row_i][col_i]
        th_cols = [f"mma_{scope}_{diff}_th{th}" for th in PIXEL_THRESHOLDS]

        for _, data_row in df.iterrows():
            combo    = data_row["combination"]
            mma_vals = [data_row[c] for c in th_cols]
            line, = ax.plot(
                PIXEL_THRESHOLDS, mma_vals,
                color=combo_color[combo],
                linewidth=1.8,
                marker="o", markersize=3,
                label=combo,
            )
            combo_all_lines[combo].append(line)

        ax.set_title(diff, fontweight="bold")
        ax.set_xlabel("Pixel threshold")
        ax.set_ylabel("MMA")
        ax.set_xticks(PIXEL_THRESHOLDS)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide any unused subplot panels
    for extra in range(n, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    # ---- Shared interactive legend outside the subplots ----
    # Use the lines from the first panel as legend proxies
    proxy_lines = [combo_all_lines[combo][0] for combo in combo_all_lines if combo_all_lines[combo]]
    combos_ordered = [combo for combo in combo_all_lines if combo_all_lines[combo]]

    leg = fig.legend(
        proxy_lines, combos_ordered,
        loc="outside right upper",
        fontsize=8,
        title="Method",
        title_fontsize=9,
        framealpha=0.9,
    )

    # lined: legend_line → all data lines for that combo
    lined = {
        legline: combo_all_lines[combo]
        for legline, combo in zip(leg.get_lines(), combos_ordered)
    }
    make_interactive_legend(fig, leg, lined)
    plt.tight_layout()

# ============================================================
# REPEATABILITY CURVE PLOTS  (x = pixel threshold, same as MMA)
# ============================================================
rep_scope_to_diffs = defaultdict(list)
for scope, diff in REP_CURVE_CATEGORIES:
    if diff not in rep_scope_to_diffs[scope]:
        rep_scope_to_diffs[scope].append(diff)

for scope, diffs in rep_scope_to_diffs.items():
    valid_diffs = [
        d for d in diffs
        if all(f"rep_{scope}_{d}_th{th}" in df.columns for th in PIXEL_THRESHOLDS)
    ]
    if not valid_diffs:
        continue

    n     = len(valid_diffs)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                              dpi=FIG_DPI, squeeze=False)

    title = f"Repeatability — {scope}"
    set_fig_title(fig, title)
    fig.suptitle(title, fontweight="bold")

    combo_all_lines: dict[str, list] = {row["combination"]: [] for _, row in df.iterrows()}

    for panel_idx, diff in enumerate(valid_diffs):
        row_i, col_i = divmod(panel_idx, ncols)
        ax = axes[row_i][col_i]
        th_cols = [f"rep_{scope}_{diff}_th{th}" for th in PIXEL_THRESHOLDS]

        for _, data_row in df.iterrows():
            combo = data_row["combination"]
            vals  = [data_row[c] for c in th_cols]
            line, = ax.plot(
                PIXEL_THRESHOLDS, vals,
                color=combo_color[combo],
                linewidth=1.8,
                marker="o", markersize=3,
                label=combo,
            )
            combo_all_lines[combo].append(line)

        ax.set_title(diff, fontweight="bold")
        ax.set_xlabel("Pixel threshold")
        ax.set_ylabel("Repeatability")
        ax.set_xticks(PIXEL_THRESHOLDS)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)

    for extra in range(n, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    proxy_lines    = [combo_all_lines[combo][0] for combo in combo_all_lines if combo_all_lines[combo]]
    combos_ordered = [combo for combo in combo_all_lines if combo_all_lines[combo]]
    leg = fig.legend(proxy_lines, combos_ordered, loc="outside right upper",
                     fontsize=8, title="Method", title_fontsize=9, framealpha=0.9)
    lined = {legline: combo_all_lines[combo]
             for legline, combo in zip(leg.get_lines(), combos_ordered)}
    make_interactive_legend(fig, leg, lined)
    plt.tight_layout()

# ============================================================
# HOMOGRAPHY ACCURACY CURVE PLOTS  (x = epsilon threshold)
# ============================================================
hom_scope_to_diffs = defaultdict(list)
for scope, diff in HOM_CURVE_CATEGORIES:
    if diff not in hom_scope_to_diffs[scope]:
        hom_scope_to_diffs[scope].append(diff)

for scope, diffs in hom_scope_to_diffs.items():
    valid_diffs = [
        d for d in diffs
        if all(f"hom_acc_{scope}_{d}_eps{eps}" in df.columns for eps in HOM_THRESHOLDS)
    ]
    if not valid_diffs:
        continue

    n     = len(valid_diffs)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                              dpi=FIG_DPI, squeeze=False)

    title = f"Homography accuracy — {scope}"
    set_fig_title(fig, title)
    fig.suptitle(title, fontweight="bold")

    combo_all_lines: dict[str, list] = {row["combination"]: [] for _, row in df.iterrows()}

    for panel_idx, diff in enumerate(valid_diffs):
        row_i, col_i = divmod(panel_idx, ncols)
        ax = axes[row_i][col_i]
        eps_cols = [f"hom_acc_{scope}_{diff}_eps{eps}" for eps in HOM_THRESHOLDS]

        for _, data_row in df.iterrows():
            combo = data_row["combination"]
            vals  = [data_row[c] for c in eps_cols]
            line, = ax.plot(
                HOM_THRESHOLDS, vals,
                color=combo_color[combo],
                linewidth=1.8,
                marker="o", markersize=3,
                label=combo,
            )
            combo_all_lines[combo].append(line)

        ax.set_title(diff, fontweight="bold")
        ax.set_xlabel("Corner error threshold (px)")
        ax.set_ylabel("Fraction correct")
        ax.set_xticks(HOM_THRESHOLDS)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)

    for extra in range(n, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    proxy_lines    = [combo_all_lines[combo][0] for combo in combo_all_lines if combo_all_lines[combo]]
    combos_ordered = [combo for combo in combo_all_lines if combo_all_lines[combo]]
    leg = fig.legend(proxy_lines, combos_ordered, loc="outside right upper",
                     fontsize=8, title="Method", title_fontsize=9, framealpha=0.9)
    lined = {legline: combo_all_lines[combo]
             for legline, combo in zip(leg.get_lines(), combos_ordered)}
    make_interactive_legend(fig, leg, lined)
    plt.tight_layout()

# ============================================================
# PER-IMAGE LINEPLOTS  (x = image index 2–6)
# One figure per scope; two subplots: MMA AUC and Hom-acc AUC.
# ============================================================
for scope in IMG_CURVE_SCOPES:
    mma_cols = [f"mma_{scope}_img{i}_auc"     for i in IMG_INDICES]
    rep_cols = [f"rep_{scope}_img{i}_auc"     for i in IMG_INDICES]
    hom_cols = [f"hom_acc_{scope}_img{i}_auc" for i in IMG_INDICES]
    has_mma  = all(c in df.columns for c in mma_cols)
    has_rep  = all(c in df.columns for c in rep_cols)
    has_hom  = all(c in df.columns for c in hom_cols)
    if not has_mma and not has_rep and not has_hom:
        continue

    panels = []
    if has_mma: panels.append(("MMA AUC",          mma_cols))
    if has_rep: panels.append(("Repeatability AUC", rep_cols))
    if has_hom: panels.append(("Hom-acc AUC",       hom_cols))

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5),
                              dpi=FIG_DPI, squeeze=False)
    title = f"Performance vs image index — {scope}"
    set_fig_title(fig, title)
    fig.suptitle(title, fontweight="bold")

    combo_all_lines: dict[str, list] = {row["combination"]: [] for _, row in df.iterrows()}

    for col_i, (ylabel, cols) in enumerate(panels):
        ax = axes[0][col_i]
        for _, data_row in df.iterrows():
            combo = data_row["combination"]
            vals  = [data_row[c] for c in cols]
            line, = ax.plot(
                IMG_INDICES, vals,
                color=combo_color[combo],
                linewidth=1.8,
                marker="o", markersize=3,
                label=combo,
            )
            combo_all_lines[combo].append(line)

        ax.set_title(ylabel, fontweight="bold")
        ax.set_xlabel("Image index")
        ax.set_ylabel(ylabel)
        ax.set_xticks(IMG_INDICES)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", alpha=0.5)

    proxy_lines    = [combo_all_lines[combo][0] for combo in combo_all_lines if combo_all_lines[combo]]
    combos_ordered = [combo for combo in combo_all_lines if combo_all_lines[combo]]
    leg = fig.legend(proxy_lines, combos_ordered, loc="outside right upper",
                     fontsize=8, title="Method", title_fontsize=9, framealpha=0.9)
    lined = {legline: combo_all_lines[combo]
             for legline, combo in zip(leg.get_lines(), combos_ordered)}
    make_interactive_legend(fig, leg, lined)
    plt.tight_layout()

plt.show()
