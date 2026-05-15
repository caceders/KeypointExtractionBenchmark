import colorsys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "results/mma_results_SIFT.csv"
FIG_DPI  = 120

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one matplotlib figure.
#
# Required:
#   title    — window title and figure suptitle
#   x        — CSV column for the x-axis
#   lines    — CSV column whose unique values become separate lines
#   y_stat   — which value column to read:
#                 "mean" | "std" | "min" | "max"   (per-bucket stats)
#                 or any other numeric column, e.g. "avg_num_features"
#
# Optional:
#   subplots — CSV column that creates subplot panels (default: None → one panel)
#   agg      — list of collapse steps applied left to right, each:
#                 {"col": column_name, "fn": "auc"|"mean"|"std"|"min"|"max"}
#                 {"col": column_name, "fn": "auc", "range": [lo, hi]}
#              "auc" = mean over the values (equally-spaced thresholds).
#              "range" restricts which values of col are included before aggregating,
#              e.g. {"col": "threshold", "fn": "auc", "range": [3, 8]} computes
#              AUC only for pixel thresholds 3–8.
#              Steps reduce one dimension at a time; after all steps, one scalar
#              y-value remains per (subplot, x, line) combination.
#              Omit or use [] to plot y_stat directly.
#   select   — dict that fixes column values not covered by x/lines/subplots/agg:
#                 scalar value → equality
#                 list of values → "any of these" (isin)
#              If a column has multiple values and isn't assigned to any dimension,
#              the plotter warns and takes the mean.
#
# Long-format CSV column reference:
#   Identity:   combination, downsample_level, max_features,
#               ratio_threshold, ransac_reproj
#   Summaries:  avg_num_features, frac_below_max_features, avg_num_matches
#   Dimensions: scope, difficulty, metric, threshold
#   Stats:      mean, std, min, max, count
# ──────────────────────────────────────────────────────────────────────────────
PLOTS = [
    # MMA curve: threshold vs mean MMA, subplots per difficulty
    {
        "title":    "MMA curve — overall",
        "x":        "threshold",
        "lines":    "combination",
        "subplots": "difficulty",
        "y_stat":   "mean",
        "agg":      [],
        "select": {
            "metric":           "mma",
            "scope":            "overall",
            "difficulty":       ["all", "easy", "normal", "hard"],
            "max_features":     1000,
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
            "downsample_level": 0,
        },
    },
    # MMA AUC vs max_features sweep
    {
        "title":    "MMA AUC vs max features",
        "x":        "max_features",
        "lines":    "combination",
        "subplots": None,
        "y_stat":   "mean",
        "agg":      [{"col": "threshold", "fn": "auc"}],
        "select": {
            "metric":           "mma",
            "scope":            "overall",
            "difficulty":       "all",
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
            "downsample_level": 0,
        },
    },
    # Repeatability curve
    {
        "title":    "Repeatability curve — overall",
        "x":        "threshold",
        "lines":    "combination",
        "subplots": "difficulty",
        "y_stat":   "mean",
        "agg":      [],
        "select": {
            "metric":           "rep",
            "scope":            "overall",
            "difficulty":       ["all", "easy", "normal", "hard"],
            "max_features":     1000,
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
            "downsample_level": 0,
        },
    },
    # Homography accuracy curve
    {
        "title":    "Homography accuracy curve — overall",
        "x":        "threshold",
        "lines":    "combination",
        "subplots": "difficulty",
        "y_stat":   "mean",
        "agg":      [],
        "select": {
            "metric":           "hom_acc",
            "scope":            "overall",
            "difficulty":       ["all", "easy", "normal", "hard"],
            "max_features":     1000,
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
            "downsample_level": 0,
        },
    },
    # MMA AUC per image index (difficulty progression)
    {
        "title":    "MMA AUC per image — overall",
        "x":        "difficulty",
        "lines":    "combination",
        "subplots": None,
        "y_stat":   "mean",
        "agg":      [{"col": "threshold", "fn": "auc"}],
        "select": {
            "metric":           "mma",
            "scope":            "overall",
            "difficulty":       ["img2", "img3", "img4", "img5", "img6"],
            "max_features":     1000,
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
            "downsample_level": 0,
        },
    },
    # MMA AUC vs downsample level
    {
        "title":    "MMA AUC vs downsample level",
        "x":        "downsample_level",
        "lines":    "combination",
        "subplots": None,
        "y_stat":   "mean",
        "agg":      [{"col": "threshold", "fn": "auc"}],
        "select": {
            "metric":           "mma",
            "scope":            "overall",
            "difficulty":       "all",
            "max_features":     1000,
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
        },
    },
    # Avg features detected vs max_features
    {
        "title":    "Avg features detected vs max features",
        "x":        "max_features",
        "lines":    "combination",
        "subplots": None,
        "y_stat":   "avg_num_features",
        "agg":      [],
        "select": {
            "scope":            "overall",
            "difficulty":       "all",
            "metric":           "mma",
            "threshold":        1,
            "ratio_threshold":  0.8,
            "ransac_reproj":    3.0,
            "downsample_level": 0,
        },
    },
]

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


def _distinct_colors(n):
    """Maximally distinct colors via golden-ratio hue stepping + alternating lightness."""
    golden = 0.618033988749895
    h = 0.0
    colors = []
    for i in range(n):
        s = 0.85 if i % 2 == 0 else 0.65
        l = 0.38 if (i // 2) % 2 == 0 else 0.58
        colors.append(colorsys.hls_to_rgb(h % 1.0, l, s))
        h += golden
    return colors


def make_interactive_legend(fig, leg, lined):
    """
    lined: {legend_line_artist → [data_line, ...]}
    Single-click → toggle visibility.
    Double-click → isolate (double-click again → restore all).
    """
    text_to_entry = {}
    for legline, legtext, data_lines in zip(leg.get_lines(), leg.get_texts(), lined.values()):
        legline.set_picker(6)
        legtext.set_picker(6)
        text_to_entry[legtext] = (legline, data_lines)

    _state = {"isolated": None, "suppress_pick": False}

    def _find_entry(event):
        try:
            renderer = fig.canvas.get_renderer()
        except Exception:
            return None, None
        for legline, legtext in zip(leg.get_lines(), leg.get_texts()):
            lb = legline.get_window_extent(renderer)
            tb = legtext.get_window_extent(renderer)
            if lb.contains(event.x, event.y) or tb.contains(event.x, event.y):
                return legline, lined[legline]
        return None, None

    def on_pick(event):
        if _state["suppress_pick"]:
            _state["suppress_pick"] = False
            return
        artist = event.artist
        if artist in lined:
            legline, data_lines = artist, lined[artist]
        elif artist in text_to_entry:
            legline, data_lines = text_to_entry[artist]
        else:
            return
        vis = not data_lines[0].get_visible()
        for line in data_lines:
            line.set_visible(vis)
        legline.set_alpha(1.0 if vis else 0.2)
        _state["isolated"] = None
        event.artist.get_figure().canvas.draw()

    def on_button_press(event):
        if not event.dblclick:
            return
        try:
            in_legend = leg.get_window_extent().contains(event.x, event.y)
        except Exception:
            in_legend = True
        if not in_legend:
            return
        legline, data_lines = _find_entry(event)
        if legline is None:
            return
        _state["suppress_pick"] = True
        if _state["isolated"] is legline:
            for ll, dls in lined.items():
                for line in dls:
                    line.set_visible(True)
                ll.set_alpha(1.0)
            _state["isolated"] = None
        else:
            for ll, dls in lined.items():
                vis = (ll is legline)
                for line in dls:
                    line.set_visible(vis)
                ll.set_alpha(1.0 if vis else 0.2)
            _state["isolated"] = legline
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", on_button_press)


def _apply_fn(values, fn):
    arr = np.array(values, dtype=np.float64)
    if fn in ("auc", "mean"):
        return float(np.mean(arr))
    elif fn == "std":
        return float(np.std(arr))
    elif fn == "min":
        return float(np.min(arr))
    elif fn == "max":
        return float(np.max(arr))
    else:
        raise ValueError(f"Unknown agg fn '{fn}'. Use: auc | mean | std | min | max")


def _collapse(df, y_col, agg_steps):
    """
    Recursively collapse one dimension at a time.

    Each step groups the current DataFrame by every column except the step's
    column and y_col, applies the aggregation function to y_col, and produces
    a smaller DataFrame. After all steps, a single scalar y-value is returned.
    """
    if not agg_steps:
        if len(df) != 1:
            # Multiple rows remain — values should all be equal (summary column);
            # silently take the mean.
            return float(df[y_col].mean())
        return float(df[y_col].iloc[0])

    step       = agg_steps[0]
    col        = step["col"]
    fn         = step["fn"]
    th_range   = step.get("range")          # optional [lo, hi] to restrict before aggregating
    other_cols = [c for c in df.columns if c != col and c != y_col]

    def _filtered(subdf):
        if th_range is not None:
            lo, hi = th_range
            subdf = subdf[(subdf[col] >= lo) & (subdf[col] <= hi)]
        return subdf

    if not other_cols:
        return _apply_fn(_filtered(df)[y_col].values, fn)

    results = []
    for keys, subdf in df.groupby(other_cols, sort=False, dropna=False):
        agg_val = _apply_fn(_filtered(subdf)[y_col].values, fn)
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(other_cols, keys))
        row[y_col] = agg_val
        results.append(row)

    return _collapse(pd.DataFrame(results), y_col, agg_steps[1:])


def make_plot(cfg, df, combo_color):
    title        = cfg["title"]
    x_col        = cfg["x"]
    lines_col    = cfg["lines"]
    subplots_col = cfg.get("subplots")
    y_stat       = cfg.get("y_stat", "mean")
    agg_steps    = cfg.get("agg", [])
    select       = cfg.get("select", {})

    # ── Apply select ──────────────────────────────────────────────────────────
    dfs = df.copy()
    for col, val in select.items():
        if col not in dfs.columns:
            print(f"[{title}] select: column '{col}' not in CSV — skipping this filter.")
            continue
        if isinstance(val, list):
            dfs = dfs[dfs[col].isin(val)]
        elif isinstance(val, float) and math.isnan(val):
            dfs = dfs[dfs[col].isna()]
        else:
            dfs = dfs[dfs[col] == val]

    if dfs.empty:
        print(f"[{title}] No data after select — skipping plot.")
        return

    # ── Determine subplot panels ──────────────────────────────────────────────
    if subplots_col and subplots_col in dfs.columns:
        raw_panels = dfs[subplots_col].unique()
        try:
            panels = sorted(raw_panels, key=float)
        except (ValueError, TypeError):
            panels = sorted(raw_panels, key=str)
    else:
        panels = [None]

    n_panels = len(panels)
    ncols    = min(n_panels, 3)
    nrows    = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7 * ncols, 5 * nrows),
                              dpi=FIG_DPI, squeeze=False)
    set_fig_title(fig, title)
    fig.suptitle(title, fontweight="bold")

    # ── Color assignment ──────────────────────────────────────────────────────
    all_line_vals = sorted(dfs[lines_col].unique(), key=str)
    if lines_col == "combination":
        line_color = {v: combo_color.get(v) for v in all_line_vals}
    else:
        line_color = dict(zip(all_line_vals, _distinct_colors(len(all_line_vals))))

    # Track all Line2D objects per line value (across ALL panels) for shared legend
    line_artists: dict = {v: [] for v in all_line_vals}

    # Columns that carry y-relevant variance (used to slim down the DataFrame per group)
    agg_cols = [s["col"] for s in agg_steps]

    for panel_idx, panel_val in enumerate(panels):
        row_i, col_i = divmod(panel_idx, ncols)
        ax = axes[row_i][col_i]

        panel_df = (dfs[dfs[subplots_col] == panel_val]
                    if panel_val is not None else dfs)

        if panel_val is not None:
            ax.set_title(str(panel_val), fontweight="bold")

        # Sort x values numerically if possible, else alphabetically
        x_vals = panel_df[x_col].unique()
        try:
            x_vals = sorted(x_vals, key=float)
        except (ValueError, TypeError):
            x_vals = sorted(x_vals, key=str)

        for line_val in all_line_vals:
            line_df = panel_df[panel_df[lines_col] == line_val]
            if line_df.empty:
                continue

            y_vals = []
            for xv in x_vals:
                group_df = line_df[line_df[x_col] == xv]
                if group_df.empty:
                    y_vals.append(float("nan"))
                    continue

                if agg_steps:
                    # Keep only the columns needed for the agg chain
                    keep = [c for c in [y_stat] + agg_cols if c in group_df.columns]
                    yv   = _collapse(group_df[keep], y_stat, agg_steps)
                else:
                    unique_vals = group_df[y_stat].dropna().unique()
                    if len(unique_vals) > 1:
                        print(f"[{title}] Warning: {len(group_df)} rows with differing "
                              f"'{y_stat}' for {lines_col}={line_val!r} x={xv!r} — "
                              f"taking mean. Add agg or select to disambiguate.")
                    yv = float(group_df[y_stat].mean())
                y_vals.append(yv)

            line_obj, = ax.plot(
                x_vals, y_vals,
                color=line_color.get(line_val),
                linewidth=1.8,
                marker="o", markersize=3,
                label=str(line_val),
            )
            line_artists[line_val].append(line_obj)

        ax.set_xlabel(x_col)
        y_label = y_stat if not agg_steps else f"{agg_steps[-1]['fn']}({y_stat})"
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide unused panels
    for extra in range(n_panels, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    # ── Shared interactive legend ─────────────────────────────────────────────
    present = {v: arts for v, arts in line_artists.items() if arts}
    if present:
        proxy_lines = [arts[0] for arts in present.values()]
        labels      = list(present.keys())
        leg = fig.legend(
            proxy_lines, labels,
            loc="outside right upper",
            fontsize=8,
            title=lines_col,
            title_fontsize=9,
            framealpha=0.9,
        )
        lined = {legline: arts
                 for legline, arts in zip(leg.get_lines(), present.values())}
        make_interactive_legend(fig, leg, lined)

    plt.tight_layout()


# ============================================================
# LOAD CSV
# ============================================================
df = pd.read_csv(CSV_PATH)
df["combination"] = df["combination"].astype(str).str.strip()

parts = df["combination"].str.split("+", n=1, expand=True)
df["detector"]   = parts[0].str.strip()
df["descriptor"] = parts[1].str.strip()

all_combos  = list(df["combination"].unique())
combo_color = dict(zip(all_combos, _distinct_colors(len(all_combos))))

# ============================================================
# RENDER ALL PLOTS
# ============================================================
for cfg in PLOTS:
    make_plot(cfg, df, combo_color)

plt.show()
