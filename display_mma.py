import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "results/mma_results.csv"

y_default = "mma_matches_mean"
x_default = "combination"
lines_default = "combination"

PLOTS = [

    # {
    #     "y":        y_default,
    #     "x":        x_default,
    #     # "lines":    lines_default,
    #     # "subplots" : "downsample_level",
    #     "bar":      True,
    #     "select": {
    #         # [COL]: {"values": [VALS], "fn": [FUNCTION]},
    #         "tag": {"values": "baseline"},
    #     },
    # },

    {
        "y":        "mma_matches_mean",
        "x":        "max_features",
        "lines":    "combination",
        "subplots" : "downsample_level",
        "bar":      False,
        "select": {
            # [COL]: {"values": [VALS], "fn": [FUNCTION]}
            "threshold": {"values": np.arange(0,6), "fn": "auc"},
            "tag": {"values": "autoscale"},
        },
    },


]

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS — each entry produces one matplotlib figure.
#
# ── Axes ──────────────────────────────────────────────────────────────────────
#   x        — CSV column for the x-axis
#   y        — CSV column to use as the y-value, e.g.:
#                 "mma_kps_mean"    "mma_matches_mean"
#                 "rep_mean"        "hom_acc_mean"
#                 "avg_num_features"  "avg_num_matches"  ...
#   lines    — CSV column whose unique values become separate lines
#   subplots — CSV column that creates subplot panels (default: None → one panel)
#
# ── Select ────────────────────────────────────────────────────────────────────
#   select — dict mapping column → spec, processed in key order:
#                 {"values": [...]}                  filter to these values only
#                 {"fn": "mean"}                     collapse all values with fn
#                 {"fn": "auc", "range": [lo, hi]}   collapse values in range
#                 {"values": [...], "fn": "mean"}    filter then collapse
#              Columns with "fn" are collapsed in dict insertion order (order matters).
#              Columns without "fn" are pinned to their values (filter only).
#              Columns not mentioned and not in x/lines/subplots collapse silently.
#              fn options: "auc" (= mean) | "mean" | "std" | "min" | "max"
#
# ── Labels (all auto-generated if omitted) ────────────────────────────────────
#   title         — figure window title and suptitle
#   x_label       — x-axis label  (auto: column name with underscores → spaces)
#   y_label       — y-axis label  (auto: agg ops nested around y)
#   subplot_label — title for each subplot panel
#
# ── CSV column reference ──────────────────────────────────────────────────────
#   Identity:        combination, tag, downsample_level, max_features,
#                    ratio_threshold, ransac_reproj
#   Summaries:       avg_num_features, frac_below_max_features, avg_num_matches
#   Dimensions:      scope, difficulty, threshold
#   Per-metric stats (wide format — one row per scope/difficulty/threshold):
#     mma_kps_mean,     mma_kps_std,     mma_kps_min,     mma_kps_max       — correct / n_ref_keypoints
#     mma_matches_mean, mma_matches_std, mma_matches_min, mma_matches_max   — correct / n_putative_matches
#     rep_mean,         rep_std,         rep_min,         rep_max
#     hom_acc_mean,     hom_acc_std,     hom_acc_min,     hom_acc_max
# ──────────────────────────────────────────────────────────────────────────────

FIG_DPI  = 120

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
    """Hand-picked palette — one of each colour family, no two look alike."""
    palette = [
        (0.902, 0.098, 0.294),  # red
        (0.263, 0.388, 0.847),  # blue
        (0.961, 0.510, 0.192),  # orange
        (0.235, 0.706, 0.294),  # green
        (0.569, 0.118, 0.706),  # purple
        (1.000, 0.882, 0.098),  # yellow
        (0.275, 0.600, 0.565),  # teal
        (0.941, 0.196, 0.902),  # magenta
        (0.604, 0.388, 0.141),  # brown
        (0.259, 0.831, 0.957),  # sky blue
        (0.502, 0.000, 0.000),  # maroon
        (0.863, 0.745, 1.000),  # lavender
    ]
    def _lighten(rgb, amount):
        return tuple(min(1.0, c + amount) for c in rgb)
    shifts = [0.0, 0.25, -0.15]
    return [_lighten(palette[i % len(palette)], shifts[(i // len(palette)) % len(shifts)]) for i in range(n)]


def make_interactive_legend(fig, leg, artist_map):
    """
    artist_map: {legend_handle → [data_artists]}
    Works with Line2D (line plots) or Patch/BarContainer (bar plots).
    Single-click → toggle visibility. Double-click → isolate (again → restore all).
    """
    def _get_vis(data_artists):
        a = data_artists[0]
        ps = getattr(a, "patches", None)
        return ps[0].get_visible() if ps else a.get_visible()

    def _set_vis(data_artists, vis):
        for a in data_artists:
            ps = getattr(a, "patches", None)
            if ps:
                for p in ps:
                    p.set_visible(vis)
            else:
                a.set_visible(vis)

    leg_handles = list(artist_map.keys())
    text_to_entry = {}
    for handle, legtext in zip(leg_handles, leg.get_texts()):
        handle.set_picker(6)
        legtext.set_picker(6)
        text_to_entry[legtext] = (handle, artist_map[handle])

    _state = {"isolated": None, "suppress_pick": False}

    def _find_entry(event):
        try:
            renderer = fig.canvas.get_renderer()
        except Exception:
            return None, None
        for handle, legtext in zip(leg_handles, leg.get_texts()):
            hb = handle.get_window_extent(renderer)
            tb = legtext.get_window_extent(renderer)
            if hb.contains(event.x, event.y) or tb.contains(event.x, event.y):
                return handle, artist_map[handle]
        return None, None

    def on_pick(event):
        if _state["suppress_pick"]:
            _state["suppress_pick"] = False
            return
        artist = event.artist
        if artist in artist_map:
            handle, data_artists = artist, artist_map[artist]
        elif artist in text_to_entry:
            handle, data_artists = text_to_entry[artist]
        else:
            return
        vis = not _get_vis(data_artists)
        _set_vis(data_artists, vis)
        handle.set_alpha(1.0 if vis else 0.2)
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
        handle, _ = _find_entry(event)
        if handle is None:
            return
        _state["suppress_pick"] = True
        if _state["isolated"] is handle:
            for h, das in artist_map.items():
                _set_vis(das, True)
                h.set_alpha(1.0)
            _state["isolated"] = None
        else:
            for h, das in artist_map.items():
                vis = (h is handle)
                _set_vis(das, vis)
                h.set_alpha(1.0 if vis else 0.2)
            _state["isolated"] = handle
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", on_button_press)


def _sorted_vals(vals):
    """Sort values numerically if all are numeric, else alphabetically."""
    try:
        return sorted(vals, key=float)
    except (ValueError, TypeError):
        return sorted(vals, key=str)


def _m(text):
    return r"\mathrm{" + text.replace("_", r"\ ") + "}"


def _auto_y_label(y, agg_steps):
    """Build y-axis label with mathtext subscripts: mean_scope(mean_difficulty(y))."""
    if not agg_steps:
        return y.replace("_", " ")
    label = _m(y)
    for step in reversed(agg_steps):
        fn  = step["fn"]
        col = step["col"]
        rng = step.get("range")
        core = r"\underset{" + _m(col) + r"}{" + _m(fn) + r"}(" + label + ")"
        label = core if not rng else core + f"[{rng[0]}-{rng[1]}]"
    return "$" + label + "$"


def _auto_title(cfg, agg_steps):
    y_label = _auto_y_label(cfg.get("y", "mma_kps_mean"), agg_steps)
    x_col   = cfg.get("x")
    lines_col = cfg.get("lines")
    x_label = x_col.replace("_", " ") if x_col else (lines_col.replace("_", " ") if lines_col else "")
    agg     = cfg.get("select", {})
    parts   = []
    for k, spec in agg.items():
        if not isinstance(spec, dict):
            spec = {"values": spec}
        v = spec.get("values")
        if v is None:
            continue
        parts.append(f"{k}=[{','.join(str(i) for i in v)}]" if isinstance(v, list) else f"{k}={v}")
    title = f"{y_label} vs {x_label}"
    if parts:
        title += "  (" + ", ".join(parts) + ")"
    return title


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
            return float(df[y_col].mean())
        return float(df[y_col].iloc[0])

    step       = agg_steps[0]
    col        = step["col"]
    fn         = step["fn"]
    th_range   = step.get("range")
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


def make_plot(cfg, df, combo_color, tag_color):
    x_col        = cfg.get("x")
    bar_mode     = cfg.get("bar", False)
    lines_col    = cfg.get("lines")
    subplots_col = cfg.get("subplots")
    y            = cfg.get("y", "mma_kps_mean")
    select       = cfg.get("select", {})

    # ── Parse select: build filter + ordered agg_steps ────────────────────
    agg_steps = []
    dfs = df.copy()
    for col, spec in select.items():
        if not isinstance(spec, dict):
            spec = {"values": spec}
        values = spec.get("values")
        if values is not None:
            if col not in dfs.columns:
                print(f"[{cfg.get('title', '?')}] select: column '{col}' not in CSV — skipping.")
                continue
            if isinstance(values, (list, np.ndarray)):
                dfs = dfs[dfs[col].isin(values)]
            elif isinstance(values, float) and math.isnan(values):
                dfs = dfs[dfs[col].isna()]
            else:
                dfs = dfs[dfs[col] == values]
        if "fn" in spec:
            step = {"col": col, "fn": spec["fn"]}
            if "range" in spec:
                step["range"] = spec["range"]
            agg_steps.append(step)

    # ── Auto-generate labels (all overridable) ────────────────────────────────
    title   = cfg.get("title",   _auto_title(cfg, agg_steps))
    x_label = cfg.get("x_label", x_col.replace("_", " ") if x_col else (lines_col.replace("_", " ") if lines_col else ""))
    y_label = cfg.get("y_label", _auto_y_label(y, agg_steps))

    if dfs.empty:
        print(f"[{title}] No data after filter — skipping plot.")
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
    if lines_col is not None:
        all_line_vals = _sorted_vals(dfs[lines_col].unique())
        if lines_col == "combination":
            line_color = {v: combo_color.get(v) for v in all_line_vals}
        elif lines_col == "tag":
            line_color = {v: tag_color.get(v) for v in all_line_vals}
        else:
            line_color = dict(zip(all_line_vals, _distinct_colors(len(all_line_vals))))
    else:
        all_line_vals = [None]
        line_color    = {None: "#1f77b4"}

    line_artists:   dict = {v: [] for v in all_line_vals}
    bar_containers: dict = {v: [] for v in all_line_vals}

    agg_cols = [s["col"] for s in agg_steps]

    for panel_idx, panel_val in enumerate(panels):
        row_i, col_i = divmod(panel_idx, ncols)
        ax = axes[row_i][col_i]

        panel_df = (dfs[dfs[subplots_col] == panel_val]
                    if panel_val is not None else dfs)

        if panel_val is not None:
            subplot_label = cfg.get("subplot_label", f"{subplots_col.replace('_', ' ')} = {panel_val}")
            ax.set_title(subplot_label, fontweight="bold")

        if bar_mode:
            x_vals    = _sorted_vals(panel_df[x_col].unique())
            n_groups  = len(x_vals)
            n_bars    = len(all_line_vals)
            ungrouped = lines_col is None
            height    = 0.97 if ungrouped else 0.97 / max(n_bars, 1)
            offsets   = [0] * n_bars if ungrouped else [(i - (n_bars - 1) / 2) * height for i in range(n_bars)]

            for bar_idx, line_val in enumerate(all_line_vals):
                line_df = panel_df if line_val is None else panel_df[panel_df[lines_col] == line_val]
                bar_vals = []
                for xv in x_vals:
                    group_df = line_df[line_df[x_col] == xv]
                    if group_df.empty:
                        bar_vals.append(float("nan"))
                    elif agg_steps:
                        keep = [c for c in [y] + agg_cols if c in group_df.columns]
                        bar_vals.append(_collapse(group_df[keep], y, agg_steps))
                    else:
                        bar_vals.append(float(group_df[y].mean()))
                positions = [i + offsets[bar_idx] for i in range(n_groups)]
                container = ax.barh(positions, bar_vals, height=height,
                                    color=line_color.get(line_val),
                                    label=str(line_val) if line_val is not None else None)
                bar_containers[line_val].append(container)

            ax.set_yticks(list(range(n_groups)))
            ax.set_yticklabels([str(v) for v in x_vals])
        else:
            x_vals = _sorted_vals(panel_df[x_col].unique())

            for line_val in all_line_vals:
                line_df = panel_df if line_val is None else panel_df[panel_df[lines_col] == line_val]
                if line_df.empty:
                    continue

                y_vals = []
                for xv in x_vals:
                    group_df = line_df[line_df[x_col] == xv]
                    if group_df.empty:
                        y_vals.append(float("nan"))
                        continue

                    if agg_steps:
                        keep = [c for c in [y] + agg_cols if c in group_df.columns]
                        yv   = _collapse(group_df[keep], y, agg_steps)
                    else:
                        unique_vals = group_df[y].dropna().unique()
                        if len(unique_vals) > 1:
                            print(f"[{title}] Warning: {len(group_df)} rows with differing "
                                  f"'{y}' for {lines_col}={line_val!r} x={xv!r} — "
                                  f"taking mean. Add agg or filter to disambiguate.")
                        yv = float(group_df[y].mean())
                    y_vals.append(yv)

                line_obj, = ax.plot(
                    x_vals, y_vals,
                    color=line_color.get(line_val),
                    linewidth=1.8,
                    marker="o", markersize=3,
                    label=str(line_val) if line_val is not None else None,
                )
                line_artists[line_val].append(line_obj)

        if bar_mode:
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide unused panels
    for extra in range(n_panels, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    if bar_mode:
        present_bars = {v: cs for v, cs in bar_containers.items() if cs}
        if present_bars:
            proxy_patches = [cs[0] for cs in present_bars.values()]
            labels        = [str(v) for v in present_bars.keys()]
            leg = fig.legend(proxy_patches, labels,
                             loc="outside right upper",
                             fontsize=8,
                             title=lines_col,
                             title_fontsize=9,
                             framealpha=0.9)
            patched = {lp: cs for lp, cs in zip(leg.get_patches(), present_bars.values())}
            make_interactive_legend(fig, leg, patched)
    else:
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
if "tag" not in df.columns:
    df["tag"] = df["combination"]

all_combos  = list(df["combination"].unique())
combo_color = dict(zip(all_combos, _distinct_colors(len(all_combos))))

all_tags  = list(df["tag"].unique())
tag_color = dict(zip(all_tags, _distinct_colors(len(all_tags))))

# ============================================================
# RENDER ALL PLOTS
# ============================================================
for cfg in PLOTS:
    make_plot(cfg, df, combo_color, tag_color)

plt.show()
