import math
import inspect
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIG_DPI = 120
plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 1.5})


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
        return tuple(min(1.0, max(0.0, c + amount)) for c in rgb)
    shifts = [0.0, 0.15, -0.15]
    return [_lighten(palette[i % len(palette)], shifts[(i // len(palette)) % len(shifts)]) for i in range(n)]


def make_interactive_legend(fig, leg, artist_map):
    """
    artist_map: {legend_handle → [data_artists]}
    Works with Line2D (line plots) or Patch/BarContainer (bar plots).
    Handles with empty data_artists lists are treated as non-interactive headers.
    Single-click → toggle visibility. Double-click → isolate (again → restore all).
    """
    def _get_vis(data_artists):
        if not data_artists:
            return True
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
        if not data_artists:
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
        return sorted(vals, key=lambda v: tuple(str(x) for x in v) if isinstance(v, tuple) else str(v))


def _as_cols(col):
    """Normalize a column spec (str | list | None) → list | None."""
    if col is None:
        return None
    return col if isinstance(col, list) else [col]


def _group_vals(df, cols):
    """Unique scalar or tuple values for a list of columns, sorted."""
    if len(cols) == 1:
        return _sorted_vals(df[cols[0]].unique())
    return _sorted_vals([tuple(r) for r in df[cols].drop_duplicates().values])


def _filter_group(df, cols, val):
    """Filter df to rows where cols == val (scalar or tuple)."""
    if not isinstance(val, tuple):
        return df[df[cols[0]] == val]
    mask = pd.Series(True, index=df.index)
    for c, v in zip(cols, val):
        mask &= df[c] == v
    return df[mask]


def _fmt_val(val):
    """Human-readable label for a scalar or composite (tuple) value."""
    if isinstance(val, tuple):
        return " / ".join("" if v == "-" else str(v) for v in val)
    return "" if val == "-" else str(val)


def _tick_label(val):
    if isinstance(val, tuple):
        return " / ".join("" if v == "-" else str(v) for v in val)
    return "" if val == "-" else str(val)


_DISPLAY_LABELS = {
    "RPEtrans": r"$\mathrm{RPE}_{trans}$",
    "RPErot":   r"$\mathrm{RPE}_{rot}$",
}
_TITLE_LABELS = {
    "RPEtrans": r"$\mathbf{RPE}_{\boldsymbol{trans}}$",
    "RPErot":   r"$\mathbf{RPE}_{\boldsymbol{rot}}$",
}


def _cols_label(cols, units=None):
    parts = []
    for c in cols:
        if c in _DISPLAY_LABELS:
            label = _DISPLAY_LABELS[c]
        else:
            label = c.replace("_", " ")
        if units and c in units:
            label += f" ({units[c]})"
        parts.append(label)
    return " / ".join(parts)


_LINESTYLES = ["-", "--", "-.", ":"]


def _coordinated_colors(df, cols, combo_color, tag_color):
    """Color by leftmost column; shade variations for subsequent columns."""
    if len(cols) == 1:
        vals = _group_vals(df, cols)
        if cols == ["method"]:
            return {v: combo_color.get(v) for v in vals}
        if cols == ["tag"]:
            return {v: tag_color.get(v) for v in vals}
        return dict(zip(vals, _distinct_colors(len(vals))))
    primary_vals = _sorted_vals(df[cols[0]].unique())
    if cols[0] == "method":
        base_colors = {pv: combo_color.get(pv) for pv in primary_vals}
    elif cols[0] == "tag":
        base_colors = {pv: tag_color.get(pv) for pv in primary_vals}
    else:
        base_colors = dict(zip(primary_vals, _distinct_colors(len(primary_vals))))
    result = {}
    for pv in primary_vals:
        sub = df[df[cols[0]] == pv]
        rest = cols[1:]
        if len(rest) == 1:
            sec_vals = _sorted_vals(sub[rest[0]].unique())
        else:
            sec_vals = _sorted_vals([tuple(r) for r in sub[rest].drop_duplicates().values])
        n = len(sec_vals)
        base = base_colors[pv] or (0.5, 0.5, 0.5)
        for i, sv in enumerate(sec_vals):
            shift = 0.18 - 0.36 * (i / max(n - 1, 1))
            shade = tuple(min(1.0, max(0.0, c + shift)) for c in base)
            key = (pv, sv) if len(cols) == 2 else (pv,) + (sv if isinstance(sv, tuple) else (sv,))
            result[key] = shade
    return result


def _linestyles_for(all_vals, cols):
    """Vary linestyle by secondary column within each primary group."""
    if len(cols) <= 1:
        return {v: "-" for v in all_vals}
    from collections import defaultdict
    groups = defaultdict(list)
    for v in all_vals:
        groups[v[0]].append(v)
    result = {}
    for group in groups.values():
        for i, lv in enumerate(group):
            result[lv] = _LINESTYLES[i % len(_LINESTYLES)]
    return result


def _bar_positions(x_vals, x_cols, gap=0.6):
    """y-positions with an extra gap between primary groups when multi-column."""
    if len(x_cols) <= 1:
        return list(range(len(x_vals)))
    positions = []
    pos = 0.0
    prev_primary = None
    for v in x_vals:
        primary = v[0]
        if prev_primary is not None and primary != prev_primary:
            pos += gap
        positions.append(pos)
        pos += 1.0
        prev_primary = primary
    return positions


def _lambda_label(fn):
    """Extract a readable label from a lambda or named callable."""
    if fn.__name__ != "<lambda>":
        return fn.__name__.replace("_", " ")
    try:
        src = inspect.getsource(fn).strip()
        m = re.search(r'lambda\s+\w+\s*:\s*(.+?)(?:,\s*$|\s*$)', src, re.DOTALL)
        if m:
            body = m.group(1).strip().rstrip(',').strip()
            body = re.sub(r'df\["([^"]+)"\]', r'\1', body)
            body = re.sub(r"df\['([^']+)'\]", r'\1', body)
            return body
    except Exception:
        pass
    return "derived"


def _auto_title(cfg, plain=False):
    y       = cfg.get("y", "mma_kps_mean")
    if callable(y):
        y_str = _lambda_label(y)
    elif plain:
        y_str = y if isinstance(y, str) else "derived"
    else:
        y_str = _TITLE_LABELS.get(y, y.replace("_", " "))
    x_cols  = _as_cols(cfg.get("x"))
    l_cols  = _as_cols(cfg.get("lines"))
    x_label = _cols_label(x_cols) if x_cols else (_cols_label(l_cols) if l_cols else "")
    return f"{y_str} vs {x_label}"


def _info_str(cfg, agg_steps):
    select = cfg.get("select", {})
    if not select:
        return "(no filters)"
    lines = []
    for col, spec in select.items():
        if not isinstance(spec, dict):
            spec = {"values": spec}
        parts = []
        v  = spec.get("values")
        fn = spec.get("fn")
        rng = spec.get("range")
        if v is not None:
            if isinstance(v, (list, np.ndarray)):
                parts.append(f"[{', '.join(str(x) for x in v)}]")
            else:
                parts.append(str(v))
        if fn is not None:
            parts.append(f"fn={fn}" + (f" range={rng}" if rng else ""))
        lines.append(f"  {col}: {'  |  '.join(parts)}")
    return "select\n" + "\n".join(lines)


def _to_numeric(values):
    """Convert values to float64 array, dropping '-' and other non-numeric entries."""
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=np.float64)


def _apply_fn(values, fn):
    arr = _to_numeric(values)
    if fn in ("auc", "mean"):
        return float(np.mean(arr)) if len(arr) else float("nan")
    elif fn == "std":
        return float(np.std(arr)) if len(arr) else float("nan")
    elif fn == "min":
        return float(np.min(arr)) if len(arr) else float("nan")
    elif fn == "max":
        return float(np.max(arr)) if len(arr) else float("nan")
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
            return float(pd.to_numeric(df[y_col], errors="coerce").mean())
        return float(pd.to_numeric(df[y_col], errors="coerce").iloc[0])

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


def make_plot(cfg, df, combo_color, tag_color, units=None):
    x_cols       = _as_cols(cfg.get("x"))
    lines_cols   = _as_cols(cfg.get("lines"))
    bar_mode     = lines_cols is None
    subplots_col = cfg.get("subplots")
    y_raw        = cfg.get("y", "mma_kps_mean")
    is_derived   = callable(y_raw)
    y            = "__derived_y__" if is_derived else y_raw
    select       = cfg.get("select", {})

    # ── Validate axis / lines / subplots column names ────────────────────────
    def _check_col(col, role):
        if col not in df.columns:
            raise ValueError(
                f"{role}: column '{col}' not found in CSV.\n"
                f"  Available: {list(df.columns)}"
            )

    if x_cols:
        for c in x_cols:
            _check_col(c, "x")
    if lines_cols:
        for c in lines_cols:
            _check_col(c, "lines")
    if subplots_col:
        _check_col(subplots_col, "subplots")

    # ── Parse select: build filter + ordered agg_steps ────────────────────
    agg_steps = []
    dfs = df.copy()
    for col, spec in select.items():
        if not isinstance(spec, dict):
            spec = {"values": spec}
        values = spec.get("values")
        if values is not None:
            if col not in dfs.columns:
                raise ValueError(
                    f"select: column '{col}' not found in CSV.\n"
                    f"  Available: {list(dfs.columns)}"
                )
            avail = sorted(dfs[col].unique(), key=str)
            n_before = len(dfs)
            if isinstance(values, (list, np.ndarray)):
                str_values = [str(v) for v in values]
                dfs = dfs[dfs[col].isin(values) | dfs[col].astype(str).isin(str_values)]
            elif isinstance(values, float) and math.isnan(values):
                dfs = dfs[dfs[col].isna()]
            else:
                dfs = dfs[(dfs[col] == values) | (dfs[col].astype(str) == str(values))]
            if dfs.empty and n_before > 0:
                raise ValueError(
                    f"select: '{col}' = {values!r} matched no rows.\n"
                    f"  Available values: {avail}"
                )
        if "fn" in spec:
            step = {"col": col, "fn": spec["fn"]}
            if "range" in spec:
                step["range"] = spec["range"]
            agg_steps.append(step)

    # ── Compute derived y column if y is a lambda ─────────────────────────────
    if is_derived:
        dfs[y] = y_raw(dfs)

    # ── AUC suffix for labels (from any select spec with fn="auc") ─────────────
    auc_suffix = ""
    for _col, _spec in select.items():
        if isinstance(_spec, dict) and _spec.get("fn") == "auc":
            _vals = _spec.get("values")
            if _vals is not None and len(_vals) > 0:
                try:
                    _max = max(float(v) for v in _vals)
                    _max_str = str(int(_max)) if _max == int(_max) else f"{_max:g}"
                except (TypeError, ValueError):
                    _max_str = str(max(_vals))
                _unit = (units.get(_col, "") if units else "")
                auc_suffix = f" AUC({_max_str}{_unit})"
                break

    # ── Auto-generate labels (all overridable) ────────────────────────────────
    def _ylabel_default():
        base = _lambda_label(y_raw) if is_derived else _DISPLAY_LABELS.get(y, y.replace("_", " "))
        if units and y in units:
            base += f" ({units[y]})"
        return base + auc_suffix

    _base_title       = _auto_title(cfg)
    _base_title_plain = _auto_title(cfg, plain=True)
    if auc_suffix:
        _base_title       = _base_title.replace(" vs ", f"{auc_suffix} vs ", 1)
        _base_title_plain = _base_title_plain.replace(" vs ", f"{auc_suffix} vs ", 1)
    title       = cfg.get("title", _base_title)
    title_plain = cfg.get("title", _base_title_plain)
    x_label = cfg.get("x_label", _cols_label(x_cols, units) if x_cols else (_cols_label(lines_cols, units) if lines_cols else ""))
    y_label = cfg.get("y_label", _ylabel_default())

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

    fig_width  = (14 if n_panels > 1 else 7) * 1.1
    fig_height = 5 * nrows

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(fig_width, fig_height),
                              dpi=FIG_DPI, squeeze=False,
                              layout="constrained")
    set_fig_title(fig, title_plain)
    fig.suptitle(title, fontweight="bold")

    # ── Toggleable info overlay ───────────────────────────────────────────────
    info_box = fig.text(0.5, 0.5, _info_str(cfg, agg_steps),
                        ha="center", va="center", fontsize=9,
                        fontfamily="monospace", visible=False, zorder=10,
                        bbox=dict(boxstyle="round,pad=0.7", facecolor="lightyellow",
                                  edgecolor="#888888", alpha=0.95),
                        transform=fig.transFigure)
    info_toggle = fig.text(0.005, 0.995, "ⓘ", ha="left", va="top",
                           fontsize=11, color="#aaaaaa", picker=True,
                           transform=fig.transFigure)

    def _on_info_pick(event):
        if event.artist is not info_toggle:
            return
        vis = not info_box.get_visible()
        info_box.set_visible(vis)
        info_toggle.set_color("black" if vis else "#aaaaaa")
        fig.canvas.draw()

    fig.canvas.mpl_connect("pick_event", _on_info_pick)

    # ── Color assignment ──────────────────────────────────────────────────────
    if bar_mode:
        x_color = _coordinated_colors(dfs, x_cols, combo_color, tag_color)
    else:
        all_line_vals = _group_vals(dfs, lines_cols)
        line_color    = _coordinated_colors(dfs, lines_cols, combo_color, tag_color)
        line_style    = _linestyles_for(all_line_vals, lines_cols)
        line_artists: dict = {v: [] for v in all_line_vals}

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
            x_vals   = _group_vals(panel_df, x_cols)
            bar_vals = []
            for xv in x_vals:
                group_df = _filter_group(panel_df, x_cols, xv)
                if group_df.empty:
                    bar_vals.append(float("nan"))
                elif agg_steps:
                    keep = [c for c in [y] + agg_cols if c in group_df.columns]
                    bar_vals.append(_collapse(group_df[keep], y, agg_steps))
                else:
                    bar_vals.append(float(pd.to_numeric(group_df[y], errors="coerce").mean()))
            colors    = [x_color.get(xv) for xv in x_vals]
            positions = _bar_positions(x_vals, x_cols)
            ax.barh(positions, bar_vals, height=0.97, color=colors)
            ax.set_yticks(positions)
            ax.set_yticklabels([_tick_label(v) for v in x_vals])
        else:
            x_vals = _group_vals(panel_df, x_cols)

            for line_val in all_line_vals:
                line_df = panel_df if line_val is None else _filter_group(panel_df, lines_cols, line_val)
                if line_df.empty:
                    continue

                y_vals = []
                for xv in x_vals:
                    group_df = _filter_group(line_df, x_cols, xv)
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
                                  f"'{y}' for {x_cols}={_fmt_val(xv)!r} lines={_fmt_val(line_val)!r} — "
                                  f"taking mean. Add agg or filter to disambiguate.")
                        yv = float(pd.to_numeric(group_df[y], errors="coerce").mean())
                    y_vals.append(yv)

                _x_has_str = any(isinstance(v, str) for v in x_vals)
                _x_has_num = any(not isinstance(v, (str, tuple)) for v in x_vals)
                _x_mixed   = _x_has_str and _x_has_num
                line_obj, = ax.plot(
                    [_fmt_val(v) if isinstance(v, tuple) else (str(v) if _x_mixed else v) for v in x_vals],
                    y_vals,
                    color=line_color.get(line_val),
                    linestyle=line_style[line_val],
                    linewidth=1.8,
                    marker="o", markersize=3,
                    label=_fmt_val(line_val) if line_val is not None else None,
                )
                line_artists[line_val].append(line_obj)

        if bar_mode:
            ax.set_xlabel(y_label)
            ax.set_ylabel(x_label)
            if y_raw == "RPEtrans":
                ax.set_xlim(left=0.015)
            elif y_raw == "RPErot":
                ax.set_xlim(left=0.065)
        else:
            ax.set_xlabel(x_label)
            if n_panels == 1:
                ax.set_ylabel(y_label)
        ax.grid(True, linestyle=":", alpha=0.5)

    # Hide unused panels
    for extra in range(n_panels, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    if not bar_mode and n_panels > 1:
        fig.supylabel(y_label)

    # ── Shared Y range across panels ───────────────────────────────────────────
    if n_panels > 1 and not bar_mode:
        visible_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)
                        if r * ncols + c < n_panels]
        all_ylims = [ax.get_ylim() for ax in visible_axes]
        g_ymin = min(yl[0] for yl in all_ylims)
        g_ymax = max(yl[1] for yl in all_ylims)
        for ax in visible_axes:
            ax.set_ylim(g_ymin, g_ymax)

    # ── Legend ────────────────────────────────────────────────────────────────
    if not bar_mode:
        present = {v: arts for v, arts in line_artists.items() if arts}
        if present:
            _fs = round(plt.rcParams['font.size'] * 0.8)

            if len(lines_cols) == 1:
                proxy_lines = [arts[0] for arts in present.values()]
                labels      = [_fmt_val(v) for v in present.keys()]
                leg = fig.legend(
                    proxy_lines, labels,
                    loc="outside right center",
                    fontsize=_fs,
                    title=_cols_label(lines_cols, units),
                    title_fontsize=_fs,
                    framealpha=0.9,
                )
                lined = {ll: arts for ll, arts in zip(leg.get_lines(), present.values())}
                make_interactive_legend(fig, leg, lined)

            else:
                # Primary group: one solid colored line per unique cols[0] value
                prim_vals = _sorted_vals(list({lv[0] for lv in present}))
                p_proxies, p_labels, p_map = [], [], {}
                for pv in prim_vals:
                    matching = [lv for lv in present if lv[0] == pv]
                    if not matching:
                        continue
                    color = line_color.get(matching[0])
                    proxy = plt.Line2D([0], [0], color=color, linewidth=1.8, linestyle="-")
                    arts  = [a for lv, al in present.items() if lv[0] == pv for a in al]
                    p_proxies.append(proxy)
                    p_labels.append(_fmt_val(pv))
                    p_map[proxy] = arts

                # Secondary group: one dark-gray line per unique cols[1:] value, linestyle varies
                sec_key  = (lambda lv: lv[1]) if len(lines_cols) == 2 else (lambda lv: lv[1:])
                sec_vals = _sorted_vals(list({sec_key(lv) for lv in present}))
                s_proxies, s_labels, s_map = [], [], {}
                for sv in sec_vals:
                    matching = [lv for lv in present if sec_key(lv) == sv]
                    if not matching:
                        continue
                    ls    = line_style.get(matching[0], "-")
                    proxy = plt.Line2D([0], [0], color="#444444", linewidth=1.8, linestyle=ls)
                    arts  = [a for lv, al in present.items() if sec_key(lv) == sv for a in al]
                    s_proxies.append(proxy)
                    s_labels.append(_fmt_val(sv))
                    s_map[proxy] = arts

                # Single combined legend with bold section headers separating the two groups
                p_hdr = plt.Line2D([0], [0], linewidth=0, color="none")
                s_sep = plt.Line2D([0], [0], linewidth=0, color="none")  # blank row for spacing
                s_hdr = plt.Line2D([0], [0], linewidth=0, color="none")

                n_p = len(p_proxies)
                # indices: 0=p_hdr, 1..n_p=p_proxies, n_p+1=s_sep, n_p+2=s_hdr, n_p+3..=s_proxies
                _hdr_indices = {0, n_p + 2}

                all_handles = [p_hdr] + p_proxies + [s_sep, s_hdr] + s_proxies
                all_labels  = (
                    [_cols_label([lines_cols[0]], units)] + p_labels +
                    ["", _cols_label(lines_cols[1:], units)] + s_labels
                )

                leg = fig.legend(
                    all_handles, all_labels,
                    loc="outside right center",
                    fontsize=_fs,
                    framealpha=0.9,
                )

                for i, txt in enumerate(leg.get_texts()):
                    if i in _hdr_indices:
                        txt.set_fontweight("bold")

                # Left-align header text with the legend frame's interior left edge
                _hdr_shifted = [False]
                def _align_legend_headers(event=None):
                    if _hdr_shifted[0]:
                        return
                    try:
                        renderer = fig.canvas.get_renderer()
                    except Exception:
                        return
                    texts     = leg.get_texts()
                    frame_ext = leg.get_frame().get_window_extent(renderer)
                    border_px = leg.borderpad * renderer.points_to_pixels(_fs)
                    target_x  = frame_ext.x0 + border_px
                    from matplotlib.transforms import Affine2D
                    for i, txt in enumerate(texts):
                        if i in _hdr_indices:
                            t_ext = txt.get_window_extent(renderer)
                            shift = target_x - t_ext.x0
                            txt.set_transform(Affine2D().translate(shift, 0) + txt.get_transform())
                    _hdr_shifted[0] = True
                    fig.canvas.draw_idle()

                fig.canvas.mpl_connect('draw_event', _align_legend_headers)

                _artist_map = {p_hdr: [], s_sep: [], s_hdr: []}
                _artist_map.update({p: p_map[p] for p in p_proxies})
                _artist_map.update({s: s_map[s] for s in s_proxies})
                lined = {ll: _artist_map[h]
                         for ll, h in zip(leg.get_lines(), all_handles)}
                make_interactive_legend(fig, leg, lined)


def _escape_latex(s):
    for old, new in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"),  ("#", r"\#"),  ("_", r"\_"),
        ("{",  r"\{"), ("}",  r"\}"), ("~", r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def _df_to_latex(df, units=None, float_precision=4, caption=""):
    cols    = list(df.columns)
    headers = [_escape_latex(_cols_label([c], units)) for c in cols]
    col_spec = "l" + "r" * (len(cols) - 1)

    def _fmt(val):
        if isinstance(val, float):
            return "-" if math.isnan(val) else f"{val:.{float_precision}f}"
        return _escape_latex(str(val))

    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[h]",
        r"\centering",
    ]
    if caption:
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    lines += [
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _apply_select_filters(df, select, title=""):
    dfs = df.copy()
    for col, spec in select.items():
        if not isinstance(spec, dict):
            spec = {"values": spec}
        values = spec.get("values")
        if values is None:
            continue
        if col not in dfs.columns:
            print(f"[{title}] select: '{col}' not in CSV — skipping filter.")
            continue
        if isinstance(values, (list, np.ndarray)):
            str_vals = [str(v) for v in values]
            dfs = dfs[dfs[col].isin(values) | dfs[col].astype(str).isin(str_vals)]
        elif isinstance(values, float) and math.isnan(values):
            dfs = dfs[dfs[col].isna()]
        else:
            dfs = dfs[(dfs[col] == values) | (dfs[col].astype(str) == str(values))]
    return dfs


def find_min_or_max(df, cfg, units=None):
    """
    Find the best row(s) for a metric, grouped by group_by columns.

    cfg keys:
        metric          — column to optimise
        mode            — "min" | "max"  (default "min")
        group_by        — list of columns to group by (e.g. ["Method"])
        agg_over        — columns to average out before picking best row
                          (e.g. ["Sequence"] for KITTI, ["distance_threshold",
                          "transformation"] for mma)
        agg_fn          — "mean" | "min" | "max"  (default "mean")
        select          — filter dict (same format as plot select, values only)
        show_cols       — columns to include in output (default: all remaining)
        title           — heading string
        export          — "latex" | "csv" | ["latex", "csv"] | None
        output_path     — base file path for export (no extension; required for csv)
        float_precision — decimal places for floats in output (default 4)
    """
    metric      = cfg["metric"]
    mode        = cfg.get("mode", "min")
    group_by    = cfg.get("group_by", [])
    agg_over    = cfg.get("agg_over", [])
    agg_fn      = cfg.get("agg_fn", "mean")
    select      = cfg.get("select", {})
    show_cols   = cfg.get("show_cols")
    title       = cfg.get("title",
                           f"{'Max' if mode == 'max' else 'Min'} {metric}"
                           + (f" per {', '.join(group_by)}" if group_by else ""))
    export      = cfg.get("export")
    output_path = cfg.get("output_path")
    float_prec  = cfg.get("float_precision", 4)

    if isinstance(export, str):
        export = [export]

    dfs = _apply_select_filters(df, select, title)
    if dfs.empty:
        print(f"[{title}] No data after filter — skipping.")
        return None

    # Average out agg_over dimensions so each config has one metric value
    if agg_over:
        id_cols = [c for c in dfs.columns if c not in agg_over and c != metric]
        dfs = dfs.copy()
        dfs[metric] = pd.to_numeric(dfs[metric], errors="coerce")
        dfs = dfs.groupby(id_cols, dropna=False, sort=False)[metric].agg(agg_fn).reset_index()

    # Find best row per group
    dfs = dfs.copy()
    dfs["__m__"] = pd.to_numeric(dfs[metric], errors="coerce")
    fn_name = "idxmin" if mode == "min" else "idxmax"

    if group_by:
        best_idx = dfs.groupby(group_by, dropna=False)["__m__"].agg(fn_name)
        result = dfs.loc[best_idx.dropna().astype(int).values].drop(columns=["__m__"])
    else:
        idx = int(getattr(dfs["__m__"], fn_name)())
        result = dfs.drop(columns=["__m__"]).iloc[[idx]]

    if show_cols:
        missing = [c for c in show_cols if c not in result.columns]
        if missing:
            print(f"[{title}] show_cols missing: {missing}")
        result = result[[c for c in show_cols if c in result.columns]]

    result = result.reset_index(drop=True)

    # Console output with units in headers
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    display_df = result.rename(columns={c: _cols_label([c], units) for c in result.columns})
    print(display_df.to_string(index=False))

    if export:
        if "latex" in export:
            latex_str = _df_to_latex(result, units=units, float_precision=float_prec, caption=title)
            print(f"\n--- LaTeX ---\n{latex_str}\n")
            if output_path:
                with open(output_path + ".tex", "w", encoding="utf-8") as f:
                    f.write(latex_str)
                print(f"[{title}] Written to {output_path}.tex")
        if "csv" in export:
            if output_path:
                display_df.to_csv(output_path + ".csv", index=False)
                print(f"[{title}] CSV written to {output_path}.csv")
            else:
                print(f"[{title}] export='csv' requires output_path.")

    return result


def run_display(csv_path, plots, units=None, table_queries=None):
    df = pd.read_csv(csv_path, na_values=[], keep_default_na=False, low_memory=False)
    method_col = "Method" if "Method" in df.columns else "method"
    df[method_col] = df[method_col].astype(str).str.strip()
    if "tag" not in df.columns and "Invariance configuration" not in df.columns:
        df[method_col] = df[method_col]

    all_combos  = list(df[method_col].unique())
    combo_color = dict(zip(all_combos, _distinct_colors(len(all_combos))))

    tag_col = next((c for c in df.columns if c in ("tag", "Invariance configuration",
                                                     "Response threshold")), None)
    all_tags  = list(df[tag_col].unique()) if tag_col else []
    tag_color = dict(zip(all_tags, _distinct_colors(len(all_tags))))

    for cfg in plots:
        make_plot(cfg, df, combo_color, tag_color, units=units)

    if table_queries:
        for cfg in table_queries:
            find_min_or_max(df, cfg, units=units)

    plt.show()
