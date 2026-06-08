import pandas as pd
import webbrowser
import tempfile
import os
import numpy as np

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "shared_results/KITTI/FINAL_baseline_complete/results.csv"

SEARCH_COLUMN = "RPE - translational"
# SEARCH_COLUMN = "RPE - rotational"
# SEARCH_COLUMN = "ATE"
# SEARCH_COLUMN = lambda df: pd.to_numeric(df["RPE - translational"], errors="coerce") + pd.to_numeric(df["RPE - rotational"], errors="coerce")

SEARCH_LABEL = None  # display name when SEARCH_COLUMN is a lambda (ignored for strings)

MINIMIZE = True    # True → find min, False → find max
SORT_BY_COLS = True  # True → sort by COLS first, then SEARCH_COLUMN; False → sort by SEARCH_COLUMN only

# Columns whose unique combinations define a "configuration".
# One best row is returned per unique combination of these columns.
COLS = ["Method"]

# Identity/config columns — used as group keys when collapsing a dimension with fn.
# Should include all columns that identify a configuration, excluding result metrics.
# KITTI: Method, Invariance configuration, Matching algorithm, Ratio test threshold,
#        Ratio test directionality, RANSAC threshold, Epipolar threshold,
#        Max features, Downsample level, Gaussian blur, Active frames
IDENTITY_COLS = [
    "Method", "Invariance configuration", "Matching algorithm", "Ratio test threshold",
    "Ratio test directionality", "RANSAC threshold", "Epipolar threshold",
    "Max features", "Downsample level", "Gaussian blur", "Active frames",
]

SELECT = {
    # "Method" : ["ORB", "SIFT"],
    # "Invariance configuration" : "Both",
    # "Matching algorithm" : ["MNN"],
    # "Ratio test threshold" : 1,
    # "Ratio test directionality" : "Unidirectional",
    # "RANSAC threshold" : 10,
    # "Epipolar threshold" : 1,
    # "Max features" : 1000,
    # "Downsample level" : 0,
    # "Gaussian blur" : 0,
    # [COL]: value                               — keep rows where col == value
    # [COL]: [val1, val2]                        — keep rows where col is in list
    # [COL]: {"vals": [...], "fn": "auc"}        — filter then collapse over col; renames metrics e.g. RPE AUC(10px)
    # [COL]: {"fn": "auc"}                       — collapse over all values of col
    # fn options: "auc" | "mean" | "std" | "min" | "max"
}

UNITS = {
    "Gaussian blur":       "σ",
    "RANSAC threshold":    "px",
    "ATE":                 "m",
    "RPE - translational": "m",
    "RPE - rotational":    "°",
}

EXPORT = None       # "latex" | "csv" | ["latex", "csv"] | None
OUTPUT_PATH = None  # base path without extension (required for csv file export)

# ============================================================
# RUN
# ============================================================
df = pd.read_csv(CSV_PATH, na_values=[], keep_default_na=False, low_memory=False)

_SEARCH_COL   = "__search__" if callable(SEARCH_COLUMN) else SEARCH_COLUMN
_SEARCH_LABEL = (SEARCH_LABEL or "score") if callable(SEARCH_COLUMN) else (SEARCH_LABEL or SEARCH_COLUMN)
_FN_MAP = {"auc": "mean", "mean": "mean", "std": "std", "min": "min", "max": "max"}

# Pass 1: value filters
for col, spec in SELECT.items():
    if isinstance(spec, dict):
        vals = spec.get("values", spec.get("vals", None))
        if vals is not None:
            df = df[df[col].isin(vals)]
    else:
        values = spec if isinstance(spec, list) else [spec]
        df = df[df[col].isin(values)]

# Compute derived search column before collapsing, drop rows where it is NaN
if callable(SEARCH_COLUMN):
    df[_SEARCH_COL] = pd.to_numeric(SEARCH_COLUMN(df), errors="coerce")
    df = df.dropna(subset=[_SEARCH_COL])

# Pass 2: collapses
for col, spec in SELECT.items():
    if isinstance(spec, dict) and spec.get("fn"):
        fn = spec["fn"]
        group_cols = [c for c in IDENTITY_COLS if c in df.columns and c != col]
        agg_cols = [c for c in df.columns if c not in group_cols and c != col]
        for c in agg_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.groupby(group_cols, dropna=False, as_index=False)[agg_cols].agg(_FN_MAP[fn])

if df.empty:
    print("No rows match the given filters.")
else:
    df[_SEARCH_COL] = pd.to_numeric(df[_SEARCH_COL], errors="coerce")
    agg = df.groupby(COLS, dropna=False)[_SEARCH_COL]
    sort_cols = COLS + [_SEARCH_COL] if SORT_BY_COLS else [_SEARCH_COL]
    ascending = ([True] * len(COLS) + [MINIMIZE]) if SORT_BY_COLS else [MINIMIZE]
    best_rows = (
        df.loc[(agg.idxmin() if MINIMIZE else agg.idxmax()).values]
          .sort_values(sort_cols, ascending=ascending)
          .reset_index(drop=True)
    )
    best_rows = best_rows.rename(columns={_SEARCH_COL: _SEARCH_LABEL})
    other_cols = [c for c in best_rows.columns if c not in COLS and c != _SEARCH_LABEL]
    best_rows = best_rows[COLS + [_SEARCH_LABEL] + other_cols]

    # AUC suffix: applied to metric columns when any SELECT entry uses fn="auc"
    auc_suffix = ""
    for _col, _spec in SELECT.items():
        if isinstance(_spec, dict) and _spec.get("fn") == "auc":
            _vals = _spec.get("vals", _spec.get("values", None))
            if _vals is not None:
                try:
                    _max_str = f"{max(_vals):.4g}"
                except (TypeError, ValueError):
                    _max_str = str(max(_vals))
                _unit = UNITS.get(_col, "") if UNITS else ""
                auc_suffix = f" AUC({_max_str}{_unit})"
                break

    _metric_cols = set(best_rows.columns) - set(IDENTITY_COLS) - set(COLS)

    def _col_hdr(c):
        name = f"{c} ({UNITS[c]})" if UNITS and c in UNITS else c
        if auc_suffix and c in _metric_cols:
            name = f"{name}{auc_suffix}"
        return name

    display_rows = best_rows.rename(columns={c: _col_hdr(c) for c in best_rows.columns})

    import math as _math, json
    def _esc(s):
        for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                         ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                         ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
            s = s.replace(old, new)
        return s
    _cols_l  = list(display_rows.columns)
    _col_spec = "l" + "r" * (len(_cols_l) - 1)
    def _fmtv(v):
        if isinstance(v, float): return "-" if _math.isnan(v) else f"{v:.4f}"
        return _esc(str(v))
    latex_str = "\n".join([
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[h]", r"\centering",
        f"\\caption{{{_esc(_col_hdr(_SEARCH_LABEL))}}}",
        f"\\begin{{tabular}}{{{_col_spec}}}", r"\toprule",
        " & ".join(_esc(c) for c in _cols_l) + r" \\", r"\midrule",
        *[" & ".join(_fmtv(display_rows.loc[i, c]) for c in _cols_l) + r" \\"
          for i in display_rows.index],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])

    _export_list = [EXPORT] if isinstance(EXPORT, str) else (EXPORT or [])
    if "latex" in _export_list:
        print("\n--- LaTeX ---\n" + latex_str + "\n")
        if OUTPUT_PATH:
            with open(OUTPUT_PATH + ".tex", "w", encoding="utf-8") as _f:
                _f.write(latex_str)
            print(f"Written to {OUTPUT_PATH}.tex")
    if "csv" in _export_list:
        if OUTPUT_PATH:
            display_rows.to_csv(OUTPUT_PATH + ".csv", index=False)
            print(f"CSV written to {OUTPUT_PATH}.csv")
        else:
            print(display_rows.to_csv(index=False))

    html          = display_rows.to_html(index=False)
    caption_js    = json.dumps(_col_hdr(_SEARCH_LABEL))
    search_orig_idx = list(display_rows.columns).index(_col_hdr(_SEARCH_LABEL))
    _latex_js     = r"""
    // ── Capture original grid ─────────────────────────────────────────────────
    var _origGrid = Array.from(rows).map(function(tr) {
        return Array.from(tr.cells).map(function(td) { return td.textContent.trim(); });
    });
    var _isT = false, _nSt0 = nSticky, _nC0 = nCols;

    // ── Visible columns in current display order ──────────────────────────────
    function _getVisCols() {
        return Array.from(cbDiv.querySelectorAll('input'))
            .filter(function(cb) { return cb.checked; })
            .map(function(cb) { return +cb.dataset.idx; });
    }

    // ── Rebuild table from _origGrid using current col order + visibility ─────
    function _rebuildTable() {
        var vc = _getVisCols();
        var h;
        if (!_isT) {
            h = '<thead><tr>';
            for (var i = 0; i < vc.length; i++) h += '<th>' + _origGrid[0][vc[i]] + '</th>';
            h += '</tr></thead><tbody>';
            for (var r = 1; r < _origGrid.length; r++) {
                h += '<tr>';
                for (var i = 0; i < vc.length; i++) h += '<td>' + _origGrid[r][vc[i]] + '</td>';
                h += '</tr>';
            }
        } else {
            if (!vc.length) { table.innerHTML = ''; rows = table.rows; return; }
            h = '<thead><tr>';
            for (var r = 0; r < _origGrid.length; r++) h += '<th>' + _origGrid[r][vc[0]] + '</th>';
            h += '</tr></thead><tbody>';
            for (var i = 1; i < vc.length; i++) {
                var ci = vc[i];
                h += '<tr><th>' + _origGrid[0][ci] + '</th>';
                for (var r = 1; r < _origGrid.length; r++) h += '<td>' + _origGrid[r][ci] + '</td>';
                h += '</tr>';
            }
        }
        table.innerHTML = h + '</tbody>';
        rows = table.rows;
    }

    // ── Bold best value in search column ─────────────────────────────────────
    function _applyBold() {
        var vc = _getVisCols();
        var pos = vc.indexOf(searchOrigIdx);
        if (pos < 0) return;
        if (!_isT) {
            var bestV = null;
            for (var r = 1; r < rows.length; r++) {
                var v = rows[r].cells[pos] ? parseFloat(rows[r].cells[pos].textContent) : NaN;
                if (!isNaN(v) && (bestV === null || (minimize ? v < bestV : v > bestV))) bestV = v;
            }
            for (var r = 1; r < rows.length; r++) {
                if (rows[r].cells[pos])
                    rows[r].cells[pos].style.fontWeight = (parseFloat(rows[r].cells[pos].textContent) === bestV) ? 'bold' : '';
            }
        } else {
            var brow = rows[pos];
            if (!brow) return;
            var bestV = null;
            for (var c = 1; c < brow.cells.length; c++) {
                var v = parseFloat(brow.cells[c].textContent);
                if (!isNaN(v) && (bestV === null || (minimize ? v < bestV : v > bestV))) bestV = v;
            }
            for (var c = 1; c < brow.cells.length; c++) {
                var v = parseFloat(brow.cells[c].textContent);
                brow.cells[c].style.fontWeight = (!isNaN(v) && v === bestV) ? 'bold' : '';
            }
        }
    }

    // ── Layout (sticky cols + group separators) ───────────────────────────────
    function _applyLayout() {
        var offsets = [];
        for (var c = 0; c < nSticky; c++) {
            offsets.push(c === 0 ? 0 : offsets[c-1] + rows[0].cells[c-1].getBoundingClientRect().width);
        }
        for (var r = 0; r < rows.length; r++) {
            var isHdr = rows[r].cells[0].tagName === 'TH';
            for (var c = 0; c < nSticky; c++) {
                var cell = rows[r].cells[c];
                cell.style.position = 'sticky';
                cell.style.left = offsets[c] + 'px';
                cell.style.zIndex = isHdr ? 3 : 2;
                if (!isHdr) cell.style.background = r % 2 === 0 ? '#f9f9f9' : '#fff';
            }
        }
        if (sortByCols && nCols > 0) {
            var bw = [4, 2.5, 1.5], bc = ['#222', '#666', '#999'];
            for (var r = 2; r < rows.length; r++) {
                var prev = rows[r-1], curr = rows[r], sl = -1;
                for (var c = 0; c < nCols; c++) {
                    if (prev.cells[c] && curr.cells[c] && prev.cells[c].textContent !== curr.cells[c].textContent) { sl = c; break; }
                }
                if (sl >= 0) {
                    for (var c = 0; c < curr.cells.length; c++)
                        curr.cells[c].style.borderTop = (bw[sl]||1.5) + 'px solid ' + (bc[sl]||'#999');
                    if (sl === 0) { rows[r-1].style.paddingBottom = '4px'; curr.style.paddingTop = '4px'; }
                }
            }
        }
    }
    _applyLayout();

    // ── Column checkboxes with ▲▼ reorder buttons ─────────────────────────────
    var cbDiv = document.getElementById('col-checks');
    function _update() { _rebuildTable(); _applyBold(); _applyLayout(); }
    function _makeEntry(name, idx) {
        var lbl = document.createElement('label');
        lbl.dataset.idx = idx;
        var cb = document.createElement('input');
        cb.type = 'checkbox'; cb.checked = true; cb.dataset.idx = idx;
        cb.addEventListener('change', _update);
        var up = document.createElement('button');
        up.className = 'col-arrow'; up.type = 'button'; up.textContent = '▲';
        up.addEventListener('click', function() {
            var prev = lbl.previousElementSibling;
            if (prev) { cbDiv.insertBefore(lbl, prev); _update(); }
        });
        var dn = document.createElement('button');
        dn.className = 'col-arrow'; dn.type = 'button'; dn.textContent = '▼';
        dn.addEventListener('click', function() {
            var next = lbl.nextElementSibling;
            if (next) { cbDiv.insertBefore(next, lbl); _update(); }
        });
        lbl.appendChild(cb);
        lbl.appendChild(document.createTextNode(' ' + name));
        lbl.appendChild(up);
        lbl.appendChild(dn);
        return lbl;
    }
    _origGrid[0].forEach(function(name, i) { cbDiv.appendChild(_makeEntry(name, i)); });
    _applyBold();

    // ── Transpose toggle ──────────────────────────────────────────────────────
    document.getElementById('transpose-btn').addEventListener('click', function() {
        _isT = !_isT;
        nSticky = _isT ? 1 : _nSt0;
        nCols   = _isT ? 0 : _nC0;
        _rebuildTable(); _applyBold(); _applyLayout();
    });

    // ── Generate LaTeX (current col order, normal orientation) ────────────────
    document.getElementById('gen-latex').addEventListener('click', function() {
        var vc = _getVisCols();
        if (!vc.length) return;
        function esc(s) {
            return s.replace(/\\/g, '\\textbackslash{}')
                    .replace(/&/g, '\\&').replace(/%/g, '\\%').replace(/\$/g, '\\$')
                    .replace(/#/g, '\\#').replace(/_/g, '\\_')
                    .replace(/\{/g, '\\{').replace(/\}/g, '\\}');
        }
        var hdrs = vc.map(function(i) { return _origGrid[0][i]; });
        var body = [];
        for (var r = 1; r < _origGrid.length; r++)
            body.push(vc.map(function(i) { return _origGrid[r][i]; }));
        var colSpec = 'l' + 'r'.repeat(vc.length - 1);
        var lines = [
            '% requires \\usepackage{booktabs}',
            '\\begin{table}[h]', '\\centering',
            '\\caption{' + esc(caption) + '}',
            '\\begin{tabular}{' + colSpec + '}', '\\toprule',
            hdrs.map(esc).join(' & ') + ' \\\\', '\\midrule',
        ];
        body.forEach(function(row) { lines.push(row.map(esc).join(' & ') + ' \\\\'); });
        lines.push('\\bottomrule', '\\end{tabular}', '\\end{table}');
        var ta = document.getElementById('latex-out');
        ta.value = lines.join('\n');
        ta.style.display = 'block';
        ta.select();
    });"""

    def _fmt_val(v):
        import numpy as np
        if isinstance(v, np.ndarray):
            return f"[{v[0]}..{v[-1]}]" if len(v) > 4 else str(list(v))
        if isinstance(v, list) and len(v) > 4:
            return f"[{v[0]}..{v[-1]}]"
        return repr(v)
    config_lines = []
    config_lines.append(f"csv:    {CSV_PATH}")
    config_lines.append(f"search: {_col_hdr(_SEARCH_LABEL)}  ({'min' if MINIMIZE else 'max'})")
    config_lines.append(f"cols:   {COLS}")
    for col, spec in SELECT.items():
        if isinstance(spec, dict):
            vals = spec.get("vals", spec.get("values", None))
            fn   = spec.get("fn", None)
            val_str = _fmt_val(vals) if vals is not None else "*"
            config_lines.append(f"  {col} = {val_str}  →  {fn}")
        else:
            config_lines.append(f"  {col} = {_fmt_val(spec)}")
    config_html = "\n".join(config_lines)

    n_sticky = len(COLS) + 1
    n_cols = len(COLS)
    minimize_js = "true" if MINIMIZE else "false"
    sort_by_cols_js = "true" if SORT_BY_COLS else "false"
    styled = f"""<!DOCTYPE html>
<html><head><style>
body {{ font-family: monospace; padding: 1em; }}
div.wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; white-space: nowrap; }}
th {{ background: #eee; }}
tr:nth-child(even) td {{ background: #f9f9f9; }}
tr:nth-child(odd) td {{ background: #fff; }}
button {{ font-family: monospace; margin-bottom: 0.6em; padding: 4px 12px; cursor: pointer; }}
pre.config {{ background: #f4f4f4; border: 1px solid #ddd; padding: 0.6em 1em; margin-bottom: 1em; line-height: 1.5; }}
#col-checks {{ display: flex; flex-direction: column; gap: 0.2em; margin-bottom: 0.6em; padding: 0.5em 0.8em; background: #f4f4f4; border: 1px solid #ddd; width: fit-content; }}
#col-checks label {{ display: flex; align-items: center; gap: 0.35em; cursor: default; }}
.col-arrow {{ font-size: 0.6em; padding: 1px 4px; cursor: pointer; background: none; border: 1px solid #bbb; border-radius: 2px; margin-left: 2px; line-height: 1.6; }}
.col-arrow:hover {{ background: #ddd; }}
#latex-out {{ display: none; width: 100%; box-sizing: border-box; height: 180px; font-family: monospace; font-size: 0.85em; margin-top: 0.6em; padding: 0.5em; border: 1px solid #aaa; }}
</style></head><body>
<pre class="config">{config_html}</pre>
<div id="col-checks"></div>
<button id="transpose-btn">&#8645; Transpose</button>
<button id="gen-latex">Generate LaTeX</button>
<textarea id="latex-out" spellcheck="false"></textarea>
<div class="wrap">{html}</div><script>
(function() {{
    var table = document.querySelector('table');
    var rows = table.rows;
    var nSticky = {n_sticky};
    var nCols = {n_cols};
    var sortByCols = {sort_by_cols_js};
    var caption = {caption_js};
    var searchOrigIdx = {search_orig_idx};
    var minimize = {minimize_js};
{_latex_js}
}})();
</script></body></html>"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    tmp.write(styled)
    tmp.close()
    webbrowser.open(f"file://{os.path.abspath(tmp.name)}")
