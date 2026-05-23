import pandas as pd
import webbrowser
import tempfile
import os
import numpy as np

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "shared_results\modern\FINAL_baseline.csv"
# CSV_PATH = "shared_results/KITTI/pre_baseline_test/results.csv"

SEARCH_COLUMN = "mAP"
# SEARCH_COLUMN = "RPE1_trans_RMSE"

SEARCH_LABEL = None  # display name when SEARCH_COLUMN is a lambda (ignored for strings)


MINIMIZE = False   # True → find min, False → find max
SORT_BY_COLS = True  # True → sort by COLS first, then SEARCH_COLUMN; False → sort by SEARCH_COLUMN only

# Columns whose unique combinations define a "configuration".
# One best row is returned per unique combination of these columns.
COLS = ["method", "visibility_filter"]

# Identity/config columns — used as group keys when collapsing a dimension with fn.
# Should include all columns that identify a configuration, excluding result metrics.
# mma:   method, tag, matcher, ratio_threshold, mnn_bidirectional, ransac_threshold,
#        max_keypoints, downsample_level, initial_sigma, visibility_filter,
#        transformation, distance_threshold
# KITTI: method, tag, matcher, ratio_threshold, mnn_bidirectional, ransac_threshold,
#        epipolar_threshold, max_keypoints, downsample_level, initial_sigma, active_frames
IDENTITY_COLS = [
    "method", "tag", "matcher", "ratio_threshold", "mnn_bidirectional",
    "ransac_threshold", "max_keypoints", "downsample_level", "initial_sigma",
    "visibility_filter", "transformation", "distance_threshold",
]

SELECT = {
    "transformation" : {"vals" : ["illumination", "viewpoint"], "fn" : "mean"},
    "distance_threshold": {"vals" : np.arange(0,11), "fn" : "auc"},
    # "distance_threshold" : 5,
    # "tag" : "low_threshold", 
    # "max_keypoints" : 500,
    # "visibility_filter" : False,
    # "matcher" : "NN",
    # "ratio_threshold" : 1,
    "matcher" : "KEEM",
    # "matcher" : "NN",
    # "ratio_threshold" : 1,
    # [COL]: value                               — keep rows where col == value
    # [COL]: [val1, val2]                        — keep rows where col is in list
    # [COL]: {"vals": [...], "fn": "auc"}        — filter then collapse over col
    # [COL]: {"fn": "auc"}                       — collapse over all values of col
    # fn options: "auc" | "mean" | "std" | "min" | "max"
    # "matcher": "KEEM",
    # "mnn_bidirectional": [True, "-"],
}

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
            str_vals = [str(v) for v in vals]
            df = df[df[col].isin(vals) | df[col].astype(str).isin(str_vals)]
    else:
        values = spec if isinstance(spec, list) else [spec]
        str_values = [str(v) for v in values]
        df = df[df[col].isin(values) | df[col].astype(str).isin(str_values)]

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
    best_rows = (
        df.loc[(agg.idxmin() if MINIMIZE else agg.idxmax()).values]
          .sort_values(sort_cols, ascending=([True] * len(COLS) + [MINIMIZE]) if SORT_BY_COLS else MINIMIZE)
          .reset_index(drop=True)
    )
    best_rows["#"] = best_rows[_SEARCH_COL].rank(ascending=MINIMIZE, method="min").astype(int)
    best_rows = best_rows.rename(columns={_SEARCH_COL: _SEARCH_LABEL})
    other_cols = [c for c in best_rows.columns if c not in COLS and c != _SEARCH_LABEL and c != "#"]
    best_rows = best_rows[["#"] + COLS + [_SEARCH_LABEL] + other_cols]
    html = best_rows.to_html(index=False)
    n_sticky = len(COLS) + 2  # "#" + COLS + SEARCH_COLUMN
    n_cols = len(COLS)        # number of COLS for group separator detection
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
</style></head><body><div class="wrap">{html}</div><script>
(function() {{
    var table = document.querySelector('table');
    var rows = table.rows;
    var nSticky = {n_sticky};
    var nCols = {n_cols};
    var sortByCols = {sort_by_cols_js};
    var offsets = [];
    for (var c = 0; c < nSticky; c++) {{
        offsets.push(c === 0 ? 0 : offsets[c - 1] + rows[0].cells[c - 1].getBoundingClientRect().width);
    }}
    for (var r = 0; r < rows.length; r++) {{
        var isHeader = rows[r].cells[0].tagName === 'TH';
        for (var c = 0; c < nSticky; c++) {{
            var cell = rows[r].cells[c];
            cell.style.position = 'sticky';
            cell.style.left = offsets[c] + 'px';
            cell.style.zIndex = isHeader ? 3 : 2;
            if (!isHeader) {{
                cell.style.background = r % 2 === 0 ? '#f9f9f9' : '#fff';
            }}
        }}
    }}
    if (sortByCols && nCols > 0) {{
        var borderWidths = [4, 2.5, 1.5];
        var borderColors = ['#222', '#666', '#999'];
        var colsOffset = 1;  // skip leading "#" column
        for (var r = 2; r < rows.length; r++) {{
            var prev = rows[r - 1];
            var curr = rows[r];
            var splitLevel = -1;
            for (var c = 0; c < nCols; c++) {{
                if (prev.cells[colsOffset + c].textContent !== curr.cells[colsOffset + c].textContent) {{
                    splitLevel = c;
                    break;
                }}
            }}
            if (splitLevel >= 0) {{
                var width = (borderWidths[splitLevel] || 1.5);
                var color = (borderColors[splitLevel] || '#999');
                for (var c = 0; c < curr.cells.length; c++) {{
                    curr.cells[c].style.borderTop = width + 'px solid ' + color;
                }}
                if (splitLevel === 0) {{
                    rows[r - 1].style.paddingBottom = '4px';
                    curr.style.paddingTop = '4px';
                }}
            }}
        }}
    }}
}})();
</script></body></html>"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    tmp.write(styled)
    tmp.close()
    webbrowser.open(f"file://{os.path.abspath(tmp.name)}")
