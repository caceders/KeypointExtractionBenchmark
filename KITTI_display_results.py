import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import re

###########################################
# =============== CONFIG ==================
###########################################

RUN_DIR = Path("KITTI/results/test_4_1.2")
CSV_PATH = RUN_DIR / "results.csv"

FILTER_WHITELIST_MODE = False

SUFFIX_FILTER = [
    #"PNP0.5",
    # "PNP1",
    # "PNP2",
    # "PNP3",
    # "PNP4",
    # "PNP6",
    # "PNP10",
]

DOWNSAMPLE_FILTER = [
    # 0,
    # # 1,
    # # 2,
    # # 4,
    0,
]

METHOD_FILTER = [
    "ORB+ORB"
]



ENABLE_SUFFIX_FILTER = True
ENABLE_DOWNSAMPLE_FILTER = True
ENABLE_METHOD_FILTER = True

SORT_BY = "method"        # or "metric"
METRIC_ASCENDING = True  # lower is better for errors
GROUP_BY = "method"
SHADE_SECTIONS = True

FIG_WIDTH = 10
ROW_HEIGHT = 0.35
FONT_FAMILY_YTICKS = "monospace"

###########################################
# ============ PARSING HELPERS ============
###########################################

def parse_method(method_name):
    """
    BASE_SUFFIX_DOWNSAMPLE
    """
    parts = method_name.split("_")
    method = "_".join(parts[0:-2])
    print(method)

    if len(parts) < 3:
        return method, None, None

    suffix = parts[-2]

    try:
        downsample = int(parts[-1])
    except ValueError:
        downsample = None

    return method, suffix, downsample


def natural_key(text):
    """
    Splits a string into text and integer chunks so that
    'PNP25' > 'PNP6'.
    """
    return [
        int(tok) if tok.isdigit() else tok
        for tok in re.split(r"(\d+)", text)
    ]


def method_natural_sort_key(method_name):
    """
    Natural-sort key for full method strings:
      - base (string)
      - suffix (natural sorted: text + numbers)
      - downsample (int)
    """

    _, suffix, downsample = parse_method(method_name)

    parts = method_name.split("_")
    base = "_".join(parts[:-2]) if len(parts) >= 3 else method_name

    downsample = downsample if downsample is not None else float("inf")

    return (
        base,
        natural_key(suffix) if suffix is not None else [],
        downsample,
    )


def passes_filter(method_name):
    method, suffix, downsample = parse_method(method_name)

    checks = []

    if ENABLE_SUFFIX_FILTER:
        if FILTER_WHITELIST_MODE:
            checks.append(suffix in SUFFIX_FILTER)
        else:
            checks.append(suffix not in SUFFIX_FILTER)

    if ENABLE_DOWNSAMPLE_FILTER:
        if FILTER_WHITELIST_MODE:
            checks.append(downsample in DOWNSAMPLE_FILTER)
        else:
            checks.append(downsample not in DOWNSAMPLE_FILTER)

    if ENABLE_METHOD_FILTER:
        if FILTER_WHITELIST_MODE:
            checks.append(method in METHOD_FILTER)
        else:
            checks.append(method not in METHOD_FILTER)

    return all(checks) if checks else True

###########################################
# ============== LOAD DATA ===============
###########################################

df = pd.read_csv(CSV_PATH)

df = df[df["method"].apply(passes_filter)].reset_index(drop=True)

if df.empty:
    raise RuntimeError("All data filtered out — check filter configuration.")

metric_cols = [
    c for c in df.columns
    if c not in ("sequence", "method", "max_frames")
    and np.issubdtype(df[c].dtype, np.number)
]

###########################################
# ============ COLOR MAP ==================
###########################################

groups = sorted(df[GROUP_BY].unique(), key=method_natural_sort_key)
cmap = plt.cm.tab20
group_color = {g: cmap(i % 20) for i, g in enumerate(groups)}


def set_fig_window_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        try:
            fig.canvas.set_window_title(title)
        except Exception:
            pass

###########################################
# ============ MAIN PLOTS =================
###########################################

for metric in metric_cols:

    dfm = (
        df.groupby("method", as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value"})
    )

    if SORT_BY == "metric":
        dfm = dfm.sort_values(
            "value", ascending=METRIC_ASCENDING, kind="mergesort"
        )
    else:
        dfm = dfm.sort_values(
            by="method",
            key=lambda s: s.map(method_natural_sort_key),
            kind="mergesort",
        )

    dfm = dfm.reset_index(drop=True)

    y_labels = dfm["method"].tolist()
    y_pos = np.arange(len(dfm))

    bar_colors = [group_color[m] for m in dfm["method"]]

    fig_h = max(5, len(dfm) * ROW_HEIGHT)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_h))
    set_fig_window_title(fig, metric)

    ax.barh(y_pos, dfm["value"], color=bar_colors)

    if SHADE_SECTIONS:
        seq = dfm[GROUP_BY].tolist()
        runs = []
        start = 0
        curr = seq[0]

        for i, v in enumerate(seq[1:], 1):
            if v != curr:
                runs.append((start, i, curr))
                start = i
                curr = v
        runs.append((start, len(seq), curr))

        xmax = ax.get_xlim()[1]

        for i, (a, b, g) in enumerate(runs):
            ax.axhspan(a - 0.5, b - 0.5,
                       color=group_color[g], alpha=0.08, lw=0)
            if i > 0:
                ax.axhline(a - 0.5, color="gray", ls="--", alpha=0.6)

            mid = (a + b - 1) / 2
            ax.text(
                xmax * 1.01, mid, g,
                ha="left", va="center",
                fontweight="bold",
                color=group_color[g]
            )

    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel(metric)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontfamily=FONT_FAMILY_YTICKS)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()

###########################################
plt.show()