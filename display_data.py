import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ASCENDING = False
COLORED_SECTIONS_ALPHABETICAL = True
df = pd.read_csv("output_selected.csv")

if not ASCENDING:

    if COLORED_SECTIONS_ALPHABETICAL:
        if "combination" not in df.columns:
            df = df.copy()
            df["combination"] = df.index.astype(str)

        df = df.copy()
        df["detector"] = df["combination"].str.split("+").str[0]

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found to plot.")

        # ---------- ALIGN LABELS ON FIRST "+" AND FIX RIGHT-PART COLUMN ----------
        raw_labels = df["combination"].astype(str).tolist()
        split_labels = [lbl.split("+", 1) for lbl in raw_labels]

        left_parts = [p[0] for p in split_labels]
        right_parts = [p[1] if len(p) > 1 else "" for p in split_labels]

        max_left = max(len(s) for s in left_parts)
        max_right = max(len(s) for s in right_parts)

        aligned_labels = [
            left.ljust(max_left) + " + " + right.ljust(max_right)
            for left, right in zip(left_parts, right_parts)
        ]
        # -------------------------------------------------------------------------

        x_labels = aligned_labels
        x_pos = np.arange(len(df))

        unique_detectors = sorted(df["detector"].unique())
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        color_map = {det: palette[i % len(palette)] for i, det in enumerate(unique_detectors)}
        bar_colors = [color_map[d] for d in df["detector"].values]

        section_starts = []
        section_ends = []
        cursor = 0
        for det in unique_detectors:
            cnt = int((df["detector"] == det).sum())
            section_starts.append(cursor)
            section_ends.append(cursor + cnt)
            cursor += cnt

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.14)), dpi=120)
            fig.canvas.manager.set_window_title(col)  # <-- window title = metric

            ax.barh(x_pos, df[col].values, color=bar_colors)

            # Section shading + separators
            for i, det in enumerate(unique_detectors):
                start = section_starts[i]
                end = section_ends[i]

                ax.axhspan(start - 0.5, end - 0.5,
                           color=color_map[det], alpha=0.08, lw=0)

                if i > 0:
                    ax.axhline(start - 0.5, color="gray",
                               linestyle="--", linewidth=1, alpha=0.7)

                mid = (start + end - 1) / 2.0
                ax.text(ax.get_xlim()[1] * 1.01, mid, det,
                        va="center", ha="left",
                        fontsize=11, fontweight="bold",
                        color=color_map[det])

            ax.set_title(f"{col} by combination (sections = detectors)",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("combination")
            ax.set_xlabel(col)

            ax.set_yticks(x_pos)
            ax.set_yticklabels(x_labels, fontfamily="monospace")  # aligned labels

            ax.grid(True, axis="x", linestyle=":", alpha=0.5)

            legend_handles = [
                mpatches.Patch(color=color_map[det], label=det)
                for det in unique_detectors
            ]
            ax.legend(handles=legend_handles, title="Detector",
                      loc="upper left", bbox_to_anchor=(1.02, 1.0))

            plt.tight_layout()
        plt.show()

    else:
        if "combination" in df.columns:
            df = df.sort_values(by="combination")
            x = df["combination"]
        else:
            x = df.index.astype(str)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        for col in numeric_cols:
            plt.figure()
            plt.gcf().canvas.manager.set_window_title(col)  # <-- window title = metric
            plt.barh(x, df[col])
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel("combination")
            plt.grid(True, axis="x")

        plt.show()

else:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if "combination" in df.columns:
        x_col = "combination"
    else:
        x_col = None

    for col in numeric_cols:
        if x_col:
            df_sorted = df.sort_values(col)
            x = df_sorted[x_col]
        else:
            df_sorted = df.sort_values(col).reset_index(drop=True)
            x = df_sorted.index.astype(str)

        plt.figure()
        plt.gcf().canvas.manager.set_window_title(col)  # <-- window title = metric
        plt.barh(x, df_sorted[col])
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("combination" if x_col else "index")
        plt.grid(True, axis="x")

    plt.show()
