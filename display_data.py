import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ASCENDING = False
df = pd.read_csv("output_big_overlap_test.csv")
if not ASCENDING:
    # Load CSV
    # Sort DataFrame by 'combination' if it exists

    # Ensure 'combination' exists
    if "combination" not in df.columns:
        # Fallback to index as 'combination' (string)
        df = df.copy()
        df["combination"] = df.index.astype(str)

    # 1) Extract detector type (first token before '+')
    #    e.g. "ORB+BRISK+0.05" -> "ORB"
    df = df.copy()
    df["detector"] = df["combination"].str.split("+").str[0]

    # 2) Sort alphabetically: by detector, then by combination
    df = df.sort_values(by=["detector", "combination"], ascending=[True, True]).reset_index(drop=True)

    # 3) Identify numeric columns automatically
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to plot.")

    # 4) Prepare x positions and colors per detector
    x_labels = df["combination"].values
    x_pos = np.arange(len(df))

    # Pick distinct colors for up to 3 detectors
    unique_detectors = sorted(df["detector"].unique())
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    color_map = {det: palette[i % len(palette)] for i, det in enumerate(unique_detectors)}

    bar_colors = [color_map[d] for d in df["detector"].values]

    # Compute section boundaries for visual separation
    # Since we've sorted by detector, each detector's rows are contiguous.
    section_starts = []
    section_ends = []
    cursor = 0
    for det in unique_detectors:
        count = int((df["detector"] == det).sum())
        section_starts.append(cursor)
        section_ends.append(cursor + count)
        cursor += count

    # 5) Plot one bar chart per numeric metric with colored sections and separators
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.14), 6), dpi=120)

        # Bars colored by detector
        ax.bar(x_pos, df[col].values, color=bar_colors)

        # Vertical separators between sections + light background bands per detector
        for i, det in enumerate(unique_detectors):
            start = section_starts[i]
            end = section_ends[i]
            # Light band to highlight section
            ax.axvspan(start - 0.5, end - 0.5, color=color_map[det], alpha=0.08, lw=0)
            # Separator line (except before the first section)
            if i > 0:
                ax.axvline(start - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            # Label the detector name centered above its section
            mid = (start + end - 1) / 2.0
            ax.text(mid, ax.get_ylim()[1] * 1.02, det, ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=color_map[det], rotation=0,
                    transform=ax.get_transform())  # stays in data coords

        # Axis & labels
        ax.set_title(f"{col} by combination (sections = detectors)", fontsize=12, fontweight="bold")
        ax.set_xlabel("combination")
        ax.set_ylabel(col)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

        # Legend mapping detector -> color
        legend_handles = [mpatches.Patch(color=color_map[det], label=det) for det in unique_detectors]
        ax.legend(handles=legend_handles, title="Detector", loc="upper left", bbox_to_anchor=(1.02, 1.0))

        # Tight layout for readability
        plt.tight_layout()

        plt.show()

    
    # if "combination" in df.columns:
    #     df = df.sort_values(by="combination")
    #     x = df["combination"]
    # else:
    #     # fallback: index
    #     x = df.index.astype(str)

    # # Identify numeric columns automatically
    # numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # # Create one separate bar chart window per numeric metric
    # for col in numeric_cols:
    #     plt.figure()  # New window
    #     plt.bar(x, df[col])
    #     plt.title(col)
    #     plt.xlabel("combination")
    #     plt.ylabel(col)
    #     plt.xticks(rotation=45)
    #     plt.grid(True, axis="y")

    # plt.show()

else:

    # Identify numeric columns automatically
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Identify the x-axis
    if "combination" in df.columns:
        x_col = "combination"
    else:
        x_col = None

    for col in numeric_cols:
        # Sort by this column
        if x_col:
            df_sorted = df.sort_values(col)
            x = df_sorted[x_col]
        else:
            df_sorted = df.sort_values(col).reset_index(drop=True)
            x = df_sorted.index.astype(str)

        plt.figure()
        plt.bar(x, df_sorted[col])
        plt.title(col)
        plt.xlabel("combination" if x_col else "index")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")

    plt.show()
