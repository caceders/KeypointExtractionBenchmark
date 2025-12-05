
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_stats_multi_windows(csv_path="output_low_sigma_limited.csv"):
    df = pd.read_csv(csv_path)

    if "combination" not in df.columns:
        df["combination"] = [f"comb_{i}" for i in range(len(df))]

    metric_cols = [c for c in df.columns if c != "combination"]

    for col in metric_cols:
        plt.figure(figsize=(10, 5))
        plt.bar(df["combination"], df[col], color="#4C72B0")
        plt.title(f"{col} by combination")
        plt.ylabel(col)
        plt.xlabel("Combination")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show(block=False)  # <-- This is the trick: non-blocking show

    # Keep all windows open until user closes them
    input("Press Enter to close all plots...")

if __name__ == "__main__":
    plot_all_stats_multi_windows()

