import pandas as pd
import matplotlib.pyplot as plt

ASCENDING = True

# Load CSV
df = pd.read_csv("output.csv")

if not ASCENDING:
    # Identify numeric columns automatically
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Identify the x-axis
    if "combination" in df.columns:
        x = df["combination"]
    else:
        # fallback: index
        x = df.index.astype(str)

    # Create one separate bar chart window per numeric metric
    for col in numeric_cols:
        plt.figure()  # New window
        plt.bar(x, df[col])
        plt.title(col)
        plt.xlabel("combination")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")

    plt.show()

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
