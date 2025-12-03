import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("output.csv")

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