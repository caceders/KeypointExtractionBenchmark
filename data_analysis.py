
import pandas as pd

# Replace with your CSV file path
file_path = 'output_distance.csv'

# Load the CSV
df = pd.read_csv(file_path)

# Columns to average
cols = [
    "Matching distance mAP [%]",
    "Matching average response mAP [%]",
    "Matching distinctiveness mAP [%]",
    "Verification distance AP [%]",
    "Verification average response AP [%]",
    "Verification distinctiveness AP [%]",
    "Retrieval distance mAP [%]",
    "Retrieval average response mAP [%]",
    "Retrieval distinctiveness mAP [%]",
    "distance-to-distinctiveness correlation",
    "distance-to-average response correlation"
]

# Check columns exist
for col in cols + ["combination"]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in the file.")

# Filter out rows where second descriptor is DAISY, FREAK, or SIFT 3.2
df_filtered = df[~df["combination"].str.split("+").str[1].isin(["DAISY", "FREAK", "SIFT 3.2"])]

# Compute and print averages for all columns
print("=== Averages after filtering ===")
for col in cols:
    avg_val = df_filtered[col].mean()
    print(f"{col}: {avg_val}")
