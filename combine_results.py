"""
Generates shared_results/combined/results.csv.gz by merging:
  - shared_results/modern/FINAL_baseline_complete.csv   (mma data)
  - shared_results/KITTI/FINAL_baseline_complete/results.csv (KITTI data)

Filters applied before joining (constants → removed from output):
  - mma:   Visibility filter == False  (column dropped)
  - KITTI: Epipolar threshold == 1     (column dropped)
  - KITTI: Active frames == "0-1000"   (column dropped)

All other dimensions (Distance error threshold 1-20, Image transformation type) are
kept as separate rows so find_min_or_max_combined.py can collapse them as configured.

Output: ~7.2 M rows × 44 cols, ~96 MB gzip.
"""

import pandas as pd
import os

MMA_PATH    = "shared_results/modern/FINAL_baseline_complete.csv"
KITTI_PATH  = "shared_results/KITTI/FINAL_baseline_complete/results.csv"
OUTPUT_PATH = "shared_results/combined/results.csv.gz"

JOIN_KEYS = [
    "Method", "Invariance configuration", "Matching algorithm",
    "Ratio test threshold", "Ratio test directionality",
    "RANSAC threshold", "Downsample level", "Gaussian blur", "Max features",
]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading CSVs...")
mma   = pd.read_csv(MMA_PATH,   na_values=[], keep_default_na=False, low_memory=False)
kitti = pd.read_csv(KITTI_PATH, na_values=[], keep_default_na=False, low_memory=False)
print(f"  mma:   {len(mma):,} rows × {len(mma.columns)} cols")
print(f"  KITTI: {len(kitti):,} rows × {len(kitti.columns)} cols")

# ── Filter and drop constant columns ─────────────────────────────────────────
mma_f = mma[mma["Visibility filter"] == False].copy()
mma_f = mma_f.drop(columns=["Visibility filter"])
mma_f["Ratio test threshold"] = pd.to_numeric(mma_f["Ratio test threshold"], errors="coerce")

kitti_f = kitti[pd.to_numeric(kitti["Epipolar threshold"], errors="coerce") == 1.0].copy()
kitti_f = kitti_f.drop(columns=["Epipolar threshold", "Active frames"], errors="ignore")

print(f"  mma after filter:   {len(mma_f):,} rows")
print(f"  KITTI after filter: {len(kitti_f):,} rows")

# ── Resolve column name conflict ──────────────────────────────────────────────
mma_f   = mma_f.rename(columns={"Average number of features": "Average number of features (mma)"})
kitti_f = kitti_f.rename(columns={"Average number of features": "Average number of features (KITTI)"})

# ── Inner join ────────────────────────────────────────────────────────────────
print("Joining...")
combined = mma_f.merge(kitti_f, on=JOIN_KEYS, how="inner")
print(f"  Combined: {len(combined):,} rows × {len(combined.columns)} cols")

# ── Save (gzip for manageable file size) ─────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
print(f"Writing {OUTPUT_PATH} (gzip)...")
combined.to_csv(OUTPUT_PATH, index=False, float_format="%.6g", compression="gzip")
size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"\nWritten to {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print(f"Columns ({len(combined.columns)}): {list(combined.columns)}")
