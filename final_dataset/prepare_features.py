import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# -------------------------------------------------
# Paths / Load data
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
cleaned_path = BASE_DIR / "cleaned_data.csv"

df = pd.read_csv(cleaned_path, low_memory=False)
print("\nLoaded:", df.shape)

# -------------------------------------------------
# Sample to 50k rows (if more)
# -------------------------------------------------
TARGET_SAMPLES = 50000
if len(df) > TARGET_SAMPLES:
    df = df.sample(TARGET_SAMPLES, random_state=42)

print("Sampled:", df.shape)

# Reset index so df and any derived matrices stay aligned
df = df.reset_index(drop=True)

# -------------------------------------------------
# Numeric columns & scaling
# -------------------------------------------------
feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Drop ID-like columns if present
drop_ids = ["id", "participant_id", "PID"]
feature_cols = [c for c in feature_cols if c not in drop_ids]

print("\nNUMERIC feature columns:")
for c in feature_cols:
    print("  -", c)

# Standardize numeric columns
scaler = StandardScaler()
X_z = pd.DataFrame(
    scaler.fit_transform(df[feature_cols]),
    columns=feature_cols,
    index=df.index  # keep same index
)

# Add z-score columns to df for inspection and modeling
for c in feature_cols:
    df[c + "_z"] = X_z[c]

# -------------------------------------------------
# Domain grouping attempt (for generic datasets)
# -------------------------------------------------
memory_feats    = [c for c in feature_cols if any(k in c.lower() for k in ["mem", "delayed", "story"])]
attention_feats = [c for c in feature_cols if any(k in c.lower() for k in ["att", "digit", "focus"])]
speed_feats     = [c for c in feature_cols if any(k in c.lower() for k in ["speed", "trail", "reaction"])]
reasoning_feats = [c for c in feature_cols if any(k in c.lower() for k in ["reason", "logic", "problem"])]

print("\nDetected domain groups:")
print("Memory:", memory_feats)
print("Attention:", attention_feats)
print("Speed:", speed_feats)
print("Reasoning:", reasoning_feats)

# -------------------------------------------------
# CHI computation
# -------------------------------------------------
if (
    len(memory_feats) > 0 and
    len(attention_feats) > 0 and
    len(speed_feats) > 0 and
    len(reasoning_feats) > 0
):
    print("\nUsing domain-based CHI computation.")
    df["Memory_Index"]    = X_z[memory_feats].mean(axis=1)
    df["Attention_Index"] = X_z[attention_feats].mean(axis=1)
    df["Speed_Index"]     = X_z[speed_feats].mean(axis=1)
    df["Reasoning_Index"] = X_z[reasoning_feats].mean(axis=1)

    df["CHI"] = (
        0.30 * df["Memory_Index"] +
        0.25 * df["Attention_Index"] +
        0.20 * df["Speed_Index"] +
        0.25 * df["Reasoning_Index"]
    )
else:
    print("\n⚠️ Domain groups not found for this dataset.")
    if "grand_index" in df.columns:
        print("Using existing 'grand_index' as CHI.")
        df["CHI"] = df["grand_index"]
    else:
        print("Using mean of z-scored numeric features as CHI.")
        df["CHI"] = X_z.mean(axis=1)

# -------------------------------------------------
# Fix missing CHI values — two-pass strategy
# -------------------------------------------------
# Pass 1: use row-wise mean of z-scored numeric features where CHI is NaN
missing1 = df["CHI"].isna().sum()
if missing1 > 0:
    print(f"First pass: {missing1} CHI values are NaN. "
          f"Filling them with mean of z-scored numeric features.")
    mask = df["CHI"].isna()
    chi_backup = X_z.mean(axis=1)
    df.loc[mask, "CHI"] = chi_backup[mask]

# Pass 2: any still NaN → fill with global CHI median
missing2 = df["CHI"].isna().sum()
if missing2 > 0:
    print(f"Second pass: {missing2} CHI values are STILL NaN. "
          f"Filling them with global CHI median.")
    df["CHI"] = df["CHI"].fillna(df["CHI"].median())

print("Remaining NaN CHI values:", df["CHI"].isna().sum())

print("\nCHI example:")
print(df["CHI"].head())

# -------------------------------------------------
# Save output
# -------------------------------------------------
out_path = BASE_DIR / "final_dataset.csv"
df.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print("Final shape:", df.shape)
