import pandas as pd
from pathlib import Path

# Base directory = folder where this script lives
BASE_DIR = Path(__file__).resolve().parent

print("\nLoading cleaned_data.csv...")
cleaned_path = BASE_DIR / "cleaned_data.csv"

# low_memory=False to avoid mixed-type warning chunks
df = pd.read_csv(cleaned_path, low_memory=False)

print("Original rows:", len(df))

TARGET_SAMPLES = 50000
n_samples = min(TARGET_SAMPLES, len(df))

# Simple random sample
df_sample = df.sample(n=n_samples, random_state=42)

print("Sampled rows:", len(df_sample))

out_path = BASE_DIR / "sample_50k.csv"
df_sample.to_csv(out_path, index=False)
print("\nSaved:", out_path)
