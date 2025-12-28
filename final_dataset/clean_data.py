import pandas as pd
import os
from pathlib import Path

print("Script is running...")

# Folder where THIS script is located: .../dementia/data
BASE_DIR = Path(__file__).resolve().parent

print("Looking for CSV files in:", BASE_DIR)

# Find all CSVs in this folder
csv_paths = list(BASE_DIR.glob("*.csv"))

# OPTIONAL: exclude previously generated files so you don't re-mix them
EXCLUDE = {"cleaned_data.csv", "sample_50k.csv", "sample_25k.csv", "final_dataset.csv"}

csv_paths = [p for p in csv_paths if p.name not in EXCLUDE]

print("Files found:", [p.name for p in csv_paths])

df_list = []
for path in csv_paths:
    print("Reading:", path.name)
    df = pd.read_csv(path)
    df_list.append(df)

# If STILL empty, stop with a clear message
if not df_list:
    raise SystemExit(
        f"❌ ERROR: No input CSV files found in {BASE_DIR}.\n"
        "👉 Put your raw .csv files in this folder and run the script again."
    )

combined = pd.concat(df_list, ignore_index=True)
print("Combined shape:", combined.shape)

combined = combined.drop_duplicates()
print("After removing duplicates:", combined.shape)

output_path = BASE_DIR / "cleaned_data.csv"
combined.to_csv(output_path, index=False)
print("\nSaved cleaned_data.csv at:", output_path)
