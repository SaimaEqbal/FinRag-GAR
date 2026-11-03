
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "financial_data/processed_data/TCS_Q2_FY27_facts.csv"
)

OUTPUT_FILE = Path(
    "financial_data/processed_data/TCS_Q2_FY27_normalized.csv"
)

df = pd.read_csv(INPUT_FILE)

# Convert values to numeric
df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Classify reporting periods
def classify_period(row):
    if pd.notna(row["instant"]):
        return "instant"

    start = str(row["start_date"])
    end = str(row["end_date"])

    if start == "2026-07-01" and end == "2026-09-30":
        return "quarterly"

    if start == "2026-04-01" and end == "2026-09-30":
        return "six_months"

    return "other"

df["period_type"] = df.apply(classify_period, axis=1)

# Convert INR monetary values into crore.
# Do not convert EPS or share-count units.
def convert_to_crore(row):
    unit = str(row["unit"])

    if unit == "iso4217:INR":
        return row["value"] / 10_000_000

    return row["value"]

df["value_in_crore"] = df.apply(convert_to_crore, axis=1)

# Save normalized dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Normalization completed!")
print("Total rows:", len(df))
print("Output:", OUTPUT_FILE)

print("\nPeriod distribution:")
print(df["period_type"].value_counts())

print("\nSample normalized financial facts:")
print(
    df[
        ["metric_name", "value", "value_in_crore",
         "period_type", "dimensions"]
    ].head(20).to_string(index=False)
)
