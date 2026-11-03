
import pandas as pd
import json
from pathlib import Path

INPUT_FILE = Path(
    "financial_data/FY2026_27_corpus/processed_data/"
    "TCS/TCS_Q2_FY27_normalized.csv"
)

OUTPUT_FILE = Path(
    "financial_data/FY2026_27_corpus/processed_data/"
    "TCS/TCS_Q2_FY27_records.csv"
)

df = pd.read_csv(INPUT_FILE)

records = []

for _, row in df.iterrows():

    # Skip metadata and facts without financial units
    if pd.isna(row["unit"]) or pd.isna(row["value_in_crore"]):
        continue

    metric = str(row["metric_name"])
    value = float(row["value_in_crore"])
    period = str(row["period_type"])
    unit = str(row["unit"])

    # Identify the display unit
    if unit == "iso4217:INR":
        display_unit = "INR crore"
    elif "shares" in unit:
        display_unit = "INR per share"
    else:
        display_unit = unit

    # Extract segment identity
    segment = ""

    try:
        dimensions = json.loads(row["dimensions"])

        for dimension in dimensions:
            member = dimension.get("value", "")

            if "ReportableSegments" in member:
                number = member.split("ReportableSegments")[-1]
                number = number.replace("Member", "")
                segment = f"Reportable Segment {number}"
                break

    except (ValueError, TypeError):
        pass

    if period == "quarterly":
        reporting_period = "Q2 FY2026-27 (July-September 2026)"
    elif period == "six_months":
        reporting_period = "H1 FY2026-27 (April-September 2026)"
    elif period == "instant":
        reporting_period = "As of 30 September 2026"
    else:
        reporting_period = "FY2026-27"

    text = (
        "Company: Tata Consultancy Services (TCS). "
        f"Reporting period: {reporting_period}. "
        f"Period type: {period}. "
        f"Metric: {metric}. "
        f"Value: {value:,.2f} {display_unit}."
    )

    if segment:
        text += f" Segment: {segment}."

    records.append({
        "company": "TCS",
        "period": "Q2 FY2026-27",
        "period_type": period,
        "metric": metric,
        "value": value,
        "unit": display_unit,
        "segment": segment,
        "text": text
    })

result = pd.DataFrame(records)
result.to_csv(OUTPUT_FILE, index=False)

print("Financial records created!")
print("Total records:", len(result))
print("Output:", OUTPUT_FILE)

print("\nSample records:")
print(result[["metric", "value", "unit", "segment"]].head(15))