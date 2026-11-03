import xml.etree.ElementTree as ET
import pandas as pd
import json
from pathlib import Path
from datetime import date

# File paths
INPUT_FILE = Path(
    "financial_data/raw_data/TCS/TCS_Q2_FY27_consolidated.xml"
)

OUTPUT_FILE = Path(
    "financial_data/processed_data/TCS_Q2_FY27_facts.csv"
)


def get_local_name(tag):
    """Remove the XML namespace and return the tag name."""
    return tag.split("}")[-1]


def extract_contexts(root):
    """Extract reporting periods and dimensions from XBRL contexts."""
    contexts = {}

    for element in root.iter():
        if get_local_name(element.tag) != "context":
            continue

        context_id = element.attrib.get("id")
        if not context_id:
            continue

        context = {
            "context_id": context_id,
            "entity_identifier": None,
            "start_date": None,
            "end_date": None,
            "instant": None,
            "dimensions": []
        }

        for child in element.iter():
            name = get_local_name(child.tag)

            if name == "identifier":
                context["entity_identifier"] = child.text

            elif name == "startDate":
                context["start_date"] = child.text

            elif name == "endDate":
                context["end_date"] = child.text

            elif name == "instant":
                context["instant"] = child.text

            elif name in ("explicitMember", "typedMember"):
                context["dimensions"].append({
                    "type": name,
                    "dimension": child.attrib.get("dimension"),
                    "value": "".join(child.itertext()).strip()
                })

        contexts[context_id] = context

    return contexts


def extract_units(root):
    """Map unit IDs to their measurement units."""
    units = {}

    for element in root.iter():
        if get_local_name(element.tag) == "unit":
            unit_id = element.attrib.get("id")

            if unit_id:
                measures = [
                    child.text
                    for child in element.iter()
                    if get_local_name(child.tag) == "measure"
                ]

                units[unit_id] = "*".join(
                    measure for measure in measures if measure
                )

    return units

def classify_period(context):
    if context["instant"]:
        return "instant"

    if context["start_date"] and context["end_date"]:
        start = date.fromisoformat(context["start_date"])
        end = date.fromisoformat(context["end_date"])

        days = (end - start).days + 1

        if 80 <= days <= 100:
            return "quarterly"

        if 170 <= days <= 195:
            return "six_months"

        return "other_duration"

    return "unknown"


def parse_xbrl():
    print("Reading XBRL XML...")

    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()

    contexts = extract_contexts(root)
    units = extract_units(root)

    print(f"Contexts found: {len(contexts)}")
    print(f"Units found: {len(units)}")

    facts = []

    for element in root.iter():

        context_id = element.attrib.get("contextRef")

        # Only extract elements that refer to an XBRL context.
        if not context_id:
            continue

        context = contexts.get(context_id, {})

        namespace = (
            element.tag.split("}")[0][1:]
            if "}" in element.tag else ""
        )

        unit_id = element.attrib.get("unitRef")

        fact = {
            "metric_name": get_local_name(element.tag),
            "namespace": namespace,
            "value": "".join(element.itertext()).strip(),
            "unit": units.get(unit_id, ""),
            "context_id": context_id,
            "entity_identifier": context.get("entity_identifier"),
            "start_date": context.get("start_date"),
            "end_date": context.get("end_date"),
            "instant": context.get("instant"),
            "period_type": classify_period(context),
            "dimensions": json.dumps(
                context.get("dimensions", [])
            ),
            "decimals": element.attrib.get("decimals"),
            "source_file": INPUT_FILE.name
        }

        facts.append(fact)

    df = pd.DataFrame(facts)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nParsing completed!")
    print(f"Total fact elements extracted: {len(df)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nSample extracted facts:")
    print(
        df[["metric_name", "value", "unit",
            "start_date", "end_date", "instant"]].head(15)
    )


if __name__ == "__main__":
    parse_xbrl()