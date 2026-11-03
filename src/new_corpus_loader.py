import csv
import os


def load_annual_financial_corpus(csv_path):
    """
    Load the combined FY2025-26 annual financial corpus.
    Preserves CSV row order to maintain alignment
    with BM25 and BGE embeddings.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Annual corpus file not found: {csv_path}"
        )

    corpus = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):

            doc_id = f"ANNUAL_FY25_26_{i}"

            corpus[doc_id] = {
                "id": doc_id,
                "dataset": "Annual_FY2025_26",
                "title": (
                    f"{row['company']} - "
                    f"{row['metric']} - "
                    f"{row['period']}"
                ),
                "text": row["text"],
                "company": row["company"],
                "period": row["period"],
                "period_type": row["period_type"],
                "metric": row["metric"],
                "value": row["value"],
                "unit": row["unit"],
                "segment": row["segment"],
            }

    print(
        f"Loaded {len(corpus)} documents "
        f"from annual financial corpus."
    )

    return corpus