import json
import os

from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
)
from src.new_corpus_loader import load_annual_financial_corpus
from src.pipeline import GARRAGPipeline


# ============================================================
# Original corpus paths
# ============================================================

CORPUS_PATH = "data/unified_corpus.json"
BM25_PATH = "indexes/bm25.pkl"
EMBEDDING_PATH = "indexes/bge_embeddings.npy"


# ============================================================
# Annual FY2025-26 corpus paths
# ============================================================

ANNUAL_CORPUS_CSV = (
    "financial_data/FY2026_27_corpus/"
    "processed_data/combined_annual_corpus_clean.csv"
)

ANNUAL_BM25_PATH = (
    "financial_data/FY2026_27_corpus/"
    "embeddings/combined_bm25.pkl"
)

ANNUAL_EMBEDDING_PATH = (
    "financial_data/FY2026_27_corpus/"
    "embeddings/combined_embeddings.npy"
)


# ============================================================
# Original datasets
# ============================================================

DATASETS = [
    "FinQA",
    "ConvFinQA",
    "FinDER",
    "FinQABench",
    "FinanceBench",
    "TATQA",
    "MultiHiertt",
]


# ============================================================
# Build original unified corpus
# ============================================================

def build_unified_corpus():
    all_corpus = {}

    for dataset_name in DATASETS:
        corpus, _ = load_financial_dataset(dataset_name)

        corpus = convert_corpus_to_dict(
            corpus,
            dataset_name
        )

        all_corpus.update(corpus)

    return all_corpus


# ============================================================
# Load existing original corpus or build it
# ============================================================

def load_or_build_corpus():

    os.makedirs("data", exist_ok=True)

    if os.path.exists(CORPUS_PATH):

        with open(
            CORPUS_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            all_corpus = json.load(f)

        print(
            f"Loaded original corpus: "
            f"{len(all_corpus)} documents."
        )

        return all_corpus

    print("Building original unified corpus...")

    all_corpus = build_unified_corpus()

    with open(
        CORPUS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_corpus,
            f,
            ensure_ascii=False
        )

    print(
        f"Saved {len(all_corpus)} documents "
        f"to {CORPUS_PATH}."
    )

    return all_corpus


# ============================================================
# Load new FY2026-27 corpus
# ============================================================

def load_annual_corpus():

    if not os.path.exists(ANNUAL_CORPUS_CSV):
        raise FileNotFoundError(
            f"Annual corpus CSV not found: {ANNUAL_CORPUS_CSV}"
        )

    return load_annual_financial_corpus(
        ANNUAL_CORPUS_CSV
    )


# ============================================================
# Create pipeline
# ============================================================


def create_pipeline(all_corpus, annual_corpus):

    # Validate original index files
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(
            f"Original BM25 index not found: {BM25_PATH}"
        )

    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Original embedding matrix not found: {EMBEDDING_PATH}"
        )

    # Validate annual index files
    if not os.path.exists(ANNUAL_BM25_PATH):
        raise FileNotFoundError(
            f"Annual BM25 index not found: {ANNUAL_BM25_PATH}"
        )

    if not os.path.exists(ANNUAL_EMBEDDING_PATH):
        raise FileNotFoundError(
            f"Annual embedding matrix not found: {ANNUAL_EMBEDDING_PATH}"
        )

    pipeline = GARRAGPipeline()

    pipeline.setup_from_index(
        # Original corpus
        corpus=all_corpus,
        bm25_path=BM25_PATH,
        embedding_path=EMBEDDING_PATH,

        # Annual corpus
        annual_corpus=annual_corpus,
        annual_bm25_path=ANNUAL_BM25_PATH,
        annual_embedding_path=ANNUAL_EMBEDDING_PATH,
    )

    print("\nBoth retrieval systems loaded successfully.")
    print(f"Original corpus: {len(all_corpus)} documents")
    print(f"Annual corpus: {len(annual_corpus)} documents")

    return pipeline


# ============================================================
# Run user query
# ============================================================

def run_user_query(pipeline):

    print("\n" + "=" * 60)
    print("FINRAG-GAR")
    print("=" * 60)

    user_query = input(
        "\nEnter your financial question: "
    ).strip()

    if not user_query:
        print("No query entered.")
        return

    result = pipeline.run(
        user_query,
        top_k=10
    )

    # --------------------------------------------------------
    # Retrieval summary
    # --------------------------------------------------------

    print(
        f"\nRetrieved: "
        f"{len(result['retrieved_documents'])} documents"
    )

    print(
        f"Selected: "
        f"{len(result['selected_documents'])} documents"
    )

    # --------------------------------------------------------
    # Retrieval source
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("RETRIEVAL SOURCE")
    print("-" * 60)

    for item in result["history"]:
        if item.get("stage") == "evidence_check":

            print(
                f"Query: {item.get('query')}"
            )

            print(
                f"Source: {item.get('retrieval_source')}"
            )

            if item.get("new_corpus_attempt"):
                print(
                    "New corpus assessment:",
                    item["new_corpus_attempt"]
                )

    # --------------------------------------------------------
    # Selected documents
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SELECTED DOCUMENTS")
    print("-" * 60)

    if result["selected_documents"]:

        for i, doc in enumerate(
            result["selected_documents"],
            start=1
        ):

            print(
                f"{i}. "
                f"{doc['id']} — "
                f"{doc['title']}"
            )

    else:

        print("No documents selected.")

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result["answer"])


# ============================================================
# Main
# ============================================================

def main():

    # Load original corpus.
    all_corpus = load_or_build_corpus()

    # Load new FY2026-27 corpus.
    annual_corpus = load_annual_corpus()

    # Initialize both retrieval systems.
    pipeline = create_pipeline(
        all_corpus=all_corpus,
        annual_corpus=annual_corpus,
    )

    # Run the existing agentic RAG pipeline.
    run_user_query(pipeline)


if __name__ == "__main__":
    main()