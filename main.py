import json
import os

from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
)
from src.pipeline import GARRAGPipeline


# ============================================================
# Paths
# ============================================================

CORPUS_PATH = "data/unified_corpus.json"
BM25_PATH = "indexes/bm25.pkl"
EMBEDDING_PATH = "indexes/bge_embeddings.npy"


# ============================================================
# Datasets
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
# Build unified corpus
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
# Load existing corpus or build it
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

        return all_corpus

    print("Building unified corpus...")

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

    print(f"Saved {len(all_corpus)} documents.")

    return all_corpus


# ============================================================
# Create pipeline
# ============================================================

def create_pipeline(all_corpus):

    os.makedirs("indexes", exist_ok=True)

    pipeline = GARRAGPipeline()

    if (
        os.path.exists(BM25_PATH)
        and os.path.exists(EMBEDDING_PATH)
    ):

        pipeline.setup_from_index(
            all_corpus,
            BM25_PATH,
            EMBEDDING_PATH,
        )

    else:

        print("Building retrieval indexes...")

        pipeline.setup(
            all_corpus,
            bm25_path=BM25_PATH,
            embedding_path=EMBEDDING_PATH,
        )

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

    all_corpus = load_or_build_corpus()

    pipeline = create_pipeline(
        all_corpus
    )

    run_user_query(
        pipeline
    )


if __name__ == "__main__":
    main()