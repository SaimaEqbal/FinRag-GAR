
import json
import os
import pickle

from sentence_transformers import SentenceTransformer

from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
)
from src.new_corpus_loader import load_annual_financial_corpus
from src.pipeline import GARRAGPipeline


# ============================================================
# PATHS
# ============================================================

CORPUS_PATH = "data/unified_corpus.json"

BM25_PATH = "indexes/bm25.pkl"
EMBEDDING_PATH = "indexes/bge_embeddings.npy"

ANNUAL_CORPUS_CSV = (
    "financial_data/FY2026_27_corpus/"
    "processed_data/combined_annual_corpus_clean.csv"
)

ANNUAL_CORPUS_CACHE = (
    "financial_data/FY2026_27_corpus/"
    "processed_data/annual_corpus_cache.pkl"
)

ANNUAL_BM25_PATH = (
    "financial_data/FY2026_27_corpus/"
    "embeddings/combined_bm25.pkl"
)

ANNUAL_EMBEDDING_PATH = (
    "financial_data/FY2026_27_corpus/"
    "embeddings/combined_embeddings.npy"
)

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


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
# ORIGINAL CORPUS
# ============================================================

def build_unified_corpus():

    all_corpus = {}

    for dataset_name in DATASETS:

        print(f"Loading dataset: {dataset_name}")

        corpus, _ = load_financial_dataset(dataset_name)

        corpus = convert_corpus_to_dict(
            corpus,
            dataset_name,
        )

        all_corpus.update(corpus)

    return all_corpus


def load_or_build_corpus():

    os.makedirs("data", exist_ok=True)

    if os.path.exists(CORPUS_PATH):

        with open(CORPUS_PATH, "r", encoding="utf-8") as file:
            all_corpus = json.load(file)

        print(
            f"Loaded cached original corpus: "
            f"{len(all_corpus)} documents."
        )

        return all_corpus

    print("Building original unified corpus...")

    all_corpus = build_unified_corpus()

    with open(CORPUS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            all_corpus,
            file,
            ensure_ascii=False,
        )

    print(f"Saved original corpus to {CORPUS_PATH}")

    return all_corpus


# ============================================================
# ANNUAL CORPUS WITH DISK CACHE
# ============================================================

def load_annual_corpus():

    if not os.path.exists(ANNUAL_CORPUS_CSV):
        raise FileNotFoundError(
            f"Annual corpus CSV not found: {ANNUAL_CORPUS_CSV}"
        )

    os.makedirs(
        os.path.dirname(ANNUAL_CORPUS_CACHE),
        exist_ok=True,
    )

    csv_stat = os.stat(ANNUAL_CORPUS_CSV)

    current_signature = (
        csv_stat.st_mtime_ns,
        csv_stat.st_size,
    )

    # Reuse cached corpus if the CSV has not changed.
    if os.path.exists(ANNUAL_CORPUS_CACHE):

        try:
            with open(ANNUAL_CORPUS_CACHE, "rb") as file:
                cached = pickle.load(file)

            if (
                cached.get("signature") == current_signature
                and isinstance(cached.get("corpus"), dict)
            ):
                corpus = cached["corpus"]

                print(
                    f"Loaded cached annual corpus: "
                    f"{len(corpus)} documents."
                )

                return corpus

        except (OSError, pickle.PickleError, EOFError, AttributeError):
            print("Annual corpus cache is invalid. Rebuilding it.")

    print("Loading annual corpus from CSV...")

    corpus = load_annual_financial_corpus(
        ANNUAL_CORPUS_CSV
    )

    cached = {
        "signature": current_signature,
        "corpus": corpus,
    }

    with open(ANNUAL_CORPUS_CACHE, "wb") as file:
        pickle.dump(
            cached,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("Annual corpus cache saved.")

    return corpus


# ============================================================
# CREATE PIPELINE
# ============================================================

def create_pipeline(all_corpus, annual_corpus):

    required_paths = [
        BM25_PATH,
        EMBEDDING_PATH,
        ANNUAL_BM25_PATH,
        ANNUAL_EMBEDDING_PATH,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required retrieval index not found: {path}"
            )

    # Load BGE exactly once.
    print("\nLoading shared BGE model...")

    embedding_model = SentenceTransformer(
        BGE_MODEL_NAME
    )

    pipeline = GARRAGPipeline(
        embedding_model=embedding_model,
        max_selected_documents=10,
        verify_answers=True,
    )

    # Both retrievers share this exact model instance.
    pipeline.setup_from_index(
        corpus=all_corpus,
        bm25_path=BM25_PATH,
        embedding_path=EMBEDDING_PATH,
        annual_corpus=annual_corpus,
        annual_bm25_path=ANNUAL_BM25_PATH,
        annual_embedding_path=ANNUAL_EMBEDDING_PATH,
    )

    print("\nPipeline initialized successfully.")
    print(f"Original corpus: {len(all_corpus)} documents")
    print(f"Annual corpus: {len(annual_corpus)} documents")

    return pipeline


# ============================================================
# RUN QUERY
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
        top_k=10,
    )

    retrieved = result["retrieved_documents"]
    selected = result["selected_documents"]

    original_count = sum(
        doc.get("retrieval_source") == "original_corpus"
        for doc in retrieved
    )

    annual_count = sum(
        doc.get("retrieval_source") == "annual_corpus"
        for doc in retrieved
    )

    print("\n" + "-" * 60)
    print("RETRIEVAL SUMMARY")
    print("-" * 60)

    print(f"Retrieved documents: {len(retrieved)}")
    print(f"Original corpus results: {original_count}")
    print(f"Annual corpus results: {annual_count}")
    print(f"Selected documents: {len(selected)}")

    print("\n" + "-" * 60)
    print("SELECTED DOCUMENTS")
    print("-" * 60)

    for i, doc in enumerate(selected, start=1):
        print(
            f"{i}. {doc.get('id')} — "
            f"{doc.get('title')} "
            f"[{doc.get('retrieval_source')}]"
        )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result["answer"])

    verification = result.get("verification")

    if isinstance(verification, dict):
        print("\n" + "-" * 60)
        print("VERIFICATION")
        print("-" * 60)

        print("Status:", verification.get("status"))
        print("Fully supported:", verification.get("supported"))


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading corpora...")

    all_corpus = load_or_build_corpus()
    annual_corpus = load_annual_corpus()

    print("\nInitializing retrieval systems...")

    pipeline = create_pipeline(
        all_corpus=all_corpus,
        annual_corpus=annual_corpus,
    )

    # Keep the pipeline alive for multiple questions.
    while True:

        run_user_query(pipeline)

        again = input(
            "\nAsk another question? (y/n): "
        ).strip().lower()

        if again != "y":
            break


if __name__ == "__main__":
    main()
