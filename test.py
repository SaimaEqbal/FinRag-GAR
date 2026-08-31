from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
    convert_queries_to_dict,
)

from src.retriever import BM25Retriever
from src.selection_agent import SelectionAgent


def main():

    # =========================================
    # 1. Load FinQA dataset
    # =========================================

    print("Loading FinQA...")

    corpus, queries = load_financial_dataset("FinQA")

    corpus = convert_corpus_to_dict(corpus)
    queries = convert_queries_to_dict(queries)

    print(f"Corpus documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")

    # =========================================
    # 2. Create BM25 Retriever
    # =========================================

    print("\nCreating BM25 retriever...")

    retriever = BM25Retriever(corpus)

    print("BM25 ready.")

    # =========================================
    # 3. Get one test query
    # =========================================

    query_id = list(queries.keys())[0]
    query = queries[query_id]

    print("\n==============================")
    print("QUERY ID")
    print("==============================")
    print(query_id)

    print("\n==============================")
    print("QUERY")
    print("==============================")
    print(query)

    # =========================================
    # 4. BM25 Retrieval
    # =========================================

    print("\n==============================")
    print("BM25 RETRIEVAL")
    print("==============================")

    retrieved_documents = retriever.retrieve(
        query,
        top_k=10,
    )

    print(f"Retrieved documents: {len(retrieved_documents)}")

    # Show BM25 results
    for i, doc in enumerate(retrieved_documents, start=1):

        print(f"\n--- BM25 Document {i} ---")
        print(f"ID: {doc['id']}")
        print(f"Title: {doc['title']}")
        print(f"Score: {doc['score']:.4f}")
        print(f"Text: {doc['text'][:300]}...")

    # =========================================
    # 5. Selection Agent
    # =========================================

    print("\n==============================")
    print("SELECTION AGENT")
    print("==============================")

    # We retrieved Top-10 using BM25,
    # but send only Top-5 to the LLM.
    # This keeps the Groq request small enough.
    selection_candidates = retrieved_documents[:5]

    print(
        f"Sending {len(selection_candidates)} "
        f"documents to Selection Agent."
    )

    selector = SelectionAgent()

    selected_documents = selector.select(
        query,
        selection_candidates,
    )

    print(
        f"\nSelection Agent kept "
        f"{len(selected_documents)} documents."
    )

    # =========================================
    # 6. Display selected documents
    # =========================================

    print("\n==============================")
    print("SELECTED DOCUMENTS")
    print("==============================")

    if not selected_documents:

        print("No documents were selected.")

    else:

        for i, doc in enumerate(
            selected_documents,
            start=1,
        ):

            print(f"\n--- Selected Document {i} ---")

            print(f"ID: {doc['id']}")
            print(f"Title: {doc['title']}")
            print(f"BM25 Score: {doc['score']:.4f}")

            print("\nText:")
            print(doc["text"][:1000])

    # =========================================
    # 7. Summary
    # =========================================

    print("\n==============================")
    print("PIPELINE SUMMARY")
    print("==============================")

    print(f"Corpus size: {len(corpus)}")
    print(f"Query ID: {query_id}")
    print(f"BM25 retrieved: {len(retrieved_documents)}")
    print(f"Selection candidates: {len(selection_candidates)}")
    print(f"Selected documents: {len(selected_documents)}")

    print("\nDataset ✓")
    print("BM25 ✓")
    print("Groq ✓")
    print("Selection Agent ✓")


if __name__ == "__main__":
    main()