from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
    convert_queries_to_dict,
)

from src.retriever import BM25Retriever
from src.selection_agent import SelectionAgent
from src.answer_agent import AnswerAgent


def main():

    # =========================================
    # 1. Load dataset
    # =========================================

    print("Loading FinQA...")

    corpus, queries = load_financial_dataset("FinQA")

    corpus = convert_corpus_to_dict(corpus)
    queries = convert_queries_to_dict(queries)

    print(f"Corpus documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")

    # =========================================
    # 2. BM25
    # =========================================

    print("\nCreating BM25 retriever...")

    retriever = BM25Retriever(corpus)

    print("BM25 ready.")

    # =========================================
    # 3. Query
    # =========================================

    query_id = list(queries.keys())[0]
    query = queries[query_id]

    print("\n==============================")
    print("QUERY")
    print("==============================")

    print(query)

    # =========================================
    # 4. Retrieve
    # =========================================

    retrieved_documents = retriever.retrieve(
        query,
        top_k=10,
    )

    print(
        f"\nBM25 retrieved "
        f"{len(retrieved_documents)} documents."
    )

    # =========================================
    # 5. Selection
    # =========================================

    print("\nRunning Selection Agent...")

    # Only send Top-5 to Groq
    selection_candidates = retrieved_documents[:5]

    selector = SelectionAgent()

    selected_documents = selector.select(
        query,
        selection_candidates,
    )

    print(
        f"Selection Agent kept "
        f"{len(selected_documents)} documents."
    )

    # =========================================
    # 6. Answer
    # =========================================

    print("\nRunning Answer Agent...")

    answer_agent = AnswerAgent()

    answer = answer_agent.answer(
        query,
        selected_documents,
    )

    # =========================================
    # 7. Final answer
    # =========================================

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(answer)

    # =========================================
    # 8. Summary
    # =========================================

    print("\n==============================")
    print("SUMMARY")
    print("==============================")

    print(f"Query ID: {query_id}")
    print(f"Corpus: {len(corpus)}")
    print(f"BM25 retrieved: {len(retrieved_documents)}")
    print(f"Selection candidates: {len(selection_candidates)}")
    print(f"Selected: {len(selected_documents)}")


if __name__ == "__main__":
    main()