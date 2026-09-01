from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
    convert_queries_to_dict,
)

from src.pipeline import GARRAGPipeline


def main():

    # -----------------------------------------
    # 1. Load dataset
    # -----------------------------------------

    subset = "FinQA"

    print(f"Loading {subset}...")

    corpus, queries = load_financial_dataset(subset)

    corpus = convert_corpus_to_dict(corpus)
    queries = convert_queries_to_dict(queries)

    print(f"Corpus documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")

    # -----------------------------------------
    # 2. Create pipeline
    # -----------------------------------------

    pipeline = GARRAGPipeline()

    pipeline.setup(corpus)

    # -----------------------------------------
    # 3. Test one query
    # -----------------------------------------

    query_id = list(queries.keys())[0]
    query = queries[query_id]

    print("\n==============================")
    print("QUERY")
    print("==============================")
    print(query)

    result = pipeline.run(
        query,
        top_k=10,
    )

    print("\n==============================")
    print("RETRIEVED DOCUMENTS")
    print("==============================")

    for i, doc in enumerate(result["retrieved_documents"]):

        print(f"\nRANK: {i}")
        print(f"ID: {doc.get('id', doc.get('_id', ''))}")
        print(f"TITLE: {doc.get('title', '')}")
        print(f"TEXT: {doc.get('text', '')[:1000]}...")

    # -----------------------------------------
    # 4. Print results
    # -----------------------------------------

    print("\n==============================")
    print("SELECTED DOCUMENTS")
    print("==============================")

    for doc in result["selected_documents"]:
        print(f"\nID: {doc['id']}")
        print(f"TITLE: {doc['title']}")
        print(f"TEXT: {doc['text'][:500]}...")

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")

    print(result["answer"])


if __name__ == "__main__":
    main()