from src.dataset_loader import (
    load_financial_dataset,
    convert_corpus_to_dict,
)

from src.pipeline import GARRAGPipeline


def main():

    print("Loading FinQA corpus...")

    corpus, _ = load_financial_dataset("FinQA")
    corpus = convert_corpus_to_dict(corpus)

    print(f"Corpus documents: {len(corpus)}")

    pipeline = GARRAGPipeline()
    pipeline.setup(corpus)

    print("\n================================")
    print("          FinRAG-GAR")
    print("================================")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your financial question:\n> ")

        if query.strip().lower() == "exit":
            print("Exiting...")
            break

        if not query.strip():
            print("Please enter a question.")
            continue

        print("\nRunning FinRAG-GAR...")

        result = pipeline.run(
            query=query,
            top_k=10,
        )

        print("\n================================")
        print("FINAL ANSWER")
        print("================================")

        print(result["answer"])

        print("\n================================")
        print("SELECTED SOURCES")
        print("================================")

        for i, doc in enumerate(
            result["selected_documents"],
            start=1,
        ):
            print(f"\n[{i}] {doc['title']}")
            print(f"Document ID: {doc['id']}")


if __name__ == "__main__":
    main()