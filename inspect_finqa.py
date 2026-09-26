from datasets import load_dataset


def main():

    print("Loading raw FinQA dataset...")

    dataset = load_dataset(
        "thomaskim1130/FinanceRAG-Lingua",
        name="preprocessed-FinQA",
        split="queries",
    )

    print(f"Number of queries: {len(dataset)}")

    print("\n==============================")
    print("FIRST QUERY RAW RECORD")
    print("==============================")

    print(dataset[0])


if __name__ == "__main__":
    main()