from datasets import load_dataset


FINANCERAG_REPO = "thomaskim1130/FinanceRAG-Lingua"


DATASET_CONFIGS = {
    "FinQA": "keyword-FinQA",
    "ConvFinQA": "keyword-ConvFinQA",
    "FinDER": "keyword-FinDER",
    "FinQABench": "keyword-FinQABench",
    "FinanceBench": "keyword-FinanceBench",
    "TATQA": "keyword-TATQA",
    "MultiHiertt": "keyword-MultiHiertt",
}


def load_financial_dataset(subset: str):

    if subset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {subset}\n"
            f"Available datasets: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[subset]

    corpus = load_dataset(
        FINANCERAG_REPO,
        name=config,
        split="corpus",
    )

    queries = load_dataset(
        FINANCERAG_REPO,
        name=config,
        split="queries",
    )

    return corpus, queries


def convert_corpus_to_dict(corpus, dataset_name):

    result = {}

    for row in corpus:

        original_id = str(row["_id"])

        doc_id = f"{dataset_name}_{original_id}"

        result[doc_id] = {
            "id": doc_id,
            "dataset": dataset_name,
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }

    return result


def convert_queries_to_dict(queries, dataset_name):

    result = {}

    for row in queries:

        original_id = str(row["_id"])

        query_id = f"{dataset_name}_{original_id}"

        result[query_id] = {
            "id": query_id,
            "dataset": dataset_name,
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }

    return result