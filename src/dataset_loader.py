from datasets import load_dataset


FINANCERAG_REPO = "thomaskim1130/FinanceRAG-Lingua"


def load_financial_dataset(subset: str):

    config = f"preprocessed-{subset}"

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


def convert_corpus_to_dict(corpus):

    result = {}

    for row in corpus:
        doc_id = str(row["_id"])

        result[doc_id] = {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }

    return result


def convert_queries_to_dict(queries):

    result = {}

    for row in queries:
        query_id = str(row["_id"])

        result[query_id] = row["text"]

    return result