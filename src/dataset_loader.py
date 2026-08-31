from datasets import load_dataset


FINANCERAG_REPO = "Linq-AI-Research/FinanceRAG"


def load_financial_dataset(subset: str):
    """
    Load one FinanceRAG dataset subset.

    Available subsets in the original project include:
        FinDER
        FinQABench
        FinQA
        FinanceBench
        TATQA
        ConvFinQA
        MultiHiertt
    """

    corpus = load_dataset(
        FINANCERAG_REPO,
        name=subset,
        split="corpus",
    )

    queries = load_dataset(
        FINANCERAG_REPO,
        name=subset,
        split="queries",
    )

    return corpus, queries


def convert_corpus_to_dict(corpus):
    """
    Convert Hugging Face corpus into:

    {
        document_id: {
            "title": "...",
            "text": "..."
        }
    }
    """

    result = {}

    for row in corpus:
        doc_id = str(row["_id"])

        result[doc_id] = {
            "title": row.get("title", ""),
            "text": row.get("text", ""),
        }

    return result


def convert_queries_to_dict(queries):
    """
    Convert Hugging Face queries into:

    {
        query_id: query_text
    }
    """

    result = {}

    for row in queries:
        query_id = str(row["_id"])
        result[query_id] = row["text"]

    return result