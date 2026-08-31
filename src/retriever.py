import numpy as np
from rank_bm25 import BM25Okapi


def simple_tokenize(text):
    return text.lower().split()


class BM25Retriever:

    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())

        documents = []

        for doc_id in self.doc_ids:
            title = corpus[doc_id].get("title", "")
            text = corpus[doc_id].get("text", "")

            combined = f"{title} {text}"
            documents.append(simple_tokenize(combined))

        self.bm25 = BM25Okapi(documents)

    def retrieve(self, query, top_k=10):

        query_tokens = simple_tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            doc_id = self.doc_ids[index]

            results.append({
                "id": doc_id,
                "title": self.corpus[doc_id].get("title", ""),
                "text": self.corpus[doc_id].get("text", ""),
                "score": float(scores[index]),
            })

        return results