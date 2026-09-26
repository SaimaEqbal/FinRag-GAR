import os
import pickle

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class BM25Retriever:

    def __init__(
        self,
        corpus=None,
        index_path=None,
        embedding_path=None,
        model_name="BAAI/bge-small-en-v1.5",
    ):
        self.corpus = corpus
        self.doc_ids = None
        self.documents = None
        self.bm25 = None
        self.embeddings = None

        # --------------------------------------------------
        # Load BGE model
        # --------------------------------------------------

        print("Loading BGE embedding model...")

        self.embedding_model = SentenceTransformer(model_name)

        # --------------------------------------------------
        # Load BM25 index
        # --------------------------------------------------

        if index_path is not None:

            print("Loading BM25 index...")

            with open(index_path, "rb") as f:
                data = pickle.load(f)

            self.doc_ids = data["doc_ids"]
            self.documents = data["documents"]
            self.bm25 = data["bm25"]

            print(
                f"Loaded BM25 index with {len(self.doc_ids)} documents."
            )

        elif corpus is not None:

            self.corpus = corpus

            self.doc_ids = list(corpus.keys())

            self.documents = [
                corpus[doc_id]["text"]
                for doc_id in self.doc_ids
            ]

            print("Building BM25 index...")

            tokenized_documents = [
                document.lower().split()
                for document in self.documents
            ]

            self.bm25 = BM25Okapi(
                tokenized_documents
            )

            print(
                f"BM25 index built for {len(self.doc_ids)} documents."
            )

        # --------------------------------------------------
        # Check embedding path
        # --------------------------------------------------

        if embedding_path is None:

            raise ValueError(
                "embedding_path must be provided."
            )

        if not os.path.exists(embedding_path):

            raise FileNotFoundError(
                f"BGE embeddings not found at:\n"
                f"{embedding_path}"
            )

        # --------------------------------------------------
        # Load BGE embeddings
        # --------------------------------------------------

        print("Loading BGE embeddings...")

        self.embeddings = np.load(
            embedding_path
        )

        print(
            f"Loaded BGE embeddings with shape "
            f"{self.embeddings.shape}"
        )

        # --------------------------------------------------
        # Verify document / embedding alignment
        # --------------------------------------------------

        if len(self.doc_ids) != self.embeddings.shape[0]:

            raise ValueError(
                "Number of documents does not match "
                "number of BGE embeddings."
            )

        print(
            "Document count and embedding count match."
        )

    # ======================================================
    # SAVE BM25 INDEX
    # ======================================================

    def save(self, index_path):

        data = {
            "doc_ids": self.doc_ids,
            "documents": self.documents,
            "bm25": self.bm25,
        }

        with open(index_path, "wb") as f:

            pickle.dump(
                data,
                f
            )

        print(
            f"BM25 index saved to {index_path}"
        )

    # ======================================================
    # RETRIEVE
    # ======================================================

    def retrieve(
        self,
        query,
        top_k=10,
        bm25_k=100,
        embedding_k=500,
    ):

        # --------------------------------------------------
        # BM25 retrieval
        # --------------------------------------------------

        tokenized_query = query.lower().split()

        bm25_scores = self.bm25.get_scores(
            tokenized_query
        )

        bm25_indices = (
            bm25_scores
            .argsort()[::-1][:bm25_k]
        )

        # --------------------------------------------------
        # BGE retrieval
        # --------------------------------------------------

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        embedding_scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        embedding_indices = (
            embedding_scores
            .argsort()[::-1][:embedding_k]
        )

        # --------------------------------------------------
        # Combine candidates
        # --------------------------------------------------

        candidate_indices = set(
            bm25_indices.tolist()
        )

        candidate_indices.update(
            embedding_indices.tolist()
        )

        candidate_indices = list(
            candidate_indices
        )

        # --------------------------------------------------
        # Extract candidate scores
        # --------------------------------------------------

        candidate_bm25_scores = np.array(
            [
                bm25_scores[i]
                for i in candidate_indices
            ]
        )

        candidate_embedding_scores = np.array(
            [
                embedding_scores[i]
                for i in candidate_indices
            ]
        )

        # --------------------------------------------------
        # Normalize BM25 scores
        # --------------------------------------------------

        if (
            candidate_bm25_scores.max()
            > candidate_bm25_scores.min()
        ):

            bm25_normalized = (
                candidate_bm25_scores
                - candidate_bm25_scores.min()
            ) / (
                candidate_bm25_scores.max()
                - candidate_bm25_scores.min()
            )

        else:

            bm25_normalized = np.zeros(
                len(candidate_indices)
            )

        # --------------------------------------------------
        # Normalize BGE scores
        # --------------------------------------------------

        if (
            candidate_embedding_scores.max()
            > candidate_embedding_scores.min()
        ):

            embedding_normalized = (
                candidate_embedding_scores
                - candidate_embedding_scores.min()
            ) / (
                candidate_embedding_scores.max()
                - candidate_embedding_scores.min()
            )

        else:

            embedding_normalized = np.zeros(
                len(candidate_indices)
            )

        # --------------------------------------------------
        # Hybrid score
        # --------------------------------------------------

        hybrid_scores = (
            0.2 * bm25_normalized
            + 0.8 * embedding_normalized
        )

        # --------------------------------------------------
        # Rank candidates
        # --------------------------------------------------

        ranked_order = (
            np.argsort(hybrid_scores)[::-1]
            [:top_k]
        )

        # --------------------------------------------------
        # Build results
        # --------------------------------------------------

        results = []

        for position in ranked_order:

            index = candidate_indices[position]

            doc_id = self.doc_ids[index]

            document = self.corpus[
                doc_id
            ].copy()

            document["bm25_score"] = float(
                bm25_scores[index]
            )

            document["embedding_score"] = float(
                embedding_scores[index]
            )

            document["hybrid_score"] = float(
                hybrid_scores[position]
            )

            results.append(
                document
            )

        return results