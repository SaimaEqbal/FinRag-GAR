
import os
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer


class BM25Retriever:
    def __init__(
        self,
        corpus,
        bm25_path="indexes/bm25.pkl",
        embeddings_path="indexes/bge_embeddings.npy",
        model_name="BAAI/bge-small-en-v1.5",
        rrf_k=60,
    ):
        self.corpus = corpus
        self.rrf_k = rrf_k

        # Preserve corpus ordering for index/embedding alignment.
        if isinstance(corpus, dict):
            self.doc_ids = list(corpus.keys())
            self.documents = list(corpus.values())
        else:
            self.documents = list(corpus)
            self.doc_ids = [
                str(doc.get("id", i))
                for i, doc in enumerate(self.documents)
            ]

        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embedding file not found: {embeddings_path}"
            )

        print("Loading BGE embedding model...")
        self.model = SentenceTransformer(model_name)
        self.embedding_model = self.model

        print("Loading BM25 index...")
        with open(bm25_path, "rb") as file:
            loaded_index = pickle.load(file)

        # Support a directly serialized BM25 object or a dictionary
        # that wraps one. Do not guess how to reconstruct unknown formats.
        self.bm25 = self._unwrap_bm25_index(loaded_index)

        print("Loading BGE embeddings...")
        self.embeddings = np.load(embeddings_path)

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got "
                f"shape {self.embeddings.shape}"
            )

        if len(self.documents) != len(self.embeddings):
            raise ValueError(
                f"Corpus has {len(self.documents)} documents, but "
                f"embeddings have {len(self.embeddings)} rows."
            )

        bm25_count = self._get_bm25_document_count()

        if bm25_count is not None and len(self.documents) != bm25_count:
            raise ValueError(
                f"Corpus has {len(self.documents)} documents, but "
                f"BM25 index reports {bm25_count} documents."
            )

        print(f"Loaded {len(self.documents)} documents.")
        print(f"Loaded embeddings: {self.embeddings.shape}")
        print("BM25 and embedding counts validated.")

    @staticmethod
    def _unwrap_bm25_index(loaded_index):
        """Extract a BM25 object from supported pickle formats."""
        if hasattr(loaded_index, "get_scores"):
            return loaded_index

        if isinstance(loaded_index, dict):
            # Common wrapper keys used when serializing an index.
            for key in ("bm25", "index", "bm25_index", "retriever"):
                candidate = loaded_index.get(key)
                if hasattr(candidate, "get_scores"):
                    return candidate

            # Some serialized objects store the BM25 fields directly.
            # Reconstruct only when the required fields are present.
            required = ("idf", "doc_len", "doc_freqs", "avgdl")
            if all(key in loaded_index for key in required):
                try:
                    from rank_bm25 import BM25Okapi

                    instance = BM25Okapi.__new__(BM25Okapi)
                    instance.__dict__.update(loaded_index)
                    if hasattr(instance, "get_scores"):
                        return instance
                except (ImportError, TypeError, AttributeError):
                    pass

            keys = list(loaded_index.keys())
            raise TypeError(
                "The saved BM25 pickle is a dictionary, but it does not "
                "contain a recognized BM25 object or complete BM25Okapi "
                f"state. Dictionary keys: {keys}. "
                "The loader must match the code that created this index."
            )

        raise TypeError(
            "Unsupported BM25 index format: "
            f"{type(loaded_index).__name__}. "
            "Expected a BM25 object or a supported dictionary."
        )

    def _get_bm25_document_count(self):
        if hasattr(self.bm25, "doc_len"):
            return len(self.bm25.doc_len)

        if hasattr(self.bm25, "corpus_size"):
            return int(self.bm25.corpus_size)

        return None

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+", str(text).lower())

    @staticmethod
    def _normalize_query(query):
        query = str(query).lower()
        query = re.sub(r"[^a-z0-9\s$.-]", " ", query)
        return re.sub(r"\s+", " ", query).strip()

    def _query_variants(self, query):
        normalized = self._normalize_query(query)
        variants = [normalized]

        expanded = normalized
        replacements = [
            (r"\brevenue\b", "revenue total revenue sales"),
            (r"\bfiscal year\b", "fiscal year year ended"),
            (r"\bincome statement\b",
             "income statement income statements"),
            (r"\bmicrosoft\b", "microsoft msft"),
            (r"\bmsft\b", "msft microsoft"),
        ]

        for pattern, replacement in replacements:
            expanded = re.sub(pattern, replacement, expanded)

        if expanded != normalized:
            variants.append(expanded)

        if (
            "revenue" in normalized
            and re.search(r"\b(?:19|20)\d{2}\b", normalized)
        ):
            variants.append(
                normalized + " total revenue financial statements"
            )

        return list(dict.fromkeys(variants))

    def _encode_query(self, query):
        return np.asarray(
            self.model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _get_title(doc):
        return str(
            doc.get("title", doc.get("document_title", ""))
        ).lower()

    @staticmethod
    def _get_text(doc):
        fields = (
            "title",
            "text",
            "content",
            "document",
            "description",
        )
        return " ".join(
            str(doc.get(field, ""))
            for field in fields
            if doc.get(field)
        ).lower()

    def _company_aliases(self, query):
        query = self._normalize_query(query)

        companies = {
            "microsoft": ["microsoft", "msft"],
            "apple": ["apple", "aapl"],
            "nvidia": ["nvidia", "nvda"],
            "adobe": ["adobe", "adbe"],
            "amazon": ["amazon", "amzn"],
            "alphabet": ["alphabet", "google", "googl", "goog"],
            "meta": ["meta", "facebook"],
            "tesla": ["tesla", "tsla"],
            "netflix": ["netflix", "nflx"],
            "jpmorgan": ["jpmorgan", "jpmorgan chase", "jpm"],
        }

        for company, aliases in companies.items():
            if any(
                re.search(r"\b" + re.escape(alias) + r"\b", query)
                for alias in aliases
            ):
                return company, aliases

        return None, []

    def _financial_relevance_score(self, doc, query):
        query = self._normalize_query(query)
        title = self._get_title(doc)
        text = self._get_text(doc)
        doc_id = str(doc.get("id", "")).lower()

        score = 0.0
        company, aliases = self._company_aliases(query)

        if company:
            if any(
                re.search(r"\b" + re.escape(alias) + r"\b", title)
                for alias in aliases
            ):
                score += 3.0
            elif any(
                re.search(r"\b" + re.escape(alias) + r"\b", doc_id)
                for alias in aliases
            ):
                score += 2.5

        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        for year in years:
            if year in title:
                score += 2.0
            if year in text:
                score += 1.0

        asks_revenue = "revenue" in query or "sales" in query
        asks_income_statement = (
            "income statement" in query
            or "income statements" in query
        )

        if asks_revenue:
            if "revenue" in title:
                score += 4.0
            if "total revenue" in text:
                score += 2.0

        if asks_income_statement:
            if "income statement" in title:
                score += 5.0
            elif "financial statements" in title:
                score += 2.0

        if asks_revenue and "overview" in title:
            score -= 1.0

        if asks_revenue and "opinion on the financial statements" in title:
            score -= 1.0

        return score

    def _find_fallback_candidates(self, query, limit=200):
        query = self._normalize_query(query)
        company, aliases = self._company_aliases(query)
        years = re.findall(r"\b(?:19|20)\d{2}\b", query)

        financial_terms = (
            "revenue",
            "sales",
            "income statement",
            "financial statement",
            "financial performance",
            "profit",
            "earnings",
        )

        if not company or not years:
            return []

        if not any(term in query for term in financial_terms):
            return []

        matches = []

        for index, doc in enumerate(self.documents):
            title = self._get_title(doc)
            text = self._get_text(doc)
            doc_id = str(doc.get("id", "")).lower()

            company_match = any(
                re.search(r"\b" + re.escape(alias) + r"\b", title)
                or re.search(r"\b" + re.escape(alias) + r"\b", doc_id)
                for alias in aliases
            )

            if not company_match:
                continue

            if not any(
                year in title or year in text for year in years
            ):
                continue

            if not any(
                term in title or term in text
                for term in financial_terms
            ):
                continue

            matches.append(
                (
                    index,
                    self._financial_relevance_score(doc, query),
                )
            )

        matches.sort(key=lambda item: item[1], reverse=True)
        return [index for index, _ in matches[:limit]]

    def retrieve(self, query, top_k=10):
        if not isinstance(query, str) or not query.strip():
            return []

        if top_k <= 0:
            return []

        variants = self._query_variants(query)
        bm25_best = {}
        bge_best = {}

        candidate_k = min(500, len(self.documents))

        for variant in variants:
            tokens = self._tokenize(variant)

            if tokens:
                scores = np.asarray(
                    self.bm25.get_scores(tokens),
                    dtype=np.float32,
                )

                count = min(candidate_k, len(scores))
                if count:
                    indices = np.argpartition(scores, -count)[-count:]
                    indices = sorted(
                        (int(i) for i in indices),
                        key=lambda i: scores[i],
                        reverse=True,
                    )

                    for index in indices:
                        score = float(scores[index])
                        bm25_best[index] = max(
                            score, bm25_best.get(index, -np.inf)
                        )

            query_embedding = self._encode_query(variant)
            similarities = self.embeddings @ query_embedding

            count = min(candidate_k, len(similarities))
            if count:
                indices = np.argpartition(
                    similarities, -count
                )[-count:]
                indices = sorted(
                    (int(i) for i in indices),
                    key=lambda i: similarities[i],
                    reverse=True,
                )

                for index in indices:
                    score = float(similarities[index])
                    bge_best[index] = max(
                        score, bge_best.get(index, -np.inf)
                    )

        # Rebuild rankings from the best scores for each retriever.
        bm25_order = sorted(
            bm25_best,
            key=lambda i: bm25_best[i],
            reverse=True,
        )
        bge_order = sorted(
            bge_best,
            key=lambda i: bge_best[i],
            reverse=True,
        )

        rrf_scores = {}

        for rank, index in enumerate(bm25_order, start=1):
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        for rank, index in enumerate(bge_order, start=1):
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # Merge fallback candidates before applying top_k.
        candidate_indices = set(rrf_scores)
        candidate_indices.update(
            self._find_fallback_candidates(query)
        )

        relevance_weight = 0.005

        def final_score(index):
            relevance = self._financial_relevance_score(
                self.documents[index], query
            )
            return (
                rrf_scores.get(index, 0.0)
                + relevance_weight * relevance
            )

        ranked_indices = sorted(
            candidate_indices,
            key=final_score,
            reverse=True,
        )[:top_k]

        results = []

        for index in ranked_indices:
            doc = dict(self.documents[index])
            doc.setdefault("id", self.doc_ids[index])

            doc["bm25_score"] = float(
                bm25_best.get(index, 0.0)
            )
            doc["embedding_score"] = float(
                bge_best.get(index, 0.0)
            )
            doc["rrf_score"] = float(
                rrf_scores.get(index, 0.0)
            )
            doc["financial_relevance_score"] = float(
                self._financial_relevance_score(
                    self.documents[index], query
                )
            )
            doc["hybrid_score"] = float(final_score(index))

            results.append(doc)

        return results
