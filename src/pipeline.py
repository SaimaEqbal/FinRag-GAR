
from sentence_transformers import SentenceTransformer

from .retriever import BM25Retriever
from .answer_agent import AnswerAgent
from .answer_verifier import AnswerVerifier


class GARRAGPipeline:

    def __init__(
        self,
        embedding_model,
        max_selected_documents=10,
        verify_answers=True,
    ):
        self.embedding_model = embedding_model
        self.max_selected_documents = max_selected_documents
        self.verify_answers = verify_answers

        self.answer_agent = AnswerAgent()
        self.answer_verifier = (
            AnswerVerifier() if verify_answers else None
        )

        self.retriever = None
        self.annual_retriever = None

    # --------------------------------------------------------
    # INITIALIZE BOTH RETRIEVERS
    # --------------------------------------------------------

    def setup_from_index(
        self,
        corpus,
        bm25_path,
        embedding_path,
        annual_corpus,
        annual_bm25_path,
        annual_embedding_path,
    ):

        # Both retrievers receive the SAME model object.
        self.retriever = BM25Retriever(
            corpus=corpus,
            bm25_path=bm25_path,
            embeddings_path=embedding_path,
            embedding_model=self.embedding_model,
        )

        self.annual_retriever = BM25Retriever(
            corpus=annual_corpus,
            bm25_path=annual_bm25_path,
            embeddings_path=annual_embedding_path,
            embedding_model=self.embedding_model,
        )

        print("Both retrievers initialized using one BGE model.")

    # --------------------------------------------------------
    # COMBINE RESULTS FROM BOTH CORPORA
    # --------------------------------------------------------

    @staticmethod
    def _combine_results(
        original_results,
        annual_results,
        limit,
        rrf_k=60,
    ):

        combined = {}

        sources = [
            ("original_corpus", original_results),
            ("annual_corpus", annual_results),
        ]

        for source_name, documents in sources:

            for rank, document in enumerate(documents, start=1):

                doc = dict(document)
                doc["retrieval_source"] = source_name

                # Include dataset in the key to avoid accidentally
                # merging unrelated documents with the same ID.
                key = (
                    str(doc.get("dataset", "")),
                    str(doc.get("id", "")),
                )

                if key not in combined:
                    doc["combined_score"] = 0.0
                    combined[key] = doc

                combined[key]["combined_score"] += (
                    1.0 / (rrf_k + rank)
                )

        ranked = sorted(
            combined.values(),
            key=lambda doc: doc["combined_score"],
            reverse=True,
        )

        return ranked[:limit]

    # --------------------------------------------------------
    # RETRIEVE FROM BOTH CORPORA
    # --------------------------------------------------------

    def _retrieve_both(self, query, top_k):

        # Generate query variants only once.
        variants = self.retriever._query_variants(query)

        # Encode each variant only once using shared BGE.
        query_embeddings = {}

        for variant in variants:
            query_embeddings[variant] = self.embedding_model.encode(
                variant,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        # Search original corpus.
        original_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            query_embeddings=query_embeddings,
        )

        # Search annual corpus using the SAME query embeddings.
        annual_results = self.annual_retriever.retrieve(
            query=query,
            top_k=top_k,
            query_embeddings=query_embeddings,
        )

        # Combine both result lists.
        retrieved_documents = self._combine_results(
            original_results=original_results,
            annual_results=annual_results,
            limit=top_k * 2,
        )

        selected_documents = retrieved_documents[
            :self.max_selected_documents
        ]

        return retrieved_documents, selected_documents

    # --------------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------------

    def run(self, query, top_k=10):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.retriever is None or self.annual_retriever is None:
            raise RuntimeError(
                "Both retrievers must be initialized before running."
            )

        query = query.strip()

        # 1. Retrieve from both corpora.
        retrieved_documents, selected_documents = (
            self._retrieve_both(query, top_k)
        )

        if not selected_documents:
            return self._build_result(
                query,
                retrieved_documents,
                selected_documents,
                None,
                None,
                [],
            )

        # 2. Generate answer.
        draft_answer = self.answer_agent.answer(
            query=query,
            documents=selected_documents,
        )

        if not draft_answer:
            return self._build_result(
                query,
                retrieved_documents,
                selected_documents,
                None,
                None,
                [],
            )

        final_answer = draft_answer
        verification_result = None
        history = [
            {
                "stage": "retrieval",
                "original_count": sum(
                    d.get("retrieval_source") == "original_corpus"
                    for d in retrieved_documents
                ),
                "annual_count": sum(
                    d.get("retrieval_source") == "annual_corpus"
                    for d in retrieved_documents
                ),
                "selected_count": len(selected_documents),
            },
            {
                "stage": "answer_generation",
                "status": "completed",
            },
        ]

        # 3. Verify once. No second verification call.
        if self.verify_answers:

            try:
                verification_result = self.answer_verifier.verify(
                    query=query,
                    answer=draft_answer,
                    documents=selected_documents,
                )

                corrected_answer = (
                    verification_result.get("corrected_answer", "")
                    if isinstance(verification_result, dict)
                    else ""
                )

                # Use a correction if the verifier supplied one.
                # A failed verification does not erase the draft.
                if corrected_answer:
                    final_answer = corrected_answer

                history.append({
                    "stage": "answer_verification",
                    "status": verification_result.get("status"),
                    "supported": verification_result.get("supported"),
                })

            except Exception as error:
                print(f"Answer verification failed: {error}")
                history.append({
                    "stage": "answer_verification",
                    "status": "error",
                })

        return self._build_result(
            query,
            retrieved_documents,
            selected_documents,
            final_answer,
            verification_result,
            history,
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    @staticmethod
    def _build_result(
        query,
        retrieved_documents,
        selected_documents,
        answer,
        verification,
        history,
    ):

        return {
            "query": query,
            "final_search_query": query,
            "retrieved_documents": retrieved_documents,
            "selected_documents": selected_documents,
            "answer": answer,
            "evidence_check": None,
            "verification": verification,
            "retry_count": 0,
            "history": history,
        }
