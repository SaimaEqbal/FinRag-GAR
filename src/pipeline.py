
from .retriever import BM25Retriever
from .selection_agent import SelectionAgent
from .answer_agent import AnswerAgent
from .evidence_checker import EvidenceChecker
from .query_rewriter import QueryRewriter
from .answer_verifier import AnswerVerifier
from .agentic_controller import AgenticRetrievalController


class GARRAGPipeline:
    def __init__(self, max_retries=2):
        self.selection_agent = SelectionAgent()
        self.answer_agent = AnswerAgent()
        self.evidence_checker = EvidenceChecker()
        self.query_rewriter = QueryRewriter()
        self.answer_verifier = AnswerVerifier()

        self.controller = AgenticRetrievalController(
            max_retries=max_retries
        )

        self.max_retries = max_retries
        self.retriever = None

    def setup(
        self,
        corpus,
        bm25_path="indexes/bm25.pkl",
        embedding_path="indexes/bge_embeddings.npy",
    ):
        """Load the existing BM25 index and embedding matrix."""
        self.retriever = BM25Retriever(
            corpus=corpus,
            bm25_path=bm25_path,
            embeddings_path=embedding_path,
        )

    def setup_from_index(
        self,
        corpus,
        bm25_path="indexes/bm25.pkl",
        embedding_path="indexes/bge_embeddings.npy",
    ):
        """Load an existing retrieval index."""
        self.setup(
            corpus=corpus,
            bm25_path=bm25_path,
            embedding_path=embedding_path,
        )

    def run(self, query, top_k=10):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.retriever is None:
            raise RuntimeError(
                "Retriever is not initialized. Call setup() first."
            )

        original_query = query.strip()
        current_query = original_query

        rewrite_used = False
        retry_count = 0
        history = []

        retrieved_documents = []
        selected_documents = []
        evidence_result = None

        # Retrieval and evidence-assessment loop.
        while True:
            effective_top_k = top_k * (retry_count + 1)

            retrieved_documents = self.retriever.retrieve(
                query=current_query,
                top_k=effective_top_k,
            )

            selected_documents = self.selection_agent.select(
                query=current_query,
                documents=retrieved_documents,
            )

            if not selected_documents:
                selected_documents = retrieved_documents[:3]

            evidence_result = self.evidence_checker.check_evidence(
                query=original_query,
                documents=selected_documents,
            )

            decision = self.controller.decide(
                evidence_result=evidence_result,
                retry_count=retry_count,
                query_changed=(current_query != original_query),
                rewrite_used=rewrite_used,
            )

            history.append({
                "stage": "evidence_check",
                "query": current_query,
                "retry_count": retry_count,
                "rewrite_used": rewrite_used,
                "decision": decision,
            })

            action = decision["action"]

            if action == "GENERATE_ANSWER":
                break

            if action == "REWRITE_QUERY":
                # The rewrite stage can run only once per pipeline run.
                if rewrite_used:
                    return self._build_result(
                        original_query,
                        current_query,
                        retrieved_documents,
                        selected_documents,
                        None,
                        evidence_result,
                        None,
                        retry_count,
                        history,
                    )

                rewrite_result = self.query_rewriter.rewrite_queries(
                    query=original_query,
                    missing_information=evidence_result.get(
                        "missing_information", ""
                    ),
                )

                rewrite_decision = self.controller.select_rewrite(
                    original_query=original_query,
                    rewrite_result=rewrite_result,
                )

                history.append({
                    "stage": "query_rewrite",
                    "decision": rewrite_decision,
                })

                rewrite_used = True

                if rewrite_decision["action"] == "ABSTAIN":
                    return self._build_result(
                        original_query,
                        current_query,
                        retrieved_documents,
                        selected_documents,
                        None,
                        evidence_result,
                        None,
                        retry_count,
                        history,
                    )

                current_query = rewrite_decision["selected_query"]
                retry_count += 1
                continue

            if action == "RETRIEVE_MORE":
                if retry_count >= self.max_retries:
                    break

                retry_count += 1
                continue

            # Includes ABSTAIN and any unexpected action.
            break

        if not evidence_result or not selected_documents:
            return self._build_result(
                original_query,
                current_query,
                retrieved_documents,
                selected_documents,
                None,
                evidence_result,
                None,
                retry_count,
                history,
            )

        # Do not generate an answer unless the controller approved it.
        if not history or history[-1]["decision"]["action"] != "GENERATE_ANSWER":
            return self._build_result(
                original_query,
                current_query,
                retrieved_documents,
                selected_documents,
                None,
                evidence_result,
                None,
                retry_count,
                history,
            )

        draft_answer = self.answer_agent.answer(
            query=original_query,
            documents=selected_documents,
        )

        # Ask the controller to initiate answer verification.
        verification_action = self.controller.decide(
            answer=draft_answer,
        )

        history.append({
            "stage": "answer_generation",
            "decision": verification_action,
        })

        if verification_action["action"] != "VERIFY_ANSWER":
            return self._build_result(
                original_query,
                current_query,
                retrieved_documents,
                selected_documents,
                draft_answer,
                evidence_result,
                None,
                retry_count,
                history,
            )

        verification_result = self.answer_verifier.verify(
            query=original_query,
            answer=draft_answer,
            documents=selected_documents,
        )

        verification_decision = self.controller.decide(
            verification_result=verification_result,
            retry_count=retry_count,
        )

        history.append({
            "stage": "answer_verification",
            "decision": verification_decision,
        })

        final_answer = draft_answer

        if verification_decision["action"] == "ACCEPT_ANSWER":
            pass

        elif verification_decision["action"] == "CORRECT_ANSWER":
            corrected_answer = verification_decision.get(
                "corrected_answer"
            )

            if corrected_answer:
                second_verification = self.answer_verifier.verify(
                    query=original_query,
                    answer=corrected_answer,
                    documents=selected_documents,
                )

                second_decision = self.controller.decide(
                    verification_result=second_verification,
                    retry_count=retry_count,
                )

                history.append({
                    "stage": "corrected_answer_verification",
                    "decision": second_decision,
                })

                if second_decision["action"] == "ACCEPT_ANSWER":
                    final_answer = corrected_answer
                    verification_result = second_verification
                else:
                    final_answer = (
                        "Unable to provide a fully verified answer "
                        "from the available evidence."
                    )
                    verification_result = second_verification

        else:
            final_answer = (
                "Unable to provide a fully verified answer "
                "from the available evidence."
            )

        return self._build_result(
            original_query,
            current_query,
            retrieved_documents,
            selected_documents,
            final_answer,
            evidence_result,
            verification_result,
            retry_count,
            history,
        )

    @staticmethod
    def _build_result(
        query,
        final_search_query,
        retrieved_documents,
        selected_documents,
        answer,
        evidence_check,
        verification,
        retry_count,
        history,
    ):
        return {
            "query": query,
            "final_search_query": final_search_query,
            "retrieved_documents": retrieved_documents,
            "selected_documents": selected_documents,
            "answer": answer,
            "evidence_check": evidence_check,
            "verification": verification,
            "retry_count": retry_count,
            "history": history,
        }
