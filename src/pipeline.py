
import re

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

        # Original financial corpus
        self.retriever = None

        # Annual FY2025-26 financial corpus
        self.annual_retriever = None
        self.annual_corpus = None

    # ========================================================
    # ORIGINAL CORPUS SETUP
    # ========================================================

    def setup(
        self,
        corpus,
        bm25_path="indexes/bm25.pkl",
        embedding_path="indexes/bge_embeddings.npy",
    ):
        """Load the original corpus and its retrieval indexes."""

        self.retriever = BM25Retriever(
            corpus=corpus,
            bm25_path=bm25_path,
            embeddings_path=embedding_path,
        )

    # ========================================================
    # ANNUAL CORPUS SETUP
    # ========================================================

    def setup_annual_corpus(
        self,
        corpus,
        bm25_path,
        embedding_path,
    ):
        """Load the annual corpus and its separate indexes."""

        self.annual_corpus = corpus

        self.annual_retriever = BM25Retriever(
            corpus=corpus,
            bm25_path=bm25_path,
            embeddings_path=embedding_path,
        )

    # ========================================================
    # SETUP BOTH RETRIEVAL SYSTEMS
    # ========================================================

    def setup_from_index(
        self,
        corpus,
        bm25_path="indexes/bm25.pkl",
        embedding_path="indexes/bge_embeddings.npy",
        annual_corpus=None,
        annual_bm25_path=None,
        annual_embedding_path=None,
    ):
        """Initialize original and annual retrieval systems."""

        # Load original corpus
        self.setup(
            corpus=corpus,
            bm25_path=bm25_path,
            embedding_path=embedding_path,
        )

        # Load annual corpus if provided
        if annual_corpus is not None:

            if not annual_bm25_path or not annual_embedding_path:
                raise ValueError(
                    "Annual BM25 and embedding paths must be provided."
                )

            self.setup_annual_corpus(
                corpus=annual_corpus,
                bm25_path=annual_bm25_path,
                embedding_path=annual_embedding_path,
            )

    # ========================================================
    # CHECK WHETHER QUERY MENTIONS ANNUAL CORPUS COMPANY
    # ========================================================

    def _matches_annual_corpus_company(self, query):
        """
        Check whether the query mentions a company
        available in the annual corpus.
        """

        if not self.annual_corpus:
            return False

        query_lower = query.lower()

        company_names = set()

        for document in self.annual_corpus.values():

            company = str(
                document.get("company", "")
            ).strip()

            if company:
                company_names.add(company)

                # Common company-name aliases
                if company.lower() == "reliance industries":
                    company_names.add("reliance")

                elif company.lower() == "tata consultancy services":
                    company_names.add("tcs")

        for company_name in company_names:

            pattern = (
                r"\b"
                + re.escape(company_name.lower())
                + r"\b"
            )

            if re.search(pattern, query_lower):
                return True

        return False

    # ========================================================
    # CHECK EVIDENCE SUFFICIENCY
    # ========================================================

    @staticmethod
    def _is_evidence_sufficient(evidence_result):
        """
        Check whether the evidence assessment is sufficient.

        Uses the existing dual-assessment and conflict checks.
        """

        if not isinstance(evidence_result, dict):
            return False

        if evidence_result.get("status") != "ok":
            return False

        if evidence_result.get("sufficient") is not True:
            return False

        assessment_1 = evidence_result.get("assessment_1")
        assessment_2 = evidence_result.get("assessment_2")

        if assessment_1 is not None or assessment_2 is not None:

            if not (
                isinstance(assessment_1, dict)
                and isinstance(assessment_2, dict)
                and assessment_1.get("status") == "ok"
                and assessment_2.get("status") == "ok"
                and assessment_1.get("sufficient") is True
                and assessment_2.get("sufficient") is True
            ):
                return False

        conflicts = evidence_result.get(
            "potential_conflicts", []
        )

        if not isinstance(conflicts, list):
            return False

        blocking_conflict = any(
            isinstance(conflict, dict)
            and conflict.get("classification")
            in {"confirmed", "unresolved"}
            for conflict in conflicts
        )

        return not blocking_conflict

    # ========================================================
    # CHECK WHETHER QUERY IS QUARTERLY
    # ========================================================

    @staticmethod
    def _is_quarterly_query(query):

        query_lower = query.lower()

        quarterly_terms = [
            "quarter",
            "quarterly",
            "q1",
            "q2",
            "q3",
            "q4",
            "first quarter",
            "second quarter",
            "third quarter",
            "fourth quarter",
        ]

        return any(
            re.search(
                r"\b" + re.escape(term) + r"\b",
                query_lower,
            )
            for term in quarterly_terms
        )

    # ========================================================
    # RETRIEVAL AND EVIDENCE CHECKING
    # ========================================================

    def _retrieve_and_check(
        self,
        query,
        original_query,
        top_k,
    ):
        """
        Search annual corpus first when a matching company
        is mentioned and the question is not quarterly.

        If annual evidence is insufficient, search the
        original corpus.
        """

        annual_corpus_attempt = None

        if (
            self.annual_retriever is not None
            and self._matches_annual_corpus_company(query)
        ):
            print("\nSearching the annual financial corpus...")

            annual_retrieved = self.annual_retriever.retrieve(
                query=query,
                top_k=top_k,
            )

            annual_selected = self.selection_agent.select(
                query=query,
                documents=annual_retrieved,
            )

            if not annual_selected:
                annual_selected = annual_retrieved[:3]

            annual_evidence = self.evidence_checker.check_evidence(
                query=original_query,
                documents=annual_selected,
            )

            annual_corpus_attempt = {
                "source": "annual_FY2025_26_corpus",
                "evidence_status": annual_evidence.get("status"),
                "evidence_sufficient": annual_evidence.get("sufficient"),
                "reason": annual_evidence.get("reason"),
            }

            if self._is_evidence_sufficient(annual_evidence):

                print(
                    "Sufficient evidence found in the annual corpus."
                )

                return (
                    annual_retrieved,
                    annual_selected,
                    annual_evidence,
                    "annual_FY2025_26_corpus",
                    annual_corpus_attempt,
                )

            print(
                "Annual corpus evidence is insufficient. "
                "Falling back to the original corpus..."
            )

        # ----------------------------------------------------
        # Search original corpus
        # ----------------------------------------------------

        print("\nSearching the original financial corpus...")

        retrieved_documents = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        selected_documents = self.selection_agent.select(
            query=query,
            documents=retrieved_documents,
        )

        if not selected_documents:
            selected_documents = retrieved_documents[:3]

        evidence_result = self.evidence_checker.check_evidence(
            query=original_query,
            documents=selected_documents,
        )

        return (
            retrieved_documents,
            selected_documents,
            evidence_result,
            "original_corpus",
            annual_corpus_attempt,
        )

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(self, query, top_k=10):

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if self.retriever is None:
            raise RuntimeError(
                "Original retriever is not initialized. "
                "Call setup() first."
            )

        original_query = query.strip()
        current_query = original_query

        rewrite_used = False
        retry_count = 0
        history = []

        retrieved_documents = []
        selected_documents = []
        evidence_result = None

        # ----------------------------------------------------
        # Retrieval and evidence-assessment loop
        # ----------------------------------------------------

        while True:

            effective_top_k = top_k * (retry_count + 1)

            (
                retrieved_documents,
                selected_documents,
                evidence_result,
                retrieval_source,
                annual_corpus_attempt,
            ) = self._retrieve_and_check(
                query=current_query,
                original_query=original_query,
                top_k=effective_top_k,
            )

            decision = self.controller.decide(
                evidence_result=evidence_result,
                retry_count=retry_count,
                query_changed=(
                    current_query != original_query
                ),
                rewrite_used=rewrite_used,
            )

            history.append({
                "stage": "evidence_check",
                "query": current_query,
                "retrieval_source": retrieval_source,
                "annual_corpus_attempt": annual_corpus_attempt,
                "retry_count": retry_count,
                "rewrite_used": rewrite_used,
                "decision": decision,
            })

            action = decision["action"]

            if action == "GENERATE_ANSWER":
                break

            # ------------------------------------------------
            # Query rewriting
            # ------------------------------------------------

            if action == "REWRITE_QUERY":

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

            # ------------------------------------------------
            # Retrieve more evidence
            # ------------------------------------------------

            if action == "RETRIEVE_MORE":

                if retry_count >= self.max_retries:
                    break

                retry_count += 1
                continue

            # Includes ABSTAIN and unexpected actions
            break

        # ----------------------------------------------------
        # Stop if evidence or selected documents are missing
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Generate answer only when approved by controller
        # ----------------------------------------------------

        if (
            not history
            or history[-1]["decision"]["action"]
            != "GENERATE_ANSWER"
        ):

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

        # ----------------------------------------------------
        # Initiate answer verification
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Verify answer
        # ----------------------------------------------------

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

    # ========================================================
    # BUILD FINAL RESULT
    # ========================================================

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
