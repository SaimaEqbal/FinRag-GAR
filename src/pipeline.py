
from .retriever import BM25Retriever
from .selection_agent import SelectionAgent
from .answer_agent import AnswerAgent
from .evidence_checker import EvidenceChecker
from .query_rewriter import QueryRewriter


class GARRAGPipeline:

    def __init__(self, max_retries=2):
        self.selection_agent = SelectionAgent()
        self.answer_agent = AnswerAgent()
        self.evidence_checker = EvidenceChecker()
        self.query_rewriter = QueryRewriter()

        self.retriever = None
        self.max_retries = max_retries

    def setup(
        self,
        corpus,
        bm25_path="indexes/bm25.pkl",
        embedding_path="indexes/bge_embeddings.npy"
    ):
        self.retriever = BM25Retriever(
            corpus=corpus,
            index_path=None,
            embedding_path=embedding_path,
        )

        if bm25_path is not None:
            self.retriever.save(bm25_path)

    def setup_from_index(
        self,
        corpus,
        bm25_path,
        embedding_path
    ):
        self.retriever = BM25Retriever(
            corpus=corpus,
            index_path=bm25_path,
            embedding_path=embedding_path,
        )

    def save_index(self, index_path):
        if self.retriever is None:
            raise RuntimeError(
                "Pipeline has not been initialized."
            )

        self.retriever.save(index_path)

    def run(self, query, top_k=10):

        if self.retriever is None:
            raise RuntimeError(
                "Pipeline has not been initialized."
            )

        original_query = query
        current_query = query
        retry_count = 0
        history = []

        retrieved_documents = []
        selected_documents = []
        evidence_result = None

        # Initial retrieval + up to max_retries additional attempts
        while True:

            print(
                f"\nRetrieval attempt {retry_count + 1}"
                f" | Query: {current_query}"
            )

            # 1. Retrieve documents
            retrieved_documents = self.retriever.retrieve(
                current_query,
                top_k=top_k,
            )

            print(
                f"Retrieved {len(retrieved_documents)} documents."
            )

            # 2. Select relevant documents
            selected_documents = self.selection_agent.select(
                current_query,
                retrieved_documents,
            )

            print(
                f"Selection Agent kept "
                f"{len(selected_documents)} documents."
            )

            # 3. Check evidence sufficiency
            evidence_result = self.evidence_checker.check_evidence(
                current_query,
                selected_documents,
            )

            history.append({
                "attempt": retry_count + 1,
                "query": current_query,
                "retrieved_count": len(retrieved_documents),
                "selected_count": len(selected_documents),
                "evidence_check": evidence_result,
            })

            print(
                "Evidence sufficient:",
                evidence_result["sufficient"],
            )
            print(
                "Evidence check:",
                evidence_result["reason"],
            )

            # Stop when the evidence is sufficient
            if evidence_result["sufficient"]:
                break

            # Stop when the retry limit has been reached
            if retry_count >= self.max_retries:
                print("Retry limit reached.")
                break

            # 4. Rewrite query and retry retrieval
            try:
                rewritten_query = self.query_rewriter.rewrite_query(
                    query=original_query,
                    missing_information=(
                        evidence_result["missing_information"]
                    ),
                    previous_query=current_query,
                )

            except Exception as e:
                print(f"Query rewriting failed: {e}")
                break

            # Avoid repeating an identical search
            if (
                not rewritten_query
                or rewritten_query.strip().lower()
                == current_query.strip().lower()
            ):
                print(
                    "Query was not improved. "
                    "Stopping retries."
                )
                break

            current_query = rewritten_query.strip()
            retry_count += 1

        # 5. Generate answer only if evidence is sufficient
        if evidence_result and evidence_result["sufficient"]:

            answer = self.answer_agent.answer(
                original_query,
                selected_documents,
            )

        else:

            answer = (
                "I could not find sufficient evidence in the "
                "available documents to answer this question "
                "reliably. Please try rephrasing the question "
                "or providing additional sources."
            )

        return {
            "query": original_query,
            "final_search_query": current_query,
            "retrieved_documents": retrieved_documents,
            "selected_documents": selected_documents,
            "answer": answer,
            "evidence_check": evidence_result,
            "retry_count": retry_count,
            "history": history,
        }
