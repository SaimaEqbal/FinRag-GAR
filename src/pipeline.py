from .retriever import BM25Retriever
from .selection_agent import SelectionAgent
from .answer_agent import AnswerAgent


class GARRAGPipeline:

    def __init__(self):

        self.selection_agent = SelectionAgent()
        self.answer_agent = AnswerAgent()

        self.retriever = None

    def setup(self, corpus):

        self.retriever = BM25Retriever(corpus)

    def run(self, query, top_k=10):

        if self.retriever is None:
            raise RuntimeError(
                "Pipeline has not been initialized with a corpus."
            )

        # 1. Retrieve
        retrieved_documents = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        print(
            f"\nRetrieved "
            f"{len(retrieved_documents)} documents."
        )

        # 2. Create candidate pool for GAR selection
        # BM25 retrieves top 10, but only top 5 are
        # sent to the Selection Agent.
        selection_candidates = retrieved_documents[:5]

        print(
            f"Sending "
            f"{len(selection_candidates)} documents "
            f"to Selection Agent."
        )

        # 3. Select relevant documents
        selected_documents = self.selection_agent.select(
            query,
            selection_candidates,
        )

        print(
            f"Selection Agent kept "
            f"{len(selected_documents)} documents."
        )

        # 4. Generate answer
        answer = self.answer_agent.answer(
            query,
            selected_documents,
        )

        return {
            "query": query,
            "retrieved_documents": retrieved_documents,
            "selected_documents": selected_documents,
            "answer": answer,
        }