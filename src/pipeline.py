from .retriever import BM25Retriever
from .selection_agent import SelectionAgent
from .answer_agent import AnswerAgent


class GARRAGPipeline:

    def __init__(self):
        self.selection_agent = SelectionAgent()
        self.answer_agent = AnswerAgent()
        self.retriever = None

    def setup(
        self,
        corpus,
        bm25_path=None,
        embedding_path=None
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

        retrieved_documents = self.retriever.retrieve(
            query,
            top_k=top_k,
        )

        print(
            f"\nRetrieved "
            f"{len(retrieved_documents)} documents."
        )

        selected_documents = self.selection_agent.select(
            query,
            retrieved_documents,
        )

        print(
            f"Selection Agent kept "
            f"{len(selected_documents)} documents."
        )

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