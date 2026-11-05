
import os
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from groq import Groq


class QueryRewriter:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")

        self.client = Groq(api_key=api_key)

        self.model_1 = os.getenv(
            "GROQ_MODEL_1",
            "openai/gpt-oss-20b",
        )
        self.model_2 = os.getenv("GROQ_MODEL_2")

        if not self.model_2:
            raise ValueError("GROQ_MODEL_2 is not set.")

        if self.model_1 == self.model_2:
            raise ValueError(
                "GROQ_MODEL_1 and GROQ_MODEL_2 must be different."
            )

    def _rewrite_with_model(
        self,
        model,
        query,
        missing_information,
    ):
        """Generate one rewritten financial query using a single model."""

        prompt = f"""
            Improve the following financial question for document retrieval.

            Original question:
            {query}

            Retrieval feedback:
            {missing_information}

            Instructions:
            - Preserve the original meaning and intent.
            - Preserve all company names, financial metrics, dates,
            reporting periods and numerical references.
            - Use relevant alternative financial terminology where useful.
            - Do not introduce new entities, dates or assumptions.
            - Do not narrow the question to only the missing information.
            - Keep the rewritten question clear and natural.
            - Return only the rewritten question.
        """

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite financial questions for "
                            "retrieval-augmented generation. "
                            "Return only the rewritten query."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
            )

            rewritten_query = (
                response.choices[0].message.content or ""
            ).strip().strip('"').strip("'")

            if (
                not rewritten_query
            ):
                return None

            # Avoid returning the original query as a rewrite.
            if rewritten_query.lower() == query.strip().lower():
                return None

            return rewritten_query

        except Exception as error:
            print(
                f"[Query Rewriter] Model {model} failed: {error}"
            )
            return None

    def rewrite_queries(self, query, missing_information):
        """
        Generate two rewrite candidates in one rewrite round.

        Both models receive the same original query.
        Candidate selection is handled by agentic_controller.py.
        """

        print("\n========== QUERY REWRITING ==========")
        print(f"Original query:\n{query}\n")

        # Run both model calls concurrently.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(
                self._rewrite_with_model,
                self.model_1,
                query,
                missing_information,
            )

            future_2 = executor.submit(
                self._rewrite_with_model,
                self.model_2,
                query,
                missing_information,
            )

            candidate_1 = future_1.result()
            candidate_2 = future_2.result()

        print(f"LLM 1 ({self.model_1}):")
        print(candidate_1 or "No valid rewrite generated.")

        print(f"\nLLM 2 ({self.model_2}):")
        print(candidate_2 or "No valid rewrite generated.")

        print("=====================================\n")

        # Do not select a candidate here.
        # The controller makes that decision.
        return {
            "original_query": query,
            "candidates": [
                {
                    "model": self.model_1,
                    "query": candidate_1,
                },
                {
                    "model": self.model_2,
                    "query": candidate_2,
                },
            ],
        }
