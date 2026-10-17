
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class QueryRewriter:
    def __init__(self, model="openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY was not found.")

        self.client = Groq(api_key=api_key)
        self.model = model

    def rewrite_query(self, query, missing_information, previous_query=None):
        """
        Rewrite a query to improve retrieval when the available
        evidence is insufficient.
        """

        prompt = f"""
You are a query-rewriting component in a RAG system.

Original question:
{query}

Previous search query:
{previous_query or query}

Missing information identified by the evidence checker:
{missing_information}

Rewrite the search query to help retrieve documents containing
the missing information.

Rules:
- Preserve the original question's meaning.
- Include important entities, figures, dates, and technical terms.
- Focus on the missing information.
- Do not answer the question.
- Return only the rewritten search query, without explanation.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite search queries clearly and precisely.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        rewritten_query = response.choices[0].message.content.strip()

        if not rewritten_query:
            return query

        return rewritten_query
