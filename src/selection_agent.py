import json
from openai import OpenAI


class SelectionAgent:

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def select(self, query, documents):

        formatted_documents = []

        for i, doc in enumerate(documents):
            formatted_documents.append(
                f"""
### DOCUMENT {i}

Title:
{doc["title"]}

Text:
{doc["text"]}
"""
            )

        documents_text = "\n".join(formatted_documents)

        system_prompt = """
You are a financial document selection agent.

Your task is to identify which retrieved documents contain information
that is useful for answering the user's question.

A document is relevant if its information can help answer the question,
directly or through necessary calculations or reasoning.

Return ONLY a JSON list of document indices.

Example:
[0, 3, 7]

Do not provide explanations.
"""

        user_prompt = f"""
Question:
{query}

Retrieved documents:
{documents_text}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        try:
            selected_indices = json.loads(content)

            if not isinstance(selected_indices, list):
                raise ValueError("Selection is not a list.")

            selected_indices = [
                int(i)
                for i in selected_indices
                if 0 <= int(i) < len(documents)
            ]

        except Exception as error:
            print(f"Selection parsing error: {error}")
            print(f"Model response: {content}")

            # Safe fallback: use all retrieved documents
            selected_indices = list(range(len(documents)))

        return [
            documents[i]
            for i in selected_indices
        ]