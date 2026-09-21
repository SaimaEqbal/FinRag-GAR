import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class SelectionAgent:

    def __init__(self, model="openai/gpt-oss-20b"):

        self.client = Groq()
        self.model = model

    def select(self, query, documents):

        if not documents:
            return []

        formatted_documents = []

        for i, doc in enumerate(documents):

            # Limit the amount of document text sent to Groq.
            # This prevents the request from exceeding the TPM limit.
            text = doc.get("text", "")
            text = text[:2500]

            formatted_documents.append(
                f"""
### DOCUMENT {i}

Document ID:
{doc.get("id", "")}

Title:
{doc.get("title", "")}

Text:
{text}
"""
            )

        documents_text = "\n".join(formatted_documents)

        system_prompt = """
You are a financial document selection agent.

Your task is to identify which of the retrieved documents contain
information useful for answering the user's financial question.

A document is relevant if it:
- directly contains information needed to answer the question, OR
- contains financial data needed for calculations, OR
- provides necessary context for reasoning.

Return ONLY a valid JSON array containing the indices of relevant documents.

Example:
[0, 3]

If no document is relevant, return:
[]

Do not provide explanations.
Do not use markdown.
Do not include any text outside the JSON array.
"""

        user_prompt = f"""
Question:
{query}

Retrieved documents:
{documents_text}
"""

        try:

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

            print("\nSelection Agent raw response:")
            print(content)

            # -----------------------------------------
            # Parse JSON response
            # -----------------------------------------

            selected_indices = json.loads(content)

            if not isinstance(selected_indices, list):
                raise ValueError(
                    "Selection response is not a list."
                )

            valid_indices = []

            for index in selected_indices:

                try:

                    index = int(index)

                    if (
                        0 <= index < len(documents)
                        and index not in valid_indices
                    ):
                        valid_indices.append(index)

                except (ValueError, TypeError):
                    continue

            selected_indices = valid_indices[:5]

        except Exception as error:

            print("\nSelection Agent error:")
            print(error)

            # Safe fallback:
            # If the LLM fails, keep the BM25-ranked documents.
            selected_indices = list(range(len(documents)))

        selected_documents = [
            documents[i]
            for i in selected_indices
        ]

        return selected_documents