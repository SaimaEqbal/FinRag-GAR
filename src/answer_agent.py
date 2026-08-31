from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AnswerAgent:

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def answer(self, query, documents):

        formatted_documents = []

        for i, doc in enumerate(documents):
            formatted_documents.append(
                f"""
### DOCUMENT {i + 1}

Title:
{doc["title"]}

Text:
{doc["text"]}
"""
            )

        documents_text = "\n".join(formatted_documents)

        system_prompt = """
You are a financial question-answering assistant.

Answer the user's question using the provided financial documents.

Requirements:
- Use the documents as the primary source of information.
- Do not invent facts that are not supported by the documents.
- Perform calculations when necessary.
- Explain calculations clearly when appropriate.
- Give a direct and concise answer.
- Do not add unnecessary closing remarks.
"""

        user_prompt = f"""
Question:
{query}

Documents:
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

        return response.choices[0].message.content