from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class AnswerAgent:

    def __init__(self, model="openai/gpt-oss-20b"):

        self.client = Groq()
        self.model = model

    def answer(self, query, documents):

        if not documents:
            return "No relevant documents were found."

        formatted_documents = []

        for i, doc in enumerate(documents):

            # Limit document size to avoid exceeding Groq limits.
            text = doc.get("text", "")
            text = text[:5000]

            formatted_documents.append(
                f"""
### DOCUMENT {i + 1}

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
You are a financial question-answering assistant.

Answer the user's question using the provided financial documents.

Rules:

1. Use the documents as the primary source of information.
2. Do not invent facts that are not supported by the documents.
3. Perform calculations when necessary.
4. Show the calculation clearly when the question requires arithmetic.
5. Give the final answer clearly.
6. If the documents do not contain enough information, say so.
7. Do not use outside knowledge.
8. Do not add unnecessary closing remarks.
"""

        user_prompt = f"""
Question:
{query}

Relevant financial documents:
{documents_text}

Answer the question using only the information contained
in the documents.
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

            answer = response.choices[0].message.content.strip()

            return answer

        except Exception as error:

            print("\nAnswer Agent error:")
            print(error)

            return "Unable to generate an answer."