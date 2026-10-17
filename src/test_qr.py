
from query_rewriter import QueryRewriter

rewriter = QueryRewriter()

original_question = (
    "What is the annual increase in interest expense "
    "if LIBOR rises by 100 basis points?"
)

missing_information = (
    "The documents do not contain the actual change "
    "in annual interest expense."
)

rewritten_query = rewriter.rewrite_query(
    query=original_question,
    missing_information=missing_information,
)

print("Original query:", original_question)
print("Rewritten query:", rewritten_query)
