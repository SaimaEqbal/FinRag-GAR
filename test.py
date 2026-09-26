import json
import numpy as np

from src.retriever import BM25Retriever


# ============================================================
# PATHS
# ============================================================

CORPUS_PATH = "data/unified_corpus.json"
BM25_PATH = "indexes/bm25.pkl"
EMBEDDING_PATH = "indexes/bge_embeddings.npy"


# ============================================================
# LOAD CORPUS
# ============================================================

print("Loading corpus...")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Loaded {len(corpus)} documents.")


# ============================================================
# LOAD RETRIEVER
# ============================================================

retriever = BM25Retriever(
    corpus=corpus,
    index_path=BM25_PATH,
    embedding_path=EMBEDDING_PATH,
)


# ============================================================
# QUERY
# ============================================================

query = "What was Microsoft's revenue in fiscal year 2023?"

target_id = "FinDER_MSFT20230541"


# ============================================================
# FIND TARGET DOCUMENT INDEX
# ============================================================

target_index = retriever.doc_ids.index(target_id)


# ============================================================
# BM25 RANKING
# ============================================================

print("\nCalculating BM25 ranking...")

tokenized_query = query.lower().split()

bm25_scores = retriever.bm25.get_scores(tokenized_query)

bm25_rank = (
    np.sum(bm25_scores > bm25_scores[target_index]) + 1
)


# ============================================================
# BGE RANKING
# ============================================================

print("Calculating BGE ranking...")

query_embedding = retriever.embedding_model.encode(
    query,
    normalize_embeddings=True,
)

query_embedding = np.asarray(
    query_embedding,
    dtype=np.float32,
)

embedding_scores = np.dot(
    retriever.embeddings,
    query_embedding,
)

embedding_rank = (
    np.sum(
        embedding_scores > embedding_scores[target_index]
    ) + 1
)


# ============================================================
# CHECK BGE TOP 500
# ============================================================

bge_top_500 = embedding_scores.argsort()[::-1][:500]

target_in_top_500 = target_index in bge_top_500


# ============================================================
# CHECK BGE TOP 50
# ============================================================

bge_top_50 = embedding_scores.argsort()[::-1][:50]

target_in_top_50 = target_index in bge_top_50


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TARGET DOCUMENT RETRIEVAL DIAGNOSTIC")
print("=" * 60)

print("\nTarget document:")
print("ID:", target_id)
print("TITLE:", corpus[target_id]["title"])


print("\n" + "-" * 60)
print("BM25")
print("-" * 60)

print("Score:", bm25_scores[target_index])
print("Rank:", bm25_rank)


print("\n" + "-" * 60)
print("BGE")
print("-" * 60)

print("Score:", embedding_scores[target_index])
print("Rank:", embedding_rank)


print("\n" + "-" * 60)
print("BGE CANDIDATE CHECK")
print("-" * 60)

if target_in_top_50:
    print("✅ Target IS in BGE top 50.")
else:
    print("❌ Target is NOT in BGE top 50.")

if target_in_top_500:
    print("✅ Target IS in BGE top 500.")
else:
    print("❌ Target is NOT in BGE top 500.")


# ============================================================
# SHOW DOCUMENT
# ============================================================

print("\n" + "=" * 60)
print("TARGET DOCUMENT TEXT")
print("=" * 60)

print(corpus[target_id]["text"][:1500])


# ============================================================
# SHOW TOP 10 BGE DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 BGE DOCUMENTS")
print("=" * 60)

top_10_bge = embedding_scores.argsort()[::-1][:10]

for rank, index in enumerate(top_10_bge, start=1):

    doc_id = retriever.doc_ids[index]

    print(f"\n{rank}.")
    print("ID:", doc_id)
    print("TITLE:", retriever.corpus[doc_id]["title"])
    print("BGE SCORE:", embedding_scores[index])