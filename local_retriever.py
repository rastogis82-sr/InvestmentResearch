"""
local_retriever.py
-------------------
A dependency-free stand-in for the FAISS + embeddings retriever, used only
to smoke-test `rag_node`'s logic (thresholding, company filtering, empty-
result handling) in this sandbox. It scores documents by simple keyword
overlap instead of vector similarity — good enough to validate control
flow, not meant to demonstrate retrieval quality.

The shipped Colab notebook replaces this with a real
`HuggingFaceEmbeddings` + `FAISS` retriever behind the SAME interface:

    retrieve(query, company_filter=None, k=4) -> List[RetrievedChunk]

so `rag_node`'s code does not need to change between this local test and
the real notebook — only the object passed in as `retriever` changes.
"""

from dataclasses import dataclass
from typing import List, Optional

from corpus import DOCUMENTS


@dataclass
class RetrievedChunk:
    text: str
    source: str
    company: Optional[str]
    score: float  # 0..1, higher = more relevant (mirrors normalized cosine similarity)


def _keyword_overlap_score(query: str, text: str) -> float:
    q_words = set(w.lower() for w in query.split() if len(w) > 2)
    d_words = set(w.lower().strip(".,:;()") for w in text.split() if len(w) > 2)
    if not q_words:
        return 0.0
    overlap = q_words & d_words
    return len(overlap) / len(q_words)


class LocalKeywordRetriever:
    """Interface-compatible stand-in for the FAISS retriever."""

    def __init__(self, score_threshold: float = 0.12):
        self.score_threshold = score_threshold
        self.documents = DOCUMENTS

    def retrieve(
        self, query: str, company_filter: Optional[str] = None, k: int = 4
    ) -> List[RetrievedChunk]:
        candidates = []
        for doc in self.documents:
            if company_filter and doc["company"] not in (company_filter, None):
                continue
            score = _keyword_overlap_score(query, doc["text"])
            if score >= self.score_threshold:
                candidates.append(
                    RetrievedChunk(
                        text=doc["text"],
                        source=doc["filename"],
                        company=doc["company"],
                        score=score,
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]


if __name__ == "__main__":
    retr = LocalKeywordRetriever()

    hits = retr.retrieve("ABC Technologies profitability margin", company_filter="ABC")
    print(f"Query with real signal -> {len(hits)} hits")
    for h in hits:
        print(f"  score={h.score:.2f} source={h.source}")
    assert hits, "expected at least one relevant chunk for a real query"

    empty = retr.retrieve("quantum teleportation recipe", company_filter="ABC")
    print(f"\nUnrelated query -> {len(empty)} hits (expect 0, exercising RAG-failure path)")
    assert not empty
