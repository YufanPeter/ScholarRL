"""BM25 retriever over the subset corpus.

    from scholarrl.retriever import BM25Retriever
    r = BM25Retriever.load()                              # after building the index
    r.search("retrieval augmented generation", k=3)      # -> [(paper_id, score), ...]
    r.get(paper_id)                                       # -> {"title", "abstract"}

BM25 is a fixed, non-learned, literal-match tool — weak so the agent must
learn to rewrite queries. Build the index once with `python -m scripts.build_index`.
"""
from .bm25 import BM25Retriever, INDEX_DIR

__all__ = ["BM25Retriever", "INDEX_DIR"]
