"""Pluggable retriever interface. BM25 is V1; dense (bge/e5 + FAISS) is a drop-in V2.

    class Retriever:
        def search(self, query: str, k: int) -> list[tuple[str, float]]: ...  # (paper_id, score)
        def get(self, paper_id: str) -> dict: ...                             # {title, abstract}

Swapping the impl behind this interface yields a free ablation:
how does retriever strength change the learned policy?
"""
