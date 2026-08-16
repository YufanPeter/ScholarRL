"""BM25 retriever over the subset corpus (papers.jsonl), backed by bm25s.

BM25 is a fixed, non-learned tool: it scores documents by term frequency x inverse
document frequency (with length normalization) — pure literal word matching, no semantics.
This is deliberately a *weak* retriever, so the agent has room to learn query rewriting.

Index is built once and saved to disk (data/corpus/bm25_index/), then loaded in ms.

    r = BM25Retriever.load()
    r.search("retrieval augmented generation", k=3)   # -> [(paper_id, score), ...]
    r.get(paper_id)                                    # -> {"title": ..., "abstract": ...}
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s

from ..paths import PAPERS_JSONL, CORPUS_DIR

INDEX_DIR = CORPUS_DIR / "bm25_index"
_META_FILE = "papers_meta.jsonl"   # paper_id, title, abstract — parallel to index order


def _doc_text(title: str, abstract: str) -> str:
    """Text fed to BM25 for one paper."""
    return f"{title} {abstract}".strip()


class BM25Retriever:
    def __init__(self, model: bm25s.BM25, paper_ids: List[str], meta: Dict[str, dict]):
        self._model = model
        self._paper_ids = paper_ids          # index position -> paper_id
        self._meta = meta                    # paper_id -> {title, abstract}

    # ---- construction --------------------------------------------------------
    @classmethod
    def build_index(cls, papers_jsonl: Path = PAPERS_JSONL, index_dir: Path = INDEX_DIR) -> "BM25Retriever":
        """Read papers.jsonl, build the BM25 index, and save it to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)

        paper_ids: List[str] = []
        corpus_texts: List[str] = []
        meta: Dict[str, dict] = {}
        with open(papers_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pid = r["paper_id"]
                paper_ids.append(pid)
                corpus_texts.append(_doc_text(r.get("title", ""), r.get("abstract", "")))
                meta[pid] = {"title": r.get("title", ""), "abstract": r.get("abstract", "")}

        print(f"tokenizing {len(corpus_texts)} papers (lowercase, english stopwords)...")
        t0 = time.time()
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="english", show_progress=True)
        model = bm25s.BM25()
        model.index(corpus_tokens, show_progress=True)
        print(f"indexed in {time.time() - t0:.1f}s")

        # persist: bm25s index + our parallel meta
        model.save(str(index_dir))
        with open(index_dir / _META_FILE, "w", encoding="utf-8") as out:
            for pid in paper_ids:
                out.write(json.dumps({"paper_id": pid, **meta[pid]}, ensure_ascii=False) + "\n")

        return cls(model, paper_ids, meta)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "BM25Retriever":
        """Load a previously built index from disk (fast)."""
        meta_path = index_dir / _META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no BM25 index at {index_dir}. Run: python -m scripts.build_index"
            )
        model = bm25s.BM25.load(str(index_dir), load_corpus=False)
        paper_ids: List[str] = []
        meta: Dict[str, dict] = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pid = r["paper_id"]
                paper_ids.append(pid)
                meta[pid] = {"title": r["title"], "abstract": r["abstract"]}
        return cls(model, paper_ids, meta)

    # ---- query ---------------------------------------------------------------
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return up to k (paper_id, score), highest score first."""
        query_tokens = bm25s.tokenize(query, stopwords="english", show_progress=False)
        k = min(k, len(self._paper_ids))
        results, scores = self._model.retrieve(
            query_tokens, k=k, show_progress=False, return_as="tuple"
        )
        # results/scores shape: (1, k) — one query
        out: List[Tuple[str, float]] = []
        for idx, score in zip(results[0], scores[0]):
            out.append((self._paper_ids[int(idx)], float(score)))
        return out

    def get(self, paper_id: str) -> dict:
        """Return {title, abstract} for a paper_id (empty dict if unknown)."""
        return self._meta.get(paper_id, {})

    def __len__(self) -> int:
        return len(self._paper_ids)
