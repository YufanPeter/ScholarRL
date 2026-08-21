"""Build the subset retrieval corpus: gold papers + distractors -> papers.jsonl.

Each output line: {"paper_id": ..., "title": ..., "abstract": ...}

- gold papers use their arxiv_id as paper_id (so reward can match against answer_arxiv_id).
- distractors use their zip filename as paper_id (never collides with an arxiv id).

The corpus is a SUBSET (plan §2.2): every retrievable gold paper + a bounded set of random
distractors, sized so BM25 indexes in seconds. Distractor count = distractor_ratio * |gold|.
Hard negatives can be added later once a retriever exists.
"""
from __future__ import annotations

import json
import random
import zipfile
from typing import Dict, Optional

from ..paths import CORPUS_ZIP, PAPERS_JSONL, CORPUS_DIR
from ..data import (
    resolve_gold,
    zip_filenames,
    retrievable_gold_ids,
    gold_id_to_title,
)
from ..data.queries import _looks_like_title


def _read_paper(zf: zipfile.ZipFile, filename: str) -> Optional[dict]:
    """Read one paper entry from the zip; return {title, abstract} or None."""
    try:
        obj = json.loads(zf.read(filename))
    except (KeyError, json.JSONDecodeError):
        return None
    title = (obj.get("title") or "").strip()
    abstract = (obj.get("abstract") or "").strip()
    if not title and not abstract:
        return None
    return {"title": title, "abstract": abstract}


def build_corpus(
    distractor_ratio: int = 5,
    include_random_distractors: bool = True,
    seed: int = 42,
) -> Dict[str, int]:
    """Write the subset corpus to PAPERS_JSONL. Returns counts.

    distractor_ratio: random distractors per gold paper (difficulty knob).
    include_random_distractors: if False, corpus is gold-only (debug/easy setting).
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    names = zip_filenames()
    gold_ids = retrievable_gold_ids()          # only the ones actually in the zip (normalized)
    gold_titles = gold_id_to_title()           # normalized id -> good human title (backfill)

    # map each retrievable gold id -> its zip filename
    gold_filename: Dict[str, str] = {}
    for aid in gold_ids:
        res = resolve_gold(aid)
        if res.filename and res.filename in names:
            gold_filename[aid] = res.filename
    gold_files = set(gold_filename.values())

    # sample distractors from non-gold zip files
    if include_random_distractors:
        n_distractors = distractor_ratio * len(gold_filename)
        non_gold = [n for n in names if n not in gold_files]
        rng.shuffle(non_gold)
        distractor_files = non_gold[:n_distractors]
    else:
        distractor_files = []

    written_gold = 0
    written_distractor = 0
    skipped = 0
    backfilled = 0
    with zipfile.ZipFile(CORPUS_ZIP) as zf, open(PAPERS_JSONL, "w", encoding="utf-8") as out:
        # gold papers first (paper_id = arxiv_id)
        for aid, fname in gold_filename.items():
            paper = _read_paper(zf, fname)
            if paper is None:
                skipped += 1
                continue
            # Some zip entries have a junk section-header title ('1 Introduction').
            # Backfill from the human answer_titles so BM25 can actually find these gold papers.
            if not _looks_like_title(paper["title"]) and aid in gold_titles:
                paper["title"] = gold_titles[aid]
                backfilled += 1
            out.write(json.dumps({"paper_id": aid, **paper}, ensure_ascii=False) + "\n")
            written_gold += 1
        # distractors (paper_id = filename)
        for fname in distractor_files:
            paper = _read_paper(zf, fname)
            if paper is None:
                skipped += 1
                continue
            out.write(json.dumps({"paper_id": fname, **paper}, ensure_ascii=False) + "\n")
            written_distractor += 1

    return {
        "gold": written_gold,
        "distractors": written_distractor,
        "total": written_gold + written_distractor,
        "skipped": skipped,
        "backfilled_titles": backfilled,
    }
