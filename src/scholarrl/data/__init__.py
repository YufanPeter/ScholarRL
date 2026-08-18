"""Load AutoScholarQuery, id2paper mapping, and resolve gold papers.

Public API:
    load_queries(split)      -> list[QueryRecord]      ('train'|'dev'|'test')
    all_gold_ids()           -> set[str]               union of gold ids, all splits
    load_id2paper()          -> dict[str, str]         arxiv_id -> title (cached)
    norm_arxiv_id(id)        -> str                    id2paper-key form
    norm_title(title)        -> str                    zip-filename form
    resolve_gold(id)         -> GoldResolution         id -> title -> candidate filename

The chain: answer_arxiv_id -> norm_arxiv_id -> id2paper title -> norm_title -> zip filename.
This module stops at the filename string; reading the zip is the corpus module's job.
"""
from .queries import QueryRecord, load_queries, all_gold_ids, gold_id_to_title
from .id2paper import load_id2paper
from .normalize import norm_arxiv_id, norm_title
from .resolve import GoldResolution, resolve_gold
from .retrievable import zip_filenames, retrievable_gold_ids

__all__ = [
    "QueryRecord", "load_queries", "all_gold_ids", "gold_id_to_title",
    "load_id2paper", "norm_arxiv_id", "norm_title",
    "GoldResolution", "resolve_gold",
    "zip_filenames", "retrievable_gold_ids",
]
