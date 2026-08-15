"""Resolve a gold arxiv id along the linkage chain — up to the zip filename (no zip IO here).

    answer_arxiv_id --norm_arxiv_id--> id2paper key --> title --norm_title--> zip filename

Opening the 2.3GB zip and reading paper content is the corpus module's job. This module
stays pure string logic: cheap, fast, easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .id2paper import load_id2paper
from .normalize import norm_arxiv_id, norm_title


@dataclass
class GoldResolution:
    arxiv_id: str
    norm_id: str
    title: Optional[str]        # from id2paper, None if not found
    filename: Optional[str]     # normalized title -> zip filename, None if unresolved

    @property
    def in_id2paper(self) -> bool:
        return self.title is not None

    @property
    def resolved(self) -> bool:
        """True if we have a candidate zip filename (still to be confirmed against the zip)."""
        return bool(self.filename)


def resolve_gold(arxiv_id: str) -> GoldResolution:
    """Map one gold id to its id2paper title and candidate zip filename."""
    id2paper = load_id2paper()
    nid = norm_arxiv_id(arxiv_id)
    # try normalized id first, then the raw id as a fallback
    title = id2paper.get(nid) or id2paper.get(arxiv_id)
    filename = norm_title(title) if title else None
    return GoldResolution(arxiv_id=arxiv_id, norm_id=nid, title=title, filename=filename or None)
