"""Normalization functions — the two bridges of the data linkage chain.

    answer_arxiv_id --norm_arxiv_id--> id2paper key --(title)--> norm_title --> zip filename

Both rules were reverse-engineered and verified against the real data
(92% of gold answers resolve end-to-end). Keep these pure and side-effect free
so they stay unit-testable and cheap.
"""
import re
import unicodedata

_VERSION_RE = re.compile(r"v\d+$")
_NON_ALPHA_RE = re.compile(r"[^a-z]")

# Math alphanumeric / styled Unicode letters (e.g. "BadNL") lose their letters under a
# plain ASCII drop, which breaks the title->filename match. Fold them back to ASCII first.
def _fold_styled_letters(text: str) -> str:
    out = []
    for ch in text:
        # NFKC maps most mathematical/bold/italic letter variants to plain ASCII letters
        folded = unicodedata.normalize("NFKC", ch)
        out.append(folded)
    return "".join(out)


def norm_arxiv_id(arxiv_id: str) -> str:
    """Normalize an arxiv id to match id2paper.json keys.

    - keep only the id core (drop any category prefix like 'cs/')
    - strip a trailing version suffix ('2006.01043v2' -> '2006.01043')
    """
    if not arxiv_id:
        return ""
    core = arxiv_id.strip().split("/")[-1]
    core = _VERSION_RE.sub("", core)
    return core


def norm_title(title: str) -> str:
    """Normalize a paper title to the cs_paper_2nd.zip filename convention.

    Rule (verified): fold styled letters -> strip accents -> lowercase -> keep only a-z.
    Example: 'A Critique of Chen\\'s "The 2-MAXSAT..."' -> 'acritiqueofchensthemaxsat...'
    Note: digits are dropped too ('over 100 Years' -> 'overyears').
    """
    if not title:
        return ""
    t = _fold_styled_letters(title)
    # strip accents: decompose then drop combining marks / non-ascii
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode()
    return _NON_ALPHA_RE.sub("", t.lower())
