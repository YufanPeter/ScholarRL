"""Coverage report: how much of the gold answer set is end-to-end retrievable.

Run:  python -m scripts.coverage        (from the repo root)

Reports, per split and overall:
- query / answer statistics
- bridge 1: gold ids found in id2paper.json
- bridge 2 (end-to-end): normalized title present as a file in cs_paper_2nd.zip
- per-query answer availability: full / partial / none  (for later query filtering)

This is the one place that opens the zip (namelist only, no extraction).
"""
from __future__ import annotations

import zipfile
from collections import Counter

from scholarrl.data import load_queries, load_id2paper, resolve_gold
from scholarrl.paths import CORPUS_ZIP


def _zip_filenames() -> set:
    print("reading zip namelist (no extraction)...")
    with zipfile.ZipFile(CORPUS_ZIP) as zf:
        return set(zf.namelist())


def main() -> None:
    zip_names = _zip_filenames()
    print(f"zip files: {len(zip_names)}")
    id2paper = load_id2paper()
    print(f"id2paper entries: {len(id2paper)}\n")

    all_ids = set()
    for split in ("train", "dev", "test"):
        recs = load_queries(split)
        n_ans = sum(len(r.answer_ids) for r in recs)
        uniq = {a for r in recs for a in r.answer_ids}
        all_ids |= uniq
        avg = n_ans / len(recs) if recs else 0
        print(f"== {split} ==")
        print(f"  queries        : {len(recs)}")
        print(f"  total answers  : {n_ans}  (avg {avg:.2f}/query)")
        print(f"  unique answers : {len(uniq)}")

    # resolve every unique gold id along the full chain
    in_id2paper = 0
    end_to_end = set()          # gold ids that resolve into a real zip file
    for a in all_ids:
        res = resolve_gold(a)
        if res.in_id2paper:
            in_id2paper += 1
        if res.filename and res.filename in zip_names:
            end_to_end.add(a)

    n = len(all_ids)
    print(f"\n== coverage (union of {n} unique gold ids) ==")
    print(f"  bridge 1  in id2paper        : {in_id2paper}  ({100*in_id2paper/n:.1f}%)")
    print(f"  bridge 2  end-to-end in zip  : {len(end_to_end)}  ({100*len(end_to_end)/n:.1f}%)")

    # per-query availability (uses the end-to-end set)
    dist = Counter()
    for split in ("train", "dev", "test"):
        for r in load_queries(split):
            if not r.answer_ids:
                dist[(split, "empty")] += 1
                continue
            hit = sum(1 for a in r.answer_ids if a in end_to_end)
            if hit == len(r.answer_ids):
                dist[(split, "full")] += 1
            elif hit == 0:
                dist[(split, "none")] += 1
            else:
                dist[(split, "partial")] += 1

    print("\n== per-query answer availability ==")
    for split in ("train", "dev", "test"):
        full = dist[(split, "full")]
        part = dist[(split, "partial")]
        none = dist[(split, "none")]
        total = full + part + none + dist[(split, "empty")]
        print(f"  {split}: full={full}  partial={part}  none={none}  (of {total})")


if __name__ == "__main__":
    main()
