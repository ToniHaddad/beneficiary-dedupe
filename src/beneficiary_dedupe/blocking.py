"""Blocking: decide which pairs of records are worth comparing at all.

Ten thousand records make about fifty million pairs. Blocking puts records
into small groups that share a cheap key and only compares records inside
a group. A true pair is only found if the two records share at least one
key, so several independent passes are used and their union is kept:

* ``gf``: phonetic code of given name and family name, in either order
  (catches name-order swaps and every transliteration of both names).
* ``ny``: phonetic code of the given OR family name plus birth year
  (catches a typing error in the other name, and a dropped father's name).
* ``ph``: the normalised phone number (catches badly mangled names).
* ``fv``: phonetic code of the family name plus village skeleton
  (catches a wrong birth year together with a changed phone).

Blocks larger than ``max_block`` are skipped and counted, never silently:
the benchmark reports blocking recall so a key that drops true pairs shows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .records import Prepared


def block_keys(r: Prepared) -> set[str]:
    keys: set[str] = set()
    g, f = r.given_ph, r.family_ph
    if g and f:
        keys.add("gf:" + "|".join(sorted((g, f))))
    if r.dob.year:
        for part in (g, f):
            if part:
                keys.add(f"ny:{part}:{r.dob.year}")
    if r.phone:
        keys.add("ph:" + r.phone)
    if f and r.village_sk:
        keys.add(f"fv:{f}:{r.village_sk}")
    return keys


@dataclass
class BlockingStats:
    records: int
    blocks: int
    skipped_blocks: int
    skipped_records: int
    candidate_pairs: int

    @property
    def all_pairs(self) -> int:
        return self.records * (self.records - 1) // 2


def candidate_pairs(records: list[Prepared], max_block: int = 400) -> tuple[set[tuple[int, int]], BlockingStats]:
    """Return the set of (i, j) index pairs with i < j that share a block key."""
    blocks: dict[str, list[int]] = defaultdict(list)
    for r in records:
        for key in block_keys(r):
            blocks[key].append(r.idx)
    pairs: set[tuple[int, int]] = set()
    skipped = skipped_records = 0
    for members in blocks.values():
        if len(members) < 2:
            continue
        if len(members) > max_block:
            skipped += 1
            skipped_records += len(members)
            continue
        members = sorted(members)
        for a in range(len(members)):
            ia = members[a]
            for b in range(a + 1, len(members)):
                pairs.add((ia, members[b]))
    stats = BlockingStats(len(records), len(blocks), skipped, skipped_records, len(pairs))
    return pairs, stats
