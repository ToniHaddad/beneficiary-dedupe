"""Group matched pairs into households, without chaining strangers together.

Plain connected components chain: if A looks like B and B looks like C, A
and C end up in one cluster even when A and C clearly differ (two brothers
who both look a bit like a badly typed third record). The guard here merges
two groups only if EVERY pair across them scores at least ``guard``; pairs
that blocking never compared are scored on demand. Groups are also capped at
``max_size`` records, since one household registered more than a handful of
times is far more likely to be a clustering error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ClusterResult:
    labels: list[int]                       # cluster label per record index
    refused: list[tuple[int, int, str]] = field(default_factory=list)

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, lab in enumerate(self.labels):
            out.setdefault(lab, []).append(i)
        return out


def cluster(
    n: int,
    scored_pairs: dict[tuple[int, int], float],
    score_fn: Callable[[int, int], float],
    threshold: float = 0.5,
    guard: float = 0.02,
    max_size: int = 6,
    hold: set[tuple[int, int]] | None = None,
) -> ClusterResult:
    """Greedy merging, strongest pairs first, with a complete-link guard.

    ``scored_pairs`` maps (i, j), i < j, to a match probability.
    ``score_fn(i, j)`` scores a pair that blocking did not produce.
    ``hold`` lists pairs that are never used as the link that merges two
    groups (for example two different given names): such a pair only ends
    up together through other strong links, or through a reviewer.
    """
    hold = hold or set()
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    label = list(range(n))
    refused: list[tuple[int, int, str]] = []

    def score(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        if key not in scored_pairs:
            scored_pairs[key] = score_fn(*key)
        return scored_pairs[key]

    edges = sorted(((p, i, j) for (i, j), p in scored_pairs.items() if p >= threshold), reverse=True)
    for p, i, j in edges:
        a, b = label[i], label[j]
        if a == b:
            continue
        if len(members[a]) + len(members[b]) > max_size:
            refused.append((i, j, "size"))
            continue
        if (i, j) in hold:
            refused.append((i, j, "held for review"))
            continue
        weakest = min(score(x, y) for x in members[a] for y in members[b])
        if weakest < guard:
            refused.append((i, j, f"chain:{weakest:.2f}"))
            continue
        if len(members[a]) < len(members[b]):
            a, b = b, a
        for x in members[b]:
            label[x] = a
        members[a].extend(members.pop(b))
    return ClusterResult(labels=label, refused=refused)
