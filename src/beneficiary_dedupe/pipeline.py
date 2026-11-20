"""End-to-end run: prepare, block, compare, score, cluster."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .blocking import BlockingStats, candidate_pairs
from .cluster import ClusterResult, cluster
from .features import FIELDS, compare, feature_matrix
from .model import Model
from .records import Prepared, prepare


@dataclass
class RunResult:
    rows: list[dict]
    prepared: list[Prepared]
    pairs: list[tuple[int, int]]
    X: np.ndarray
    probs: np.ndarray
    model: Model
    clusters: ClusterResult
    blocking: BlockingStats
    timings: dict[str, float] = field(default_factory=dict)

    def band_counts(self) -> dict[str, int]:
        counts = {"match": 0, "review": 0, "non-match": 0}
        for p, x in zip(self.probs, self.X):
            counts[self.model.band(float(p), x)] += 1
        return counts

    def explain(self, k: int) -> dict:
        """Everything a reviewer needs about candidate pair ``k``."""
        p = float(self.probs[k])
        return {
            "probability": p,
            "band": self.model.band(p, self.X[k]),
            "baseline": self.model.baseline,
            "contributions": self.model.contributions(self.X[k]),
        }


def run(rows: list[dict], model: Model | None = None, guard: float = 0.02) -> RunResult:
    model = model or Model.load()
    t = {}
    start = time.perf_counter()
    prepared = [prepare(i, row) for i, row in enumerate(rows)]
    t["prepare"] = time.perf_counter() - start

    s = time.perf_counter()
    pair_set, stats = candidate_pairs(prepared)
    pairs = sorted(pair_set)
    t["blocking"] = time.perf_counter() - s

    s = time.perf_counter()
    X = feature_matrix(prepared, pairs)
    probs = model.predict(X) if len(pairs) else np.zeros(0)
    t["compare_and_score"] = time.perf_counter() - s

    s = time.perf_counter()
    scored = {pair: float(p) for pair, p in zip(pairs, probs)}

    def score_fn(i: int, j: int) -> float:
        return float(model.predict(compare(prepared[i], prepared[j])[None, :])[0])

    clusters = cluster(len(prepared), scored, score_fn, threshold=0.5, guard=guard)
    t["cluster"] = time.perf_counter() - s
    t["total"] = time.perf_counter() - start
    return RunResult(rows, prepared, pairs, X, probs, model, clusters, stats, t)


def matches_table(result: RunResult) -> list[dict]:
    """Rows for matches.csv: every pair not ruled out, with its reasons."""
    out = []
    labels = result.clusters.labels
    for k, (i, j) in enumerate(result.pairs):
        p = float(result.probs[k])
        band = result.model.band(p, result.X[k])
        if band == "non-match":
            continue
        contrib = result.model.contributions(result.X[k])
        row = {
            "record_a": result.rows[i]["record_id"],
            "record_b": result.rows[j]["record_id"],
            "probability": f"{p:.4f}",
            "band": band,
            "same_cluster": "yes" if labels[i] == labels[j] else "no",
        }
        row.update({f"why_{f}": f"{contrib[f]:+.2f}" for f in FIELDS})
        out.append(row)
    out.sort(key=lambda r: -float(r["probability"]))
    return out


def clusters_table(result: RunResult) -> list[dict]:
    groups = result.clusters.groups()
    out = []
    for n, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
        for i in members:
            out.append({"record_id": result.rows[i]["record_id"], "household": f"G{n + 1:06d}", "size": len(members)})
    return out
