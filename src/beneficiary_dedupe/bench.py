"""Benchmark on a held-out synthetic set the model never saw.

Prints pair-level precision, recall and F1 for the pipeline and two
baselines, recall on Arabic-to-Latin pairs, blocking recall, reviewer
workload, errors on hard negatives, the effect of the chaining guard, and
the time to process the whole set.
"""

from __future__ import annotations

import itertools
import platform
import time
from collections import defaultdict

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from . import names
from .blocking import block_keys
from .cluster import cluster
from .features import compare
from .generate import generate
from .model import Model
from .normalize import has_arabic, parse_dob, phonetic, soundex
from .pipeline import run
from .train import TEST_SEED, TRAIN_SEED


def true_pairs(rows: list[dict]) -> set[tuple[int, int]]:
    by: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["cluster_id"]].append(i)
    return {p for m in by.values() for p in itertools.combinations(sorted(m), 2)}


def prf(pred: set, truth: set) -> tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def cross_script(rows: list[dict], pairs: set) -> set:
    def ar(i: int) -> bool:
        return has_arabic(rows[i]["given_name"] + rows[i]["family_name"])
    return {(i, j) for i, j in pairs if ar(i) != ar(j)}


def full_name(r: dict) -> str:
    return " ".join(" ".join((r["given_name"], r["father_name"], r["family_name"])).lower().split())


def exact_baseline(rows: list[dict]) -> set:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(full_name(r), parse_dob(r["date_of_birth"]).iso())].append(i)
    return {p for m in groups.values() for p in itertools.combinations(m, 2)}


def levenshtein_scores(rows: list[dict], cutoff: float = 0.6, chunk: int = 1000) -> dict[tuple[int, int], float]:
    """Normalised Levenshtein similarity on the full name, over ALL pairs (no blocking)."""
    namelist = [full_name(r) for r in rows]
    out: dict[tuple[int, int], float] = {}
    for start in range(0, len(namelist), chunk):
        block = namelist[start:start + chunk]
        m = process.cdist(block, namelist, scorer=Levenshtein.normalized_similarity,
                          score_cutoff=cutoff, dtype=np.float32, workers=-1)
        ii, jj = np.nonzero(m)
        for a, b in zip(ii, jj):
            i, j = start + int(a), int(b)
            if i < j:
                out[(i, j)] = float(m[a, b])
    return out


def best_threshold(scores: dict, truth: set) -> float:
    best, best_t = -1.0, 0.0
    for t in np.arange(0.60, 0.99, 0.01):
        pred = {p for p, s in scores.items() if s >= t}
        f = prf(pred, truth)[2]
        if f > best:
            best, best_t = f, float(t)
    return round(best_t, 2)


def pairs_to_reach(ranked: list[tuple[int, int]], truth: set, target: float, skip: int = 0) -> int | None:
    """How many pairs, after the first ``skip`` (accepted without review), until recall >= target."""
    need = target * len(truth)
    found = 0
    for n, pair in enumerate(ranked, 1):
        found += pair in truth
        if found >= need:
            return max(0, n - skip)
    return None


def key_agreement() -> dict[str, float]:
    """Share of spelling-variant pairs from the name tables that get the same key."""
    latin_pairs, cross_pairs = [], []
    for table in (names.MALE_GIVEN, names.FEMALE_GIVEN, names.FAMILY):
        for ar, latin, _ in table:
            latin_pairs += list(itertools.combinations(latin, 2))
            cross_pairs += [(ar, x) for x in latin]
    return {
        "latin_pairs": len(latin_pairs),
        "soundex_latin": np.mean([soundex(a) == soundex(b) for a, b in latin_pairs]),
        "ours_latin": np.mean([phonetic(a) == phonetic(b) for a, b in latin_pairs]),
        "cross_pairs": len(cross_pairs),
        "ours_cross": np.mean([phonetic(a) == phonetic(b) for a, b in cross_pairs]),
    }


def describe(r: dict) -> str:
    parts = [r["given_name"], r["father_name"] or "-", r["family_name"], r["date_of_birth"], r["phone"] or "-",
             r["village"], r["household_size"]]
    return " | ".join(parts)


def main(n: int = 10_000) -> None:
    line = "-" * 72
    model = Model.load()
    print(f"Held-out set: synthetic, seed {TEST_SEED}, {n} records "
          f"(model trained on seed {TRAIN_SEED}, different households)")
    rows = generate(n, TEST_SEED)
    truth = true_pairs(rows)
    truth_x = cross_script(rows, truth)
    print(f"True duplicate pairs: {len(truth)}  (Arabic-to-Latin: {len(truth_x)})")
    print(f"Machine: {platform.system()} {platform.machine()}, Python {platform.python_version()}, one process")
    print(line)

    t0 = time.perf_counter()
    res = run(rows, model)
    elapsed = time.perf_counter() - t0
    print(f"Time to process {n} records: {elapsed:.1f} s")
    for k, v in res.timings.items():
        print(f"  {k:<18} {v:6.2f} s")
    print(line)

    cand = set(res.pairs)
    all_pairs = n * (n - 1) // 2
    print(f"Blocking: {len(cand):,} candidate pairs instead of {all_pairs:,} "
          f"({100 * len(cand) / all_pairs:.2f}%)")
    print(f"  blocking recall (true pairs that reach the scorer): {len(cand & truth) / len(truth):.4f}"
          f"  missed: {len(truth - cand)}")
    passes: dict[str, set] = defaultdict(set)
    keys = [block_keys(p) for p in res.prepared]
    for i, j in truth:
        for k in keys[i] & keys[j]:
            passes[k.split(":")[0]].add((i, j))
    for name in ("gf", "ny", "ph", "fv"):
        print(f"  pass {name}: alone finds {len(passes[name]) / len(truth):.3f} of true pairs")
    print(line)

    probs = res.probs
    scored = dict(zip(res.pairs, (float(p) for p in probs)))
    pred_pair = {p for p, s in scored.items() if s >= 0.5}
    labels = res.clusters.labels
    groups = res.clusters.groups()
    pred_cluster = {p for m in groups.values() for p in itertools.combinations(sorted(m), 2)}

    exact = exact_baseline(rows)
    print("Levenshtein baseline: tuning its threshold on the training set ...")
    train_rows = generate(n, TRAIN_SEED)
    lev_t = best_threshold(levenshtein_scores(train_rows), true_pairs(train_rows))
    lev_scores = levenshtein_scores(rows)
    lev = {p for p, s in lev_scores.items() if s >= lev_t}

    print(f"\n{'method':<44}{'precision':>10}{'recall':>8}{'F1':>7}{'AR-LA recall':>14}")
    table = [
        ("Exact name + date of birth", exact),
        (f"Levenshtein on full name >= {lev_t:.2f}", lev),
        ("This pipeline, pairs scored >= 0.5", pred_pair),
        ("This pipeline, after clustering", pred_cluster),
    ]
    for label, pred in table:
        p, r, f = prf(pred, truth)
        rx = len(pred & truth_x) / len(truth_x) if truth_x else 0
        print(f"{label:<44}{p:>10.3f}{r:>8.3f}{f:>7.3f}{rx:>14.3f}")
    print(line)

    bands = res.band_counts()
    band_of = {p: model.band(float(s), x) for p, s, x in zip(res.pairs, res.probs, res.X)}
    auto = {p for p, b in band_of.items() if b == "match"}
    review = {p for p, b in band_of.items() if b == "review"}
    print(f"Review bands (thresholds {model.lower:.3f} / {model.upper:.3f}, chosen on training pairs):")
    print(f"  certain match, not shown: {bands['match']:,}   to review: {bands['review']:,}   "
          f"certain non-match, not shown: {bands['non-match']:,}")
    p_auto, r_auto, _ = prf(auto, truth)
    p_final, r_final, f_final = prf(auto | (review & truth), truth)
    print(f"  certain-match band alone: precision {p_auto:.3f}, recall {r_auto:.3f}")
    print(f"  after a reviewer decides the whole queue correctly: precision {p_final:.3f}, "
          f"recall {r_final:.3f}, F1 {f_final:.3f}")
    hidden = len({p for p, b in band_of.items() if b == "non-match"} & truth)
    print(f"  true pairs hidden in the certain non-match band: {hidden}")
    print(line)

    # the reviewer sees the auto-accepted pairs as done, then works the queue
    # from the most likely match down
    ranked_model = sorted(auto) + [p for p, _ in sorted(((p, scored[p]) for p in review), key=lambda kv: -kv[1])]
    ranked_lev = [p for p, _ in sorted(lev_scores.items(), key=lambda kv: -kv[1])]
    n_auto = len(auto)
    print("Reviewer workload: pairs a person must look at to reach a pair recall")
    print(f"  {'recall':<8}{'model queue':>14}{'Levenshtein order':>20}{'unranked candidates':>22}")
    for target in (0.80, 0.90, 0.95, 0.98):
        a = pairs_to_reach(ranked_model, truth, target, skip=n_auto)
        b = pairs_to_reach(ranked_lev, truth, target)
        c = int(round(target * len(cand))) if len(cand & truth) >= target * len(truth) else None
        fmt = lambda v: f"{v:,}" if v is not None else "not reached"
        print(f"  {target:<8.2f}{fmt(a):>14}{fmt(b):>20}{fmt(c):>22}")
    print(f"  (model queue: the {n_auto:,} certain matches are accepted without review)")
    print(line)

    print("Hard negatives (different households built to look alike):")
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_cluster[r["cluster_id"]].append(i)
    kinds: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for cid, members in by_cluster.items():
        rel = rows[members[0]]["related_to"]
        if not rel:
            continue
        kind = next(r["gen_note"] for r in (rows[m] for m in members) if r["gen_note"].startswith("neg:"))
        for i in members:
            for j in by_cluster.get(rel, []):
                kinds[kind][0] += 1
                kinds[kind][1] += labels[i] == labels[j]
    for kind, (total, merged) in sorted(kinds.items()):
        print(f"  {kind:<28} {merged:>4} of {total:>4} pairs wrongly merged")
    print(line)

    def score_fn(i: int, j: int) -> float:
        return float(model.predict(compare(res.prepared[i], res.prepared[j])[None, :])[0])

    plain = cluster(n, dict(scored), score_fn, guard=0.0, max_size=10_000)
    plain_pairs = {p for m in plain.groups().values() for p in itertools.combinations(sorted(m), 2)}
    pp, rp, fp = prf(plain_pairs, truth)
    pg, rg, fg = prf(pred_cluster, truth)
    print("Chaining guard:")
    print(f"  without guard: precision {pp:.3f}, recall {rp:.3f}, F1 {fp:.3f}, "
          f"largest cluster {max(len(m) for m in plain.groups().values())}")
    print(f"  with guard:    precision {pg:.3f}, recall {rg:.3f}, F1 {fg:.3f}, "
          f"largest cluster {max(len(m) for m in groups.values())}, merges refused {len(res.clusters.refused)}")
    print(line)

    ka = key_agreement()
    print("Phonetic key on spelling variants from the name tables (same name, two spellings):")
    print(f"  Latin-Latin pairs ({ka['latin_pairs']}): Soundex agrees {ka['soundex_latin']:.3f}, "
          f"this key agrees {ka['ours_latin']:.3f}")
    print(f"  Arabic-Latin pairs ({ka['cross_pairs']}): this key agrees {ka['ours_cross']:.3f} "
          f"(Soundex cannot read Arabic)")
    print(line)

    fps = sorted((scored.get(p, 0.0), p) for p in pred_cluster - truth)[::-1][:4]
    fns = sorted((scored.get(p, 0.0), p) for p in truth - pred_cluster)[::-1][:4]
    print("Wrongly merged (highest score first):")
    for s, (i, j) in fps:
        print(f"  p={s:.3f}  {describe(rows[i])}\n           {describe(rows[j])}"
              f"\n           truth: {rows[i]['gen_note']} / {rows[j]['gen_note']}")
    print("Missed duplicates (highest score first):")
    for s, (i, j) in fns:
        print(f"  p={s:.3f}  {describe(rows[i])}\n           {describe(rows[j])}"
              f"\n           truth: {rows[i]['gen_note']} / {rows[j]['gen_note']}")
