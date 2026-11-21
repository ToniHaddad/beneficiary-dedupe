import itertools

import numpy as np
import pytest

from beneficiary_dedupe.blocking import candidate_pairs
from beneficiary_dedupe.cluster import cluster
from beneficiary_dedupe.features import FEATURE_NAMES, compare
from beneficiary_dedupe.generate import generate
from beneficiary_dedupe.model import Model
from beneficiary_dedupe.pipeline import run
from beneficiary_dedupe.records import prepare


@pytest.fixture(scope="module")
def rows():
    return generate(2000, seed=11)


def true_pairs(rows):
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["cluster_id"], []).append(i)
    return {p for m in by.values() for p in itertools.combinations(sorted(m), 2)}


def test_generator_is_seeded_and_labelled(rows):
    assert generate(2000, seed=11) == rows
    assert generate(50, seed=12) != generate(50, seed=11)
    notes = {r["gen_note"].split(":")[0] for r in rows}
    assert {"base", "dup", "neg"} <= notes
    assert any(r["gen_note"].startswith("dup:") and "arabic" in r["gen_note"] for r in rows)


def test_displaced_households_registered_again_at_a_shelter(rows):
    from beneficiary_dedupe import names
    by = {}
    for r in rows:
        by.setdefault(r["cluster_id"], []).append(r)
    pairs = 0
    for members in by.values():
        home = [r for r in members if "displaced" not in r["gen_note"]]
        shelter = [r for r in members if "displaced" in r["gen_note"]]
        for s in shelter:
            assert s["team"] == names.TEAM_SHELTER
            assert s["centre"] == names.CENTRES[s["district"]]
            for h in home:
                assert h["district"] != s["district"]  # another district, another village
                pairs += 1
    assert pairs > 20
    assert all(r["record_id"].startswith("REG-2026-") for r in rows)


def test_heldout_file_is_what_the_generator_makes():
    # The held-out set was frozen before any rule change; if the generator
    # drifts, the benchmark would no longer be measuring the frozen set.
    from pathlib import Path

    from beneficiary_dedupe.records import FIELDS, TRUTH_FIELDS, read_csv
    path = Path(__file__).resolve().parents[1] / "data" / "heldout.csv"
    frozen = read_csv(path)
    fresh = generate(10_000, seed=2)
    keys = FIELDS + TRUTH_FIELDS
    assert [{k: r[k] for k in keys} for r in frozen] == [{k: r[k] for k in keys} for r in fresh]


def test_shelter_record_in_another_district_is_not_held_against_the_match():
    base = {"given_name": "Jihad", "father_name": "Mahmoud", "family_name": "Hamoud",
            "date_of_birth": "22/01/1984", "phone": "03822085", "household_size": "4"}
    home = prepare(0, {**base, "village": "Khiam", "district": "Marjayoun", "centre": "Marjayoun centre",
                       "team": "Centre social department"})
    shelter = prepare(1, {**base, "village": "Saida", "district": "Saida", "centre": "Saida centre",
                          "team": "Mobile team, school shelter"})
    other = prepare(2, {**base, "village": "Saida", "district": "Saida", "centre": "Saida centre",
                        "team": "Distribution point"})
    i = FEATURE_NAMES.index("shelter_elsewhere")
    assert compare(home, shelter)[i] == 1.0
    assert compare(home, other)[i] == 0.0
    model = Model.load()
    p_shelter = model.predict(compare(home, shelter)[None, :])[0]
    p_other = model.predict(compare(home, other)[None, :])[0]
    assert p_shelter > p_other


def test_blocking_recall_is_measured_and_high(rows):
    prepared = [prepare(i, r) for i, r in enumerate(rows)]
    pairs, stats = candidate_pairs(prepared)
    truth = true_pairs(rows)
    recall = len(pairs & truth) / len(truth)
    assert recall >= 0.99, f"blocking drops true pairs: recall {recall:.4f}"
    assert stats.candidate_pairs < stats.all_pairs * 0.02  # far fewer than all pairs
    assert stats.skipped_blocks == 0


def test_blocking_reports_oversized_blocks():
    rows = [{"record_id": str(i), "given_name": "Mohammad", "family_name": "Hajj", "father_name": "",
             "date_of_birth": "1980", "phone": "", "village": "Arsal", "district": "Baalbek",
             "household_size": "4"} for i in range(30)]
    prepared = [prepare(i, r) for i, r in enumerate(rows)]
    pairs, stats = candidate_pairs(prepared, max_block=10)
    assert stats.skipped_blocks > 0 and stats.skipped_records > 0  # counted, not silent
    assert pairs == set()


def test_model_contributions_add_up_to_the_score():
    model = Model.load()
    a = prepare(0, {"given_name": "Mohamed", "father_name": "Ali", "family_name": "El Khouri",
                    "date_of_birth": "07/03/1985", "phone": "+961 3 123 456", "village": "Bar Elias",
                    "district": "Zahle", "household_size": "5"})
    b = prepare(1, {"given_name": "محمد", "father_name": "علي", "family_name": "الخوري",
                    "date_of_birth": "03/07/1985", "phone": "03123456", "village": "برالياس",
                    "district": "Zahle", "household_size": "6"})
    x = compare(a, b)
    logit = float(model.logit(x[None, :])[0])
    assert model.baseline + sum(model.contributions(x).values()) == pytest.approx(logit)
    assert model.predict(x[None, :])[0] > 0.95  # cross-script, swapped date, reformatted phone
    assert x[FEATURE_NAMES.index("dob_swapped")] == 1.0


def test_twins_never_auto_accepted():
    model = Model.load()
    base = {"father_name": "Ahmad", "family_name": "Hamdan", "date_of_birth": "08/05/1981",
            "phone": "71623528", "village": "Qana", "district": "Tyre", "household_size": "5"}
    a = prepare(0, {**base, "given_name": "Mohammad"})
    b = prepare(1, {**base, "given_name": "Amal"})
    x = compare(a, b)
    assert model.band(float(model.predict(x[None, :])[0]), x) != "match"


def test_logistic_fit_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, len(FEATURE_NAMES)))
    y = (X[:, 0] * 3 - X[:, 1] * 2 + rng.normal(size=2000) > 0).astype(float)
    m = Model.fit(X, y)
    assert m.weights[0] > 1.5 and m.weights[1] < -1.0
    assert ((m.predict(X) > 0.5) == y).mean() > 0.85


def test_chaining_guard_keeps_strangers_apart():
    # A~B and B~C look like matches, but A and C are clearly different people.
    scored = {(0, 1): 0.99, (1, 2): 0.98}
    far = {(0, 2): 0.01}
    result = cluster(3, dict(scored), lambda i, j: far[(i, j)], guard=0.2)
    groups = sorted(sorted(m) for m in result.groups().values())
    assert groups == [[0, 1], [2]]
    assert result.refused and result.refused[0][2].startswith("chain")

    # Without the guard the same scores chain all three together.
    plain = cluster(3, dict(scored), lambda i, j: far[(i, j)], guard=0.0)
    assert len(plain.groups()) == 1


def test_held_pair_is_not_merged_on_its_own():
    # 0 and 1 are twins: high score, but given names differ, so the pair is held.
    scored = {(0, 1): 0.95, (0, 2): 0.99}
    result = cluster(3, dict(scored), lambda i, j: 0.01, hold={(0, 1)})
    labels = result.labels
    assert labels[0] == labels[2] and labels[0] != labels[1]
    assert ("held for review" in r[2] for r in result.refused)


def test_cluster_size_cap():
    scored = {(i, i + 1): 0.99 for i in range(9)}
    result = cluster(10, dict(scored), lambda i, j: 0.99, guard=0.0, max_size=4)
    assert max(len(m) for m in result.groups().values()) <= 4


def test_end_to_end_quality_on_small_set(rows):
    res = run(rows)
    truth = true_pairs(rows)
    pred = {p for m in res.clusters.groups().values() for p in itertools.combinations(sorted(m), 2)}
    tp = len(pred & truth)
    assert tp / len(pred) > 0.95
    assert tp / len(truth) > 0.93
