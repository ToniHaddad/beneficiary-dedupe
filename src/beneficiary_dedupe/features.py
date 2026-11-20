"""Pairwise comparison features, grouped by the field a reviewer would read.

Every feature belongs to one field (name, father's name, date of birth,
phone, location, household size). The model's weight times the feature
value, summed per field, is the "reason" shown to the reviewer.
"""

from __future__ import annotations

import numpy as np

from .normalize import compare_dob
from .records import Prepared
from .similarity import name_similarity

FEATURES: list[tuple[str, str]] = [
    ("name", "given_sim"),
    ("name", "family_sim"),
    ("name", "both_names_high"),
    ("name", "order_swapped"),
    ("father", "father_sim"),
    ("father", "father_missing"),
    ("dob", "dob_exact"),
    ("dob", "dob_swapped"),
    ("dob", "dob_year_only"),
    ("dob", "dob_same_year"),
    ("dob", "dob_near_year"),
    ("dob", "dob_missing"),
    ("phone", "phone_same"),
    ("phone", "phone_missing"),
    ("location", "village_same"),
    ("location", "district_same"),
    ("household", "household_same"),
    ("household", "household_off_by_one"),
]
FEATURE_NAMES = [f for _, f in FEATURES]
FIELDS = ["name", "father", "dob", "phone", "location", "household"]
FIELD_LABELS = {
    "name": "Name", "father": "Father's name", "dob": "Date of birth",
    "phone": "Phone", "location": "Village / district", "household": "Household size",
}


def _sim(a: Prepared, b: Prepared, part_a: str, part_b: str) -> float:
    return name_similarity(
        getattr(a, part_a), getattr(a, part_a + "_snd"), a.arabic,
        getattr(b, part_b), getattr(b, part_b + "_snd"), b.arabic,
    )


def name_alignment(a: Prepared, b: Prepared) -> tuple[float, float, bool]:
    """Given and family similarity, aligned the better of two ways.

    Returns (given_sim, family_sim, swapped). "swapped" means the family name
    of one record lines up with the given name of the other.
    """
    g, f = _sim(a, b, "given", "given"), _sim(a, b, "family", "family")
    gs, fs = _sim(a, b, "given", "family"), _sim(a, b, "family", "given")
    if gs + fs > g + f + 0.2:
        return gs, fs, True
    return g, f, False


def compare(a: Prepared, b: Prepared) -> np.ndarray:
    g, f, swapped = name_alignment(a, b)
    x = dict.fromkeys(FEATURE_NAMES, 0.0)
    x["given_sim"] = g
    x["family_sim"] = f
    x["both_names_high"] = 1.0 if min(g, f) >= 0.9 else 0.0
    x["order_swapped"] = 1.0 if swapped else 0.0

    if a.father_snd and b.father_snd:
        x["father_sim"] = _sim(a, b, "father", "father")
    else:
        x["father_missing"] = 1.0

    level = compare_dob(a.dob, b.dob)
    if level != "differs":
        x["dob_" + level] = 1.0

    if a.phone and b.phone:
        x["phone_same"] = 1.0 if a.phone == b.phone else 0.0
    else:
        x["phone_missing"] = 1.0

    if a.village_sk and a.village_sk == b.village_sk:
        x["village_same"] = 1.0
    if a.district and a.district == b.district:
        x["district_same"] = 1.0

    if a.household is not None and b.household is not None:
        diff = abs(a.household - b.household)
        x["household_same"] = 1.0 if diff == 0 else 0.0
        x["household_off_by_one"] = 1.0 if diff == 1 else 0.0
    return np.array([x[n] for n in FEATURE_NAMES], dtype=float)


def feature_matrix(records: list[Prepared], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((len(pairs), len(FEATURE_NAMES)))
    for k, (i, j) in enumerate(pairs):
        out[k] = compare(records[i], records[j])
    return out
