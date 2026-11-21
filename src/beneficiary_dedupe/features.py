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
    ("name", "given_differs"),
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
    ("location", "shelter_elsewhere"),
    ("household", "household_same"),
    ("household", "household_off_by_one"),
]
FEATURE_NAMES = [f for _, f in FEATURES]
FIELDS = ["name", "father", "dob", "phone", "location", "household"]
FIELD_LABELS = {
    "name": "Name", "father": "Father's name", "dob": "Date of birth",
    "phone": "Phone", "location": "Village / district", "household": "Household size",
}


def part_similarity(a: Prepared, b: Prepared, part_a: str, part_b: str) -> float:
    return name_similarity(
        getattr(a, part_a), getattr(a, part_a + "_snd"), a.arabic,
        getattr(b, part_b), getattr(b, part_b + "_snd"), b.arabic,
    )


def name_alignment(a: Prepared, b: Prepared) -> tuple[float, float, bool]:
    """Given and family similarity, aligned the better of two ways.

    Returns (given_sim, family_sim, swapped). "swapped" means the family name
    of one record lines up with the given name of the other.
    """
    g, f = part_similarity(a, b, "given", "given"), part_similarity(a, b, "family", "family")
    gs, fs = part_similarity(a, b, "given", "family"), part_similarity(a, b, "family", "given")
    if gs + fs > g + f + 0.2:
        return gs, fs, True
    return g, f, False


DOB_INDEX = {lvl: FEATURE_NAMES.index("dob_" + lvl) for lvl in
             ("exact", "swapped", "year_only", "same_year", "near_year", "missing")}
IDX = {name: k for k, name in enumerate(FEATURE_NAMES)}


def compare(a: Prepared, b: Prepared) -> np.ndarray:
    g, f, swapped = name_alignment(a, b)
    x = np.zeros(len(FEATURE_NAMES))
    x[IDX["given_sim"]] = g
    x[IDX["family_sim"]] = f
    x[IDX["given_differs"]] = 1.0 if g < 0.75 else 0.0
    x[IDX["both_names_high"]] = 1.0 if min(g, f) >= 0.9 else 0.0
    x[IDX["order_swapped"]] = 1.0 if swapped else 0.0

    if a.father_snd and b.father_snd:
        x[IDX["father_sim"]] = part_similarity(a, b, "father", "father")
    else:
        x[IDX["father_missing"]] = 1.0

    level = compare_dob(a.dob, b.dob)
    if level != "differs":
        x[DOB_INDEX[level]] = 1.0

    if a.phone and b.phone:
        x[IDX["phone_same"]] = 1.0 if a.phone == b.phone else 0.0
    else:
        x[IDX["phone_missing"]] = 1.0

    if a.village_sk and a.village_sk == b.village_sk:
        x[IDX["village_same"]] = 1.0
    if a.district and a.district == b.district:
        x[IDX["district_same"]] = 1.0
    elif a.at_shelter != b.at_shelter:
        # One record taken at a shelter in another district: displaced
        # households are expected to show a different village.
        x[IDX["shelter_elsewhere"]] = 1.0

    if a.household is not None and b.household is not None:
        diff = abs(a.household - b.household)
        x[IDX["household_same"]] = 1.0 if diff == 0 else 0.0
        x[IDX["household_off_by_one"]] = 1.0 if diff == 1 else 0.0
    return x


def feature_matrix(records: list[Prepared], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((len(pairs), len(FEATURE_NAMES)))
    for k, (i, j) in enumerate(pairs):
        out[k] = compare(records[i], records[j])
    return out
