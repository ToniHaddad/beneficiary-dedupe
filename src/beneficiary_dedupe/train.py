"""Fit the scoring model on a training set generated with its own seed.

The benchmark uses a different seed, so its households, names and errors
were never seen in training.
"""

from __future__ import annotations

import numpy as np

from .blocking import candidate_pairs
from .features import feature_matrix
from .generate import generate
from .model import DEFAULT_MODEL, Model
from .records import prepare

TRAIN_SEED = 1
TEST_SEED = 2
DEMO_SEED = 3


def labelled_pairs(rows: list[dict]):
    prepared = [prepare(i, r) for i, r in enumerate(rows)]
    pair_set, stats = candidate_pairs(prepared)
    pairs = sorted(pair_set)
    X = feature_matrix(prepared, pairs)
    y = np.array([rows[i]["cluster_id"] == rows[j]["cluster_id"] for i, j in pairs], dtype=float)
    return prepared, pairs, X, y, stats


def train(n: int = 10_000, seed: int = TRAIN_SEED, out=DEFAULT_MODEL) -> Model:
    rows = generate(n, seed)
    _, pairs, X, y, stats = labelled_pairs(rows)
    model = Model.fit(X, y)
    model.choose_bands(X, y)
    model.info = {
        "trained_on": f"synthetic, seed {seed}, {n} records",
        "training_pairs": int(len(pairs)),
        "training_matches": int(y.sum()),
    }
    model.save(out)
    return model
