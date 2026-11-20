"""Logistic regression scorer whose output splits into per-field contributions.

The log-odds of a pair being the same household is

    baseline + sum over features (weight * value)

and grouping the features by field gives one number per field: "+4.1 from
the name, -2.3 from the date of birth". The numbers add up to the score the
reviewer sees, so the explanation is the model, not a story told about it.

Fitted with Newton's method and a small L2 penalty, in numpy, so there is no
heavy dependency and the weights fit in a JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, FEATURES, FIELDS

DEFAULT_MODEL = Path(__file__).with_name("model.json")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Model:
    weights: np.ndarray
    bias: float
    upper: float = 0.97   # at or above: certain match, kept out of the queue
    lower: float = 0.03   # at or below: certain non-match, kept out of the queue
    info: dict = field(default_factory=dict)

    # ---------------------------------------------------------- scoring
    def logit(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logit(X))

    def contributions(self, x: np.ndarray) -> dict[str, float]:
        """Per-field log-odds contributions for one pair; they sum to logit - bias."""
        out = dict.fromkeys(FIELDS, 0.0)
        for (fld, _), w, v in zip(FEATURES, self.weights, x):
            out[fld] += float(w * v)
        return out

    def band(self, p: float) -> str:
        if p >= self.upper:
            return "match"
        if p <= self.lower:
            return "non-match"
        return "review"

    # ---------------------------------------------------------- fitting
    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> "Model":
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        reg = np.eye(d + 1) * l2
        reg[-1, -1] = 0.0  # do not shrink the bias
        for _ in range(iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) + reg @ w
            H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(weights=w[:-1], bias=float(w[-1]))

    def choose_bands(self, X: np.ndarray, y: np.ndarray, max_auto_errors: float = 0.005) -> None:
        """Pick the two thresholds on training pairs.

        upper: lowest score where pairs at or above it are at least 99.5%
        true matches. lower: highest score where pairs at or below it hide
        at most 0.5% of all true matches.
        """
        p = self.predict(X)
        order = np.argsort(-p)
        ps, ys = p[order], y[order]
        cum_true = np.cumsum(ys)
        precision = cum_true / np.arange(1, len(ys) + 1)
        ok = np.where(precision >= 1 - max_auto_errors)[0]
        # the deepest cut in the ranking that still meets the precision bar
        self.upper = float(max(ps[ok[-1]], 0.5)) if len(ok) else 0.99
        total_true = ys.sum()
        asc = np.argsort(p)
        missed = np.cumsum(y[asc]) / max(total_true, 1)
        allowed = np.where(missed <= max_auto_errors)[0]
        self.lower = float(min(p[asc][allowed[-1]], 0.5)) if len(allowed) else 0.0

    # ---------------------------------------------------------- storage
    def to_json(self) -> dict:
        return {
            "features": FEATURE_NAMES,
            "weights": [round(float(w), 6) for w in self.weights],
            "bias": round(self.bias, 6),
            "upper": self.upper,
            "lower": self.lower,
            "info": self.info,
        }

    def save(self, path: str | Path = DEFAULT_MODEL) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> "Model":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["features"] != FEATURE_NAMES:
            raise ValueError("model.json was trained on a different feature list; run `dedupe train`")
        return cls(np.array(data["weights"]), data["bias"], data["upper"], data["lower"], data.get("info", {}))
