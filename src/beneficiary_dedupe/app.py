"""Review screen: one page, served by FastAPI on port 8108.

The queue holds only the pairs the model is unsure about, most uncertain
first. Certain matches and certain non-matches are counted, not shown.
Decisions are saved to a JSON file after every click, so a power cut loses
nothing, and can be exported as CSV.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .features import FIELD_LABELS, FIELDS, name_alignment, part_similarity
from .generate import generate
from .model import GIVEN_DIFFERS
from .normalize import compare_dob, has_arabic
from .pipeline import RunResult, run
from .records import FIELDS as RECORD_FIELDS
from .records import Prepared, read_csv
from .train import DEMO_SEED

DECISIONS = ("same", "different", "unsure")
DEMO_SIZE = 5_000


def _status_name(a: Prepared, b: Prepared, part_a: str, part_b: str, sim: float) -> dict:
    ta, tb = a.raw.get(part_a + "_name", ""), b.raw.get(part_b + "_name", "")
    if not ta or not tb:
        return {"status": "missing", "note": "not recorded on one side"}
    if getattr(a, part_a) == getattr(b, part_b):
        return {"status": "same", "note": ""}
    if sim >= 0.85:
        note = "same name, other script" if has_arabic(ta) != has_arabic(tb) else "spelled differently"
        return {"status": "variant", "note": f"{note} ({sim:.2f})"}
    return {"status": "differs", "note": f"similarity {sim:.2f}"}


def compare_fields(a: Prepared, b: Prepared) -> dict:
    """Per displayed field: same / variant / differs / missing, with a short note."""

    g, f, swapped = name_alignment(a, b)
    out = {}
    if swapped:
        out["given_name"] = _status_name(a, b, "given", "family", g)
        out["family_name"] = _status_name(a, b, "family", "given", f)
        out["given_name"]["note"] = "matches the other record's family name (order swapped)"
        out["family_name"]["note"] = "matches the other record's given name (order swapped)"
    else:
        out["given_name"] = _status_name(a, b, "given", "given", g)
        out["family_name"] = _status_name(a, b, "family", "family", f)
    if a.father_snd and b.father_snd:
        out["father_name"] = _status_name(a, b, "father", "father", part_similarity(a, b, "father", "father"))
    else:
        out["father_name"] = {"status": "missing", "note": "not recorded on one side"}

    level = compare_dob(a.dob, b.dob)
    out["date_of_birth"] = {
        "exact": {"status": "same", "note": ""},
        "swapped": {"status": "variant", "note": "day and month swapped"},
        "year_only": {"status": "variant", "note": "one side has the year only"},
        "same_year": {"status": "differs", "note": "same year, different day"},
        "near_year": {"status": "differs", "note": "one year apart"},
        "differs": {"status": "differs", "note": "different"},
        "missing": {"status": "missing", "note": "not recorded on one side"},
    }[level]
    if a.dob.iso() == b.dob.iso() and a.raw["date_of_birth"] != b.raw["date_of_birth"] and level == "exact":
        out["date_of_birth"] = {"status": "variant", "note": "same date, written differently"}

    if not a.phone or not b.phone:
        out["phone"] = {"status": "missing", "note": "not recorded on one side"}
    elif a.phone == b.phone:
        same_text = a.raw["phone"] == b.raw["phone"]
        out["phone"] = {"status": "same" if same_text else "variant",
                        "note": "" if same_text else f"same number: {a.phone}"}
    else:
        out["phone"] = {"status": "differs", "note": "different number"}

    if a.village_sk and a.village_sk == b.village_sk:
        same_text = a.raw["village"] == b.raw["village"]
        out["village"] = {"status": "same" if same_text else "variant", "note": "" if same_text else "same place, spelled differently"}
    elif a.district != b.district and a.at_shelter != b.at_shelter:
        out["village"] = {"status": "differs", "note": "one record taken at a shelter in another district"}
    else:
        out["village"] = {"status": "differs", "note": "different village"}
    out["district"] = {"status": "same" if a.district == b.district else "differs", "note": ""}
    out["centre"] = {"status": "info", "note": ""}

    if a.household is None or b.household is None:
        out["household_size"] = {"status": "missing", "note": ""}
    else:
        d = abs(a.household - b.household)
        out["household_size"] = {"status": "same" if d == 0 else ("variant" if d == 1 else "differs"),
                                 "note": "" if d == 0 else f"off by {d}"}
    out["team"] = {"status": "info", "note": ""}
    return out


class Decision(BaseModel):
    k: int
    decision: str | None


class ReviewSession:
    def __init__(self, records_path: str | None, decisions_path: str | Path, demo_size: int = DEMO_SIZE):
        if records_path:
            rows = read_csv(records_path)
            self.source = f"{Path(records_path).name}, {len(rows):,} records"
        else:
            rows = generate(demo_size, DEMO_SEED)
            self.source = f"synthetic demo list, {len(rows):,} records (seed {DEMO_SEED})"
        self.result: RunResult = run(rows)
        res = self.result
        bands = [res.model.band(float(p), x) for p, x in zip(res.probs, res.X)]
        self.auto_matches = [k for k, b in enumerate(bands) if b == "match"]
        self.auto_non = sum(1 for b in bands if b == "non-match")
        review = [k for k, b in enumerate(bands) if b == "review"]
        # most uncertain first: closest to even odds
        self.queue = sorted(review, key=lambda k: abs(float(res.probs[k]) - 0.5))
        self.path = Path(decisions_path)
        self.lock = threading.Lock()
        self.decisions: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.decisions = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.decisions = {}

    def pair_key(self, k: int) -> str:
        i, j = self.result.pairs[self.queue[k]]
        return f"{self.result.rows[i]['record_id']}|{self.result.rows[j]['record_id']}"

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.decisions, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def summary(self) -> dict:
        counts = dict.fromkeys(DECISIONS, 0)
        for k in range(len(self.queue)):
            d = self.decisions.get(self.pair_key(k))
            if d:
                counts[d["decision"]] += 1
        res = self.result
        return {
            "source": self.source,
            "records": len(res.rows),
            "possible_pairs": len(res.rows) * (len(res.rows) - 1) // 2,
            "candidate_pairs": len(res.pairs),
            "auto_match": len(self.auto_matches),
            "auto_non_match": self.auto_non,
            "queue": len(self.queue),
            "decided": counts,
            "lower": res.model.lower,
            "upper": res.model.upper,
        }

    def queue_items(self) -> list[dict]:
        res = self.result
        out = []
        for k, idx in enumerate(self.queue):
            i, j = res.pairs[idx]
            d = self.decisions.get(self.pair_key(k))
            ra, rb = res.rows[i], res.rows[j]
            out.append({
                "k": k,
                "p": round(float(res.probs[idx]), 3),
                "a": f"{ra['given_name']} {ra['family_name']}",
                "b": f"{rb['given_name']} {rb['family_name']}",
                "decision": d["decision"] if d else None,
            })
        return out

    def pair(self, k: int) -> dict:
        if not 0 <= k < len(self.queue):
            raise HTTPException(404, "no such pair")
        res = self.result
        idx = self.queue[k]
        i, j = res.pairs[idx]
        a, b = res.prepared[i], res.prepared[j]
        info = res.explain(idx)
        d = self.decisions.get(self.pair_key(k))
        return {
            "k": k,
            "probability": info["probability"],
            "baseline": info["baseline"],
            "contributions": [
                {"field": f, "label": FIELD_LABELS[f], "value": round(info["contributions"][f], 2)} for f in FIELDS
            ],
            "given_names_differ": bool(res.X[idx][GIVEN_DIFFERS] > 0),
            "fields": [f for f in RECORD_FIELDS if f != "record_id"],
            "a": {f: a.raw.get(f, "") for f in RECORD_FIELDS},
            "b": {f: b.raw.get(f, "") for f in RECORD_FIELDS},
            "compare": compare_fields(a, b),
            "decision": d["decision"] if d else None,
        }

    def decide(self, k: int, decision: str | None) -> dict:
        if not 0 <= k < len(self.queue):
            raise HTTPException(404, "no such pair")
        if decision is not None and decision not in DECISIONS:
            raise HTTPException(422, f"decision must be one of {DECISIONS} or null")
        key = self.pair_key(k)
        with self.lock:
            if decision is None:
                self.decisions.pop(key, None)
            else:
                self.decisions[key] = {
                    "decision": decision,
                    "probability": round(float(self.result.probs[self.queue[k]]), 4),
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            self.save()
        return self.summary()

    def export_csv(self) -> str:
        res = self.result
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["record_a", "record_b", "probability", "decision", "decided_by", "decided_at"])
        for k in range(len(self.queue)):
            d = self.decisions.get(self.pair_key(k))
            if d:
                a, b = self.pair_key(k).split("|")
                w.writerow([a, b, d["probability"], d["decision"], "reviewer", d["at"]])
        for idx in self.auto_matches:
            i, j = res.pairs[idx]
            w.writerow([res.rows[i]["record_id"], res.rows[j]["record_id"],
                        round(float(res.probs[idx]), 4), "same", "automatic (certain match)", ""])
        return buf.getvalue()


def create_app(records_path: str | None = None, decisions_path: str | Path = "decisions.json",
               demo_size: int = DEMO_SIZE) -> FastAPI:
    session = ReviewSession(records_path, decisions_path, demo_size)
    app = FastAPI(title="Household duplicate review")
    page = resources.files("beneficiary_dedupe").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/summary")
    def summary() -> dict:
        return session.summary()

    @app.get("/api/queue")
    def queue() -> list[dict]:
        return session.queue_items()

    @app.get("/api/pair/{k}")
    def pair(k: int) -> dict:
        return session.pair(k)

    @app.post("/api/decide")
    def decide(body: Decision) -> dict:
        return session.decide(body.k, body.decision)

    @app.get("/api/export")
    def export() -> Response:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        return Response(session.export_csv(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="dedupe-decisions-{stamp}.csv"'})

    app.state.session = session
    return app
