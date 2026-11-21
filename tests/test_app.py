import csv
import io

import pytest
from fastapi.testclient import TestClient

from beneficiary_dedupe.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("review") / "decisions.json"
    app = create_app(None, path, demo_size=1500)
    return TestClient(app), path


def test_page_and_summary(client):
    c, _ = client
    page = c.get("/")
    assert page.status_code == 200 and "Household duplicate review" in page.text
    s = c.get("/api/summary").json()
    assert s["queue"] > 0 and s["auto_match"] > 0 and s["auto_non_match"] > 0
    assert s["candidate_pairs"] == s["queue"] + s["auto_match"] + s["auto_non_match"]


def test_queue_is_most_uncertain_first(client):
    c, _ = client
    q = c.get("/api/queue").json()
    gaps = [abs(item["p"] - 0.5) for item in q]
    # probabilities are rounded to 3 decimals in the API
    assert all(a <= b + 0.001 for a, b in zip(gaps, gaps[1:]))


def test_pair_detail_explains_itself(client):
    c, _ = client
    p = c.get("/api/pair/0").json()
    assert {x["field"] for x in p["contributions"]} == {"name", "father", "dob", "phone", "location", "household"}
    assert set(p["compare"]) >= {"given_name", "date_of_birth", "phone", "village"}
    assert c.get("/api/pair/999999").status_code == 404


def test_decide_persist_and_export(client):
    c, path = client
    s = c.post("/api/decide", json={"k": 0, "decision": "same"}).json()
    assert s["decided"]["same"] == 1
    c.post("/api/decide", json={"k": 1, "decision": "different"})
    c.post("/api/decide", json={"k": 1, "decision": None})  # undo
    assert c.post("/api/decide", json={"k": 2, "decision": "maybe"}).status_code == 422
    assert path.exists() and "same" in path.read_text(encoding="utf-8")

    r = c.get("/api/export")
    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    reviewer = [row for row in rows if row["decided_by"] == "reviewer"]
    assert len(reviewer) == 1 and reviewer[0]["decision"] == "same"
    assert any(row["decided_by"].startswith("automatic") for row in rows)
