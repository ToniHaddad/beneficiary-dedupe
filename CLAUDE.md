# beneficiary-dedupe

Finds duplicate household registrations across Arabic and Latin spellings, with
a review screen (FastAPI, port 8108). All data is synthetic (seeded generator).

## Run and test

- `python -m venv .venv`, `pip install -e .[dev]`
- `dedupe serve` (or `python -m beneficiary_dedupe serve`): review screen on 8108
- `dedupe run records.csv --out matches.csv`, `dedupe generate`, `dedupe bench`
- `dedupe train` refits `src/beneficiary_dedupe/model.json` (seed 1). Rerun it
  whenever the feature list in `features.py` changes; `Model.load` refuses a
  stale file. Then rerun `dedupe bench` and update README numbers and
  `docs/benchmark.txt` from its real output.
- `pytest`: 81 tests, about 10 s.

## Layout

- `normalize.py`: Arabic/Latin normalisation, `skeleton` (shared key),
  `phonetic` (blocking key), dates, phones. `similarity.py`: vowel-aware
  cross-script name similarity.
- `blocking.py` -> `features.py` -> `model.py` (logistic regression, per-field
  contributions from a reference pair, review bands) -> `cluster.py` (chaining
  guard, hold set) -> `pipeline.py`.
- `app.py` + `static/index.html`: review screen (vanilla JS, no CDN).
- `bench.py`: held-out benchmark (seed 2). Demo list uses seed 3.

## Quirks

- Seeds: train 1, test 2, demo 3. Never benchmark on seed 1.
- Pairs whose given names clearly differ are never auto-accepted or used as a
  merge link (twin policy in `Model.band` and `cluster(hold=...)`).
- On Windows with Application Control, the `dedupe.exe` launcher may be
  blocked; use `python -m beneficiary_dedupe`.
- Screenshots: `#pair-N` in the URL opens queue item N.
