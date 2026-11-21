# beneficiary-dedupe

Finds duplicate household registrations across Arabic and Latin spellings, with
a review screen (FastAPI, port 8108). Setting: a humanitarian organization in
Lebanon (never named) registering households at its district centres, at
distribution points and at school shelters after displacement. All data is
synthetic (seeded generator). The repo is public.

## Run and test

- `python -m venv .venv`, `pip install -e .[dev]`
- `dedupe serve` (or `python -m beneficiary_dedupe serve`): review screen on 8108
- `dedupe run records.csv --out matches.csv`, `dedupe generate`, `dedupe bench`
- `dedupe train` refits `src/beneficiary_dedupe/model.json` on seed 1. Rerun it
  whenever the feature list in `features.py` changes; `Model.load` refuses a
  stale file.
- `pytest`: 84 tests, about 10 s.

## Layout

- `names.py` (names, villages, centres, teams), `generate.py` (duplicates,
  displaced households re-registered at shelters, hard negatives).
- `normalize.py`: normalisation, `skeleton` (shared key), `phonetic`
  (blocking key), dates, phones. `similarity.py`: cross-script name similarity.
- `blocking.py` -> `features.py` -> `model.py` (logistic regression, per-field
  contributions, review bands) -> `cluster.py` (chaining guard, hold set) ->
  `pipeline.py`. `app.py` + `static/index.html`: review screen (no CDN).
- `bench.py` reads the frozen held-out file `data/heldout.csv`.

## Rules and quirks

- Seeds: train 1, held-out 2 (frozen in `data/heldout.csv`), development 4,
  demo 3. Design and tune only on 1 and 4. Measure 2 once, at the end. If the
  generator changes, regenerate `data/heldout.csv` BEFORE any rule change
  (a test checks the file matches the generator).
- Pairs whose given names clearly differ are never auto-accepted or used as a
  merge link (twin policy in `Model.band` and `cluster(hold=...)`).
- `shelter_elsewhere` looks for "shelter" in the team or centre column.
- Blocked `dedupe.exe` on Windows: use `python -m beneficiary_dedupe`.
  Screenshots: `#pair-N` in the URL opens queue item N.
