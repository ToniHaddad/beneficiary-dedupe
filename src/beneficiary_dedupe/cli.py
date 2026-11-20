"""Command line: dedupe generate | train | run | serve | bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _generate(args: argparse.Namespace) -> None:
    from .generate import generate
    from .records import FIELDS, TRUTH_FIELDS, write_csv

    rows = generate(args.n, args.seed)
    fields = FIELDS + (TRUTH_FIELDS if args.truth else [])
    write_csv(args.out, rows, fields)
    print(f"Wrote {len(rows)} synthetic records to {args.out}" + (" (with ground truth columns)" if args.truth else ""))


def _train(args: argparse.Namespace) -> None:
    from .features import FEATURE_NAMES
    from .train import train

    model = train(args.n)
    print(f"Trained on {model.info['training_pairs']:,} candidate pairs "
          f"({model.info['training_matches']:,} true matches)")
    for name, w in zip(FEATURE_NAMES, model.weights):
        print(f"  {name:<22} {w:+.2f}")
    print(f"  {'baseline':<22} {model.bias:+.2f}")
    print(f"Bands: certain non-match <= {model.lower:.3f}, certain match >= {model.upper:.3f}")


def _run(args: argparse.Namespace) -> None:
    from .pipeline import clusters_table, matches_table, run
    from .records import read_csv, write_csv

    rows = read_csv(args.records)
    if not rows or "record_id" not in rows[0]:
        sys.exit("records file needs a record_id column")
    res = run(rows)
    matches = matches_table(res)
    write_csv(args.out, matches, list(matches[0].keys()) if matches else ["record_a", "record_b", "probability"])
    clusters_out = args.clusters or str(Path(args.out).with_name(Path(args.out).stem + "_households.csv"))
    write_csv(clusters_out, clusters_table(res), ["record_id", "household", "size"])
    bands = res.band_counts()
    multi = sum(1 for m in res.clusters.groups().values() if len(m) > 1)
    print(f"Read {len(rows):,} records")
    print(f"Compared {len(res.pairs):,} candidate pairs out of {len(rows) * (len(rows) - 1) // 2:,} possible")
    print(f"  certain matches:     {bands['match']:,}")
    print(f"  for a reviewer:      {bands['review']:,}")
    print(f"  certain non-matches: {bands['non-match']:,} (not written)")
    print(f"Households with more than one record: {multi:,}")
    print(f"Wrote {args.out} and {clusters_out} in {res.timings['total']:.1f} s")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .app import create_app

    app = create_app(args.records, args.decisions)
    print(f"Review screen: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _bench(args: argparse.Namespace) -> None:
    from .bench import main as bench_main

    bench_main(args.n)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="dedupe", description="Find duplicate household registrations across Arabic and Latin spellings.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a synthetic registration list")
    g.add_argument("--n", type=int, default=10_000)
    g.add_argument("--seed", type=int, default=3)
    g.add_argument("--out", default="records.csv")
    g.add_argument("--truth", action="store_true", help="include ground-truth columns")
    g.set_defaults(func=_generate)

    t = sub.add_parser("train", help="refit the scoring model on synthetic training data")
    t.add_argument("--n", type=int, default=10_000)
    t.set_defaults(func=_train)

    r = sub.add_parser("run", help="find duplicates in a CSV of records")
    r.add_argument("records")
    r.add_argument("--out", default="matches.csv")
    r.add_argument("--clusters", default=None, help="household assignment CSV (default: <out>_households.csv)")
    r.set_defaults(func=_run)

    s = sub.add_parser("serve", help="open the review screen")
    s.add_argument("records", nargs="?", default=None, help="CSV to review (default: a synthetic demo list)")
    s.add_argument("--decisions", default="decisions.json", help="where reviewer decisions are saved")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8108)
    s.set_defaults(func=_serve)

    b = sub.add_parser("bench", help="benchmark on a held-out synthetic set")
    b.add_argument("--n", type=int, default=10_000)
    b.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
