"""Command line entry point: python -m rageval run --config configs/*.json"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rageval.report import render_report, results_table
from rageval.runner import load_config, run, write_details


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rageval", description="Evaluate RAG configurations.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run configs and write a comparison report")
    run_parser.add_argument("--config", nargs="+", required=True, help="one or more config JSON files")
    run_parser.add_argument("--out", default="reports/report.md", help="markdown report path")
    run_parser.add_argument("--details-dir", default="reports/runs", help="per-question JSONL output directory")
    args = parser.parse_args(argv)

    runs = [run(load_config(path)) for path in args.config]
    for r in runs:
        write_details(r, args.details_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(runs), encoding="utf-8")
    print(results_table(runs))
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
