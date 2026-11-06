from pathlib import Path

from rageval.__main__ import main
from rageval.report import render_report
from rageval.runner import load_config, run


def test_baseline_run_meets_floor() -> None:
    summary = run(load_config("configs/01-baseline.json")).summary()
    assert summary["recall_at_k"] >= 0.9
    assert summary["grounded_sentence_rate"] == 1.0
    assert summary["unanswerable_refused"] == 5


def test_hallucination_probe_is_caught() -> None:
    summary = run(load_config("configs/05-hallucination-probe.json")).summary()
    assert summary["hallucination_rate"] > 0.2
    assert summary["grounded_sentence_rate"] < 1.0


def test_report_lists_every_config() -> None:
    runs = [run(load_config(p)) for p in ("configs/01-baseline.json", "configs/04-k1.json")]
    report = render_report(runs)
    assert report.startswith("# RAG evaluation report")
    assert "| Config |" in report
    assert "| 01-baseline |" in report and "| 04-k1 |" in report
    assert "40 questions (35 answerable, 5 unanswerable)" in report


def test_cli_writes_report_and_details(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    code = main(
        ["run", "--config", "configs/01-baseline.json", "--out", str(out), "--details-dir", str(tmp_path / "runs")]
    )
    assert code == 0
    assert "01-baseline" in out.read_text()
    assert len((tmp_path / "runs" / "01-baseline.jsonl").read_text().splitlines()) == 40
