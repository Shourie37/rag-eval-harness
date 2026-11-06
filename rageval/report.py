"""Render a markdown report that compares runs."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rageval.runner import RunResult

_COLUMNS = (
    ("Config", None),
    ("Chunk/overlap", None),
    ("Retriever", None),
    ("k", None),
    ("Generator", None),
    ("Recall@k", "recall_at_k"),
    ("MRR", "mrr"),
    ("Fact recall", "fact_recall"),
    ("Accuracy", "accuracy"),
    ("Grounded sent.", "grounded_sentence_rate"),
    ("Halluc. answers", "hallucination_rate"),
    ("Refusal acc.", "refusal_accuracy"),
    ("p50 ms", "latency_p50_ms"),
    ("p95 ms", "latency_p95_ms"),
    ("Est. $/1k q", "cost_per_1k_usd"),
)


def _fmt(key: str, value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if key.startswith("latency"):
        return f"{value:.2f}"
    if key == "cost_per_1k_usd":
        return f"{value:.3f}"
    return f"{value:.3f}"


def _retriever_label(r: RunResult) -> str:
    cfg = r.config.retrieval
    label = cfg.mode if cfg.mode == "bm25" else f"hybrid a={cfg.hybrid_alpha}"
    return label + " + rerank" if cfg.rerank else label


def results_table(runs: Sequence[RunResult]) -> str:
    lines = [
        "| " + " | ".join(name for name, _ in _COLUMNS) + " |",
        "|" + "|".join("---" if key is None else "---:" for _, key in _COLUMNS) + "|",
    ]
    for r in runs:
        s = r.summary()
        fixed = [
            r.config.name,
            f"{r.config.chunk_size}/{r.config.chunk_overlap}",
            _retriever_label(r),
            str(r.config.retrieval.k),
            r.generator_name,
        ]
        metrics = [_fmt(key, s[key]) for _, key in _COLUMNS if key is not None]
        lines.append("| " + " | ".join(fixed + metrics) + " |")
    return "\n".join(lines)


def render_report(runs: Sequence[RunResult], max_examples: int = 5) -> str:
    if not runs:
        raise ValueError("no runs to report")
    first = runs[0]
    n = len(first.results)
    n_unanswerable = sum(not q.answerable for q in first.results)
    parts = [
        "# RAG evaluation report",
        "",
        f"Eval set: {n} questions ({n - n_unanswerable} answerable, {n_unanswerable} unanswerable). "
        f"Corpus: {first.n_docs} documents.",
        "",
        "## Results",
        "",
        results_table(runs),
        "",
        "Recall@k and MRR use gold documents. Fact recall and accuracy cover answerable questions only. "
        "Grounded sent. is the share of answer sentences supported by the chunk they cite. "
        "Halluc. answers is the share of non-refused answers with at least one unsupported sentence. "
        "Refusal acc. counts a refusal as correct only for unanswerable questions. "
        "Est. $/1k q projects the token count onto each config's cost model.",
        "",
        "## Configurations",
        "",
    ]
    for r in runs:
        s = r.summary()
        parts.append(
            f"- **{r.config.name}**: {r.config.description} "
            f"Chunks: {r.n_chunks}. Cost model: `{r.config.cost_model}`. "
            f"Unanswerable refused: {s['unanswerable_refused']}/{n_unanswerable}. "
            f"False refusals: {s['false_refusals']}. "
            f"Avg tokens in/out: {s['avg_input_tokens']:.0f}/{s['avg_output_tokens']:.0f}."
        )
    parts += ["", "## Failure examples", ""]
    for r in runs:
        flagged = [(q, u) for q in r.results for u in q.unsupported][:max_examples]
        wrong = [q for q in r.results if not q.refusal_ok or (q.answerable and not q.correct)][:max_examples]
        parts.append(f"### {r.config.name}")
        parts.append("")
        if not flagged and not wrong:
            parts += ["No failures.", ""]
            continue
        for q, u in flagged:
            parts.append(f'- Unsupported ({q.id}): "{u["sentence"]}" ({u["reason"]})')
        for q in wrong:
            kind = "wrong refusal" if not q.refusal_ok else "missed fact"
            parts.append(f"- {kind.capitalize()} ({q.id}): {q.question}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
