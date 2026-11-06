"""Run one configuration over the eval set and aggregate the metrics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rageval.corpus import chunk_corpus, load_corpus
from rageval.cost import cost_usd
from rageval.generation import (
    ExtractiveMockGenerator,
    Generator,
    HallucinatingMockGenerator,
    LLMGenerator,
    is_refusal,
)
from rageval.grounding import check_grounding
from rageval.judge import LLMJudge
from rageval.llm import make_chat_model
from rageval.metrics import fact_matches, mean, percentile, recall_at_k, reciprocal_rank, refusal_correct
from rageval.retriever import RetrievalConfig, Retriever


@dataclass(frozen=True)
class RunConfig:
    name: str
    description: str = ""
    corpus_dir: str = "data/corpus"
    eval_set: str = "data/eval/questions.json"
    chunk_size: int = 120
    chunk_overlap: int = 30
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generator: dict[str, Any] = field(default_factory=lambda: {"type": "mock"})
    cost_model: str = "claude-sonnet-5-5"
    judge: dict[str, Any] | None = None


def load_config(path: str | Path) -> RunConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    chunking = data.pop("chunking", {})
    retrieval = RetrievalConfig(**data.pop("retrieval", {}))
    return RunConfig(
        chunk_size=chunking.get("size", 120),
        chunk_overlap=chunking.get("overlap", 30),
        retrieval=retrieval,
        **data,
    )


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    answer: str
    refused: bool
    retrieved_chunk_ids: list[str]
    recall: float | None
    reciprocal_rank: float | None
    facts_matched: list[bool]
    grounded_sentences: int
    total_sentences: int
    unsupported: list[dict[str, str]]
    refusal_ok: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    judge_score: float | None = None

    @property
    def correct(self) -> bool:
        return self.answerable and not self.refused and all(self.facts_matched)


@dataclass
class RunResult:
    config: RunConfig
    generator_name: str
    n_docs: int
    n_chunks: int
    results: list[QuestionResult]

    def summary(self) -> dict[str, float]:
        answerable = [r for r in self.results if r.answerable]
        answered = [r for r in self.results if not r.refused]
        total_sentences = sum(r.total_sentences for r in answered)
        judged = [r.judge_score for r in answerable if r.judge_score is not None]
        n = len(self.results)
        in_tok = sum(r.input_tokens for r in self.results)
        out_tok = sum(r.output_tokens for r in self.results)
        return {
            "recall_at_k": mean([r.recall for r in answerable if r.recall is not None]),
            "mrr": mean([r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]),
            "fact_recall": mean(
                [0.0 if r.refused else sum(r.facts_matched) / len(r.facts_matched) for r in answerable]
            ),
            "accuracy": mean([float(r.correct) for r in answerable]),
            "grounded_sentence_rate": sum(r.grounded_sentences for r in answered) / total_sentences
            if total_sentences
            else float("nan"),
            "hallucination_rate": mean([float(bool(r.unsupported)) for r in answered]),
            "refusal_accuracy": mean([float(r.refusal_ok) for r in self.results]),
            "unanswerable_refused": sum(r.refused for r in self.results if not r.answerable),
            "false_refusals": sum(r.refused for r in answerable),
            "latency_p50_ms": percentile([r.latency_ms for r in self.results], 50),
            "latency_p95_ms": percentile([r.latency_ms for r in self.results], 95),
            "avg_input_tokens": in_tok / n,
            "avg_output_tokens": out_tok / n,
            "cost_per_1k_usd": cost_usd(self.config.cost_model, in_tok, out_tok) / n * 1000,
            "judge_score": mean(judged) if judged else float("nan"),
        }


def make_generator(spec: dict[str, Any], retriever: Retriever) -> Generator:
    kind = spec.get("type", "mock")
    if kind == "mock":
        return ExtractiveMockGenerator(retriever.bm25.idf, min_coverage=spec.get("min_coverage", 0.4))
    if kind == "mock-hallucinating":
        return HallucinatingMockGenerator(
            retriever.bm25.idf, rate=spec.get("rate", 0.3), min_coverage=spec.get("min_coverage", 0.4)
        )
    return LLMGenerator(make_chat_model(spec))


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(config: RunConfig) -> RunResult:
    docs = load_corpus(config.corpus_dir)
    chunks = chunk_corpus(docs, config.chunk_size, config.chunk_overlap)
    retriever = Retriever(chunks, config.retrieval)
    generator = make_generator(config.generator, retriever)
    judge = LLMJudge(make_chat_model(config.judge)) if config.judge else None

    results = []
    for q in load_questions(config.eval_set):
        start = time.perf_counter()
        hits = retriever.retrieve(q["question"])
        retrieved = [h.chunk for h in hits]
        answer = generator.answer(q["question"], retrieved)
        latency_ms = (time.perf_counter() - start) * 1000

        refused = is_refusal(answer.text)
        doc_ids = [c.doc_id for c in retrieved]
        grounding = check_grounding(answer.text, {c.chunk_id: c.text for c in retrieved}) if not refused else None
        answerable = q["answerable"]
        result = QuestionResult(
            id=q["id"],
            question=q["question"],
            answerable=answerable,
            answer=answer.text,
            refused=refused,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved],
            recall=recall_at_k(doc_ids, q["gold_doc_ids"], config.retrieval.k) if answerable else None,
            reciprocal_rank=reciprocal_rank(doc_ids, q["gold_doc_ids"]) if answerable else None,
            facts_matched=fact_matches(answer.text, q["expected_facts"]) if answerable else [],
            grounded_sentences=grounding.supported if grounding else 0,
            total_sentences=grounding.total if grounding else 0,
            unsupported=[{"sentence": v.sentence, "reason": v.reason} for v in grounding.unsupported]
            if grounding
            else [],
            refusal_ok=refusal_correct(refused, answerable),
            latency_ms=latency_ms,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
        )
        if judge and answerable:
            result.judge_score = judge.grade(q["question"], q["expected_facts"], answer.text).score
        results.append(result)
    return RunResult(config, generator.name, len(docs), len(chunks), results)


def write_details(run_result: RunResult, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_result.config.name}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in run_result.results:
            fh.write(json.dumps(asdict(r) | {"correct": r.correct}) + "\n")
    return path
