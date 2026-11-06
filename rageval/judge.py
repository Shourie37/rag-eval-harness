"""Optional LLM-as-judge for answer correctness."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rageval.llm import ChatModel

RUBRIC = """You grade answers from a support assistant for Acme Payments.
You get the question, the facts a correct answer must state, and the assistant answer.
Score 2 if the answer states every expected fact and adds no wrong claim.
Score 1 if the answer states some expected facts, or states all of them but adds a wrong claim.
Score 0 if the answer states no expected fact, contradicts one, or refuses.
Return only JSON: {"score": <0|1|2>, "reason": "<one sentence>"}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeResult:
    score: float
    reason: str


def parse_judge_output(text: str) -> JudgeResult:
    """Return a score in [0, 1]. Raise ValueError on malformed output."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"judge output has no JSON object: {text!r}")
    data = json.loads(match.group())
    raw = data["score"]
    if raw not in (0, 1, 2):
        raise ValueError(f"judge score out of range: {raw!r}")
    return JudgeResult(score=raw / 2, reason=str(data.get("reason", "")))


class LLMJudge:
    def __init__(self, model: ChatModel) -> None:
        self._model = model
        self.name = model.name

    def grade(self, question: str, expected_facts: Sequence[str], answer: str) -> JudgeResult:
        facts = "\n".join(f"- {f}" for f in expected_facts)
        user = f"Question: {question}\n\nExpected facts:\n{facts}\n\nAnswer:\n{answer}"
        return parse_judge_output(self._model.complete(RUBRIC, user).text)
