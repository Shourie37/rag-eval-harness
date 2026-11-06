import pytest

from rageval.judge import LLMJudge, parse_judge_output
from rageval.llm import Completion


def test_parse_judge_output_scales_score() -> None:
    result = parse_judge_output('Here you go: {"score": 1, "reason": "partial"}')
    assert result.score == 0.5
    assert result.reason == "partial"


@pytest.mark.parametrize("text", ["no json", '{"score": 3}'])
def test_parse_judge_output_rejects_bad_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_judge_output(text)


class _FakeModel:
    name = "fake"

    def __init__(self) -> None:
        self.user = ""

    def complete(self, system: str, user: str) -> Completion:
        self.user = user
        return Completion('{"score": 2, "reason": "all facts"}', 50, 10)


def test_judge_sends_facts_and_answer() -> None:
    model = _FakeModel()
    result = LLMJudge(model).grade("Fee?", ["$15"], "The fee is $15.")
    assert result.score == 1.0
    assert "- $15" in model.user and "The fee is $15." in model.user
