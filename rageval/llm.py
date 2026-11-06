"""Chat model adapters. Provider SDKs are imported lazily so the offline path needs none."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-sonnet-5-5"


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    refused_by_provider: bool = False


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Completion: ...


class ClaudeChatModel:
    """Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, max_tokens: int = 4096) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("install the 'anthropic' extra: pip install -e '.[anthropic]'") from exc
        self.name = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> Completion:
        response = self._client.messages.create(
            model=self.name,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            refused_by_provider=response.stop_reason == "refusal",
        )


class BedrockChatModel:
    """Placeholder adapter for Amazon Bedrock through the boto3 Converse API. Not exercised by tests."""

    def __init__(
        self, model_id: str = DEFAULT_BEDROCK_MODEL, region: str | None = None, max_tokens: int = 4096
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("install the 'bedrock' extra: pip install -e '.[bedrock]'") from exc
        self.name = model_id
        self._max_tokens = max_tokens
        self._client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def complete(self, system: str, user: str) -> Completion:
        response: dict[str, Any] = self._client.converse(
            modelId=self.name,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": self._max_tokens},
        )
        blocks = response["output"]["message"]["content"]
        text = "".join(b.get("text", "") for b in blocks)
        usage = response.get("usage", {})
        return Completion(text, usage.get("inputTokens", 0), usage.get("outputTokens", 0))


def make_chat_model(spec: dict[str, Any]) -> ChatModel:
    kind = spec.get("type")
    if kind == "claude":
        return ClaudeChatModel(model=spec.get("model", DEFAULT_CLAUDE_MODEL))
    if kind == "bedrock":
        return BedrockChatModel(model_id=spec.get("model", DEFAULT_BEDROCK_MODEL), region=spec.get("region"))
    raise ValueError(f"unknown chat model type {kind!r}")
