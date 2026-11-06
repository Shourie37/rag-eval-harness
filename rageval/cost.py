"""Token estimates and list prices for cost projections."""

from __future__ import annotations

import math

# USD per million tokens (input, output). Check the provider price page before you rely on these.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def estimate_tokens(text: str) -> int:
    """Rough count: about four characters per token for English text."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES_PER_MTOK:
        raise KeyError(f"no price for model {model!r}")
    in_price, out_price = PRICES_PER_MTOK[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
