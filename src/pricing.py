"""Model pricing for cost estimation."""

import logging

from src.snapshot import SampleUsage

logger = logging.getLogger(__name__)

# Model name substring → (input $/M tokens, output $/M tokens, cached_input $/M tokens)
PRICING: dict[str, tuple[float, float, float]] = {
    "gemini-3.1-flash-lite-preview": (0.25, 1.50, 0.025),
    "gemini-3-flash": (0.50, 3.00, 0.05),
    "gemini-3-pro": (2.00, 12.00, 0.20),
    "gemini-3.1-pro": (2.00, 12.00, 0.20),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
    "gpt-5.2": (1.75, 14.00, 0.175),  # estimated, update when pricing is public
}


def _find_pricing(model_name: str) -> tuple[float, float, float] | None:
    """Find pricing entry by substring match against model name.

    Tries longest match first to avoid e.g. "gemini-3-flash" matching
    before "gemini-3-flash-lite" if both existed.
    """
    candidates = [(k, v) for k, v in PRICING.items() if k in model_name]
    if not candidates:
        return None
    # Longest matching key wins
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates[0][1]


def compute_cost(usage: SampleUsage, model_name: str) -> float:
    """Compute USD cost from token usage and model name.

    Returns 0.0 for unknown models (with a warning).
    """
    pricing = _find_pricing(model_name)
    if pricing is None:
        logger.warning("No pricing found for model %r, cost will be $0.00", model_name)
        return 0.0

    input_rate, output_rate, cached_rate = pricing
    non_cached_input = usage.input_tokens - usage.cache_read_tokens
    cost = (
        non_cached_input * input_rate + usage.output_tokens * output_rate + usage.cache_read_tokens * cached_rate
    ) / 1_000_000
    return cost
