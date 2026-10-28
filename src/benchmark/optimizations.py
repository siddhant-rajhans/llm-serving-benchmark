from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrefixCacheConfig:
    """KV-cache reuse for a shared system prompt.

    When enabled, every request is prefixed with ``shared_prefix`` so the
    serving engine can reuse the KV cache for that prefix across requests,
    reducing TTFT for the shared portion.  ``estimated_cache_hit_rate``
    reports the fraction of total tokens that fall within the cached prefix.
    """

    enabled: bool = False
    shared_prefix: str = ""
    prefix_token_count: int = 0

    def apply_shared_prefix(self, prompt: str) -> str:
        if not self.enabled or not self.shared_prefix:
            return prompt
        return f"{self.shared_prefix}\n\n{prompt}"

    def estimated_cache_hit_rate(self, prompts: list[str]) -> float:
        if not self.enabled or not self.shared_prefix:
            return 0.0
        # Word count approximates token count; accurate enough for a
        # cache-hit ratio estimate without pulling in a tokenizer dependency.
        prefix_len = self.prefix_token_count or len(self.shared_prefix.split())
        total_tokens = sum(len(p.split()) for p in prompts)
        cached_tokens = prefix_len * len(prompts)
        return cached_tokens / total_tokens if total_tokens > 0 else 0.0


@dataclass
class SmallModelRouter:
    """Route simple requests to a smaller (cheaper) model.

    Uses a deliberate low-overhead heuristic (word count + keyword match)
    rather than an embedding classifier, because the goal is to measure
    *routing savings* in a benchmark with negligible classification latency.
    A production router would use a trained classifier or embedding similarity.
    """

    large_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    small_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    complexity_threshold: int = 200
    keyword_escalators: tuple[str, ...] = (
        "analyze", "compare", "explain in detail", "write a",
        "implement", "design", "evaluate", "critique",
    )

    def classify_complexity(self, prompt: str) -> str:
        word_count = len(prompt.split())
        has_complex_keywords = any(kw in prompt.lower() for kw in self.keyword_escalators)

        if word_count > self.complexity_threshold or has_complex_keywords:
            return "complex"
        return "simple"

    def route(self, prompt: str) -> str:
        complexity = self.classify_complexity(prompt)
        if complexity == "simple":
            return self.small_model
        return self.large_model

    def estimate_cost_savings(self, prompts: list[str], large_cost: float, small_cost: float) -> dict:
        simple_count = sum(1 for p in prompts if self.classify_complexity(p) == "simple")
        complex_count = len(prompts) - simple_count

        baseline_cost = len(prompts) * large_cost
        routed_cost = complex_count * large_cost + simple_count * small_cost
        savings = baseline_cost - routed_cost

        logger.info(
            "Routing: %d/%d (%.1f%%) simple -> %s, %d complex -> %s, savings=%.1f%%",
            simple_count, len(prompts), 100 * simple_count / len(prompts) if prompts else 0,
            self.small_model, complex_count, self.large_model,
            100 * savings / baseline_cost if baseline_cost > 0 else 0,
        )

        return {
            "total_requests": len(prompts),
            "simple_requests": simple_count,
            "complex_requests": complex_count,
            "routing_rate": simple_count / len(prompts) if prompts else 0.0,
            "baseline_cost": baseline_cost,
            "routed_cost": routed_cost,
            "savings": savings,
            "savings_pct": savings / baseline_cost if baseline_cost > 0 else 0.0,
        }
