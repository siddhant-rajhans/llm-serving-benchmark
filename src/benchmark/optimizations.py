from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixCacheConfig:
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
        prefix_len = self.prefix_token_count or len(self.shared_prefix.split())
        total_tokens = sum(len(p.split()) for p in prompts)
        cached_tokens = prefix_len * len(prompts)
        return cached_tokens / total_tokens if total_tokens > 0 else 0.0


@dataclass
class SmallModelRouter:
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
