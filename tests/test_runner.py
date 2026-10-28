from __future__ import annotations

import numpy as np
import pytest

from benchmark.runner import BenchmarkResult, RequestResult
from benchmark.optimizations import PrefixCacheConfig, SmallModelRouter


def _make_result(latency: float, success: bool = True) -> RequestResult:
    return RequestResult(
        latency=latency,
        time_to_first_token=latency * 0.1,
        tokens_generated=50,
        prompt_tokens=100,
        tokens_per_second=50.0 / latency if latency > 0 else 0.0,
        success=success,
        error=None if success else "timeout",
    )


class TestBenchmarkResult:
    def test_success_rate(self) -> None:
        result = BenchmarkResult(
            backend="vllm", model="test", pattern="uniform",
            num_requests=10, concurrency=4,
            results=[_make_result(1.0)] * 8 + [_make_result(1.0, success=False)] * 2,
        )
        assert result.success_rate == pytest.approx(0.8)

    def test_percentiles(self) -> None:
        latencies = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
        result = BenchmarkResult(
            backend="vllm", model="test", pattern="uniform",
            num_requests=10, concurrency=4,
            results=[_make_result(l) for l in latencies],
        )
        assert result.p50_latency == pytest.approx(np.percentile(latencies, 50))
        assert result.p95_latency == pytest.approx(np.percentile(latencies, 95))
        assert result.p99_latency == pytest.approx(np.percentile(latencies, 99))

    def test_empty_results(self) -> None:
        result = BenchmarkResult(
            backend="vllm", model="test", pattern="uniform",
            num_requests=0, concurrency=4,
        )
        assert result.success_rate == 0.0
        assert result.p50_latency == 0.0
        assert result.throughput_rps == 0.0

    def test_throughput(self) -> None:
        result = BenchmarkResult(
            backend="vllm", model="test", pattern="uniform",
            num_requests=100, concurrency=4,
            results=[_make_result(1.0)] * 100,
        )
        assert result.throughput_rps == pytest.approx(100.0)

    def test_cost_per_million_tokens(self) -> None:
        result = BenchmarkResult(
            backend="vllm", model="test", pattern="uniform",
            num_requests=10, concurrency=4,
            results=[_make_result(1.0)] * 10,
        )
        cost = result.cost_per_million_tokens(2.50)
        assert cost > 0

    def test_summary_keys(self) -> None:
        result = BenchmarkResult(
            backend="vllm", model="test", pattern="uniform",
            num_requests=5, concurrency=2,
            results=[_make_result(1.0)] * 5,
        )
        summary = result.summary()
        expected_keys = {
            "backend", "model", "pattern", "num_requests", "success_rate",
            "p50_latency_s", "p95_latency_s", "p99_latency_s", "mean_ttft_s",
            "throughput_rps", "tokens_per_second", "cost_per_M_tokens",
        }
        assert set(summary.keys()) == expected_keys


class TestPrefixCacheConfig:
    def test_disabled(self) -> None:
        cfg = PrefixCacheConfig(enabled=False)
        assert cfg.apply_shared_prefix("hello") == "hello"
        assert cfg.estimated_cache_hit_rate(["hello"]) == 0.0

    def test_enabled_with_prefix(self) -> None:
        cfg = PrefixCacheConfig(enabled=True, shared_prefix="System: You are helpful.")
        result = cfg.apply_shared_prefix("What is 2+2?")
        assert result.startswith("System: You are helpful.")
        assert "What is 2+2?" in result

    def test_cache_hit_rate(self) -> None:
        cfg = PrefixCacheConfig(enabled=True, shared_prefix="a b c d e", prefix_token_count=5)
        prompts = ["one two three four five six seven eight nine ten"] * 10
        rate = cfg.estimated_cache_hit_rate(prompts)
        assert 0.0 < rate <= 1.0


class TestSmallModelRouter:
    def test_simple_request(self) -> None:
        router = SmallModelRouter()
        assert router.classify_complexity("Hello") == "simple"
        assert router.route("Hello") == router.small_model

    def test_complex_request(self) -> None:
        router = SmallModelRouter()
        prompt = "Please analyze and compare the economic impacts of " + "word " * 250
        assert router.classify_complexity(prompt) == "complex"
        assert router.route(prompt) == router.large_model

    def test_keyword_escalation(self) -> None:
        router = SmallModelRouter()
        assert router.classify_complexity("Please analyze this data") == "complex"
        assert router.classify_complexity("Hi there") == "simple"

    def test_cost_savings_estimate(self) -> None:
        router = SmallModelRouter()
        prompts = ["Hi"] * 70 + ["Please analyze and compare " + "word " * 250] * 30
        stats = router.estimate_cost_savings(prompts, large_cost=0.015, small_cost=0.002)
        assert stats["total_requests"] == 100
        assert stats["savings"] > 0
        assert 0.0 < stats["savings_pct"] < 1.0
