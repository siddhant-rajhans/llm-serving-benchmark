from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark.config import (
    BackendConfig,
    BackendType,
    BenchmarkConfig,
    CostConfig,
    TrafficConfig,
    TrafficPattern,
)


class TestBackendConfig:
    def test_default_urls(self) -> None:
        cfg = BackendConfig(backend_type=BackendType.VLLM)
        assert cfg.base_url == "http://0.0.0.0:8000"
        assert cfg.completions_url == "http://0.0.0.0:8000/v1/completions"
        assert cfg.chat_url == "http://0.0.0.0:8000/v1/chat/completions"
        assert cfg.health_url == "http://0.0.0.0:8000/health"

    def test_custom_port(self) -> None:
        cfg = BackendConfig(backend_type=BackendType.SGLANG, port=9090)
        assert cfg.base_url == "http://0.0.0.0:9090"

    def test_backend_type_enum(self) -> None:
        assert BackendType.VLLM.value == "vllm"
        assert BackendType.SGLANG.value == "sglang"


class TestTrafficConfig:
    def test_defaults(self) -> None:
        cfg = TrafficConfig()
        assert cfg.pattern == TrafficPattern.UNIFORM
        assert cfg.num_requests == 500
        assert cfg.concurrency == 16
        assert cfg.max_tokens == 256
        assert len(cfg.prompt_lengths) == 4

    def test_all_patterns(self) -> None:
        for pattern in TrafficPattern:
            cfg = TrafficConfig(pattern=pattern)
            assert cfg.pattern == pattern


class TestBenchmarkConfig:
    def test_defaults(self) -> None:
        cfg = BenchmarkConfig()
        assert cfg.warmup_requests == 10
        assert cfg.output_dir == Path("results")
        assert cfg.backend.backend_type == BackendType.VLLM

    def test_yaml_roundtrip(self) -> None:
        original = BenchmarkConfig(
            backend=BackendConfig(
                backend_type=BackendType.SGLANG,
                model="test-model",
                port=9090,
            ),
            traffic=TrafficConfig(
                pattern=TrafficPattern.BURSTY,
                num_requests=100,
                concurrency=8,
            ),
            cost=CostConfig(cost_per_gpu_hour=3.50),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            original.to_yaml(f.name)
            loaded = BenchmarkConfig.from_yaml(f.name)

        assert loaded.backend.backend_type == BackendType.SGLANG
        assert loaded.backend.model == "test-model"
        assert loaded.backend.port == 9090
        assert loaded.traffic.pattern == TrafficPattern.BURSTY
        assert loaded.traffic.num_requests == 100
        assert loaded.traffic.concurrency == 8

    def test_from_default_yaml(self) -> None:
        yaml_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if yaml_path.exists():
            cfg = BenchmarkConfig.from_yaml(yaml_path)
            assert cfg.backend.backend_type == BackendType.VLLM
            assert cfg.traffic.num_requests == 500
