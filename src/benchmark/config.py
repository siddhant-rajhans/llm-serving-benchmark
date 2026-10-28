from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


class TrafficPattern(str, Enum):
    UNIFORM = "uniform"
    BURSTY = "bursty"
    RAMP_UP = "ramp_up"
    CYCLIC = "cyclic"
    SPIKE = "spike"


@dataclass
class BackendConfig:
    backend_type: BackendType
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    enable_prefix_caching: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


@dataclass
class TrafficConfig:
    pattern: TrafficPattern = TrafficPattern.UNIFORM
    num_requests: int = 500
    concurrency: int = 16
    max_tokens: int = 256
    prompt_lengths: list[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    ramp_up_seconds: float = 30.0
    burst_size: int = 50
    burst_interval_seconds: float = 5.0
    cycle_period_seconds: float = 60.0
    spike_multiplier: float = 5.0


@dataclass
class CostConfig:
    cost_per_gpu_hour: float = 2.50
    input_token_cost: float = 0.0
    output_token_cost: float = 0.0


@dataclass
class BenchmarkConfig:
    backend: BackendConfig = field(default_factory=lambda: BackendConfig(backend_type=BackendType.VLLM))
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    warmup_requests: int = 10
    output_dir: Path = Path("results")
    run_name: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}")

        try:
            backend_raw = raw.get("backend", {})
            backend_raw["backend_type"] = BackendType(backend_raw.get("backend_type", "vllm"))
            backend = BackendConfig(**backend_raw)

            traffic_raw = raw.get("traffic", {})
            if "pattern" in traffic_raw:
                traffic_raw["pattern"] = TrafficPattern(traffic_raw["pattern"])
            traffic = TrafficConfig(**traffic_raw)

            cost_raw = raw.get("cost", {})
            cost = CostConfig(**cost_raw)

            return cls(
                backend=backend,
                traffic=traffic,
                cost=cost,
                warmup_requests=raw.get("warmup_requests", 10),
                output_dir=Path(raw.get("output_dir", "results")),
                run_name=raw.get("run_name", ""),
            )
        except TypeError as e:
            raise ValueError(f"Unexpected field in {path}: {e}") from e

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "backend": {
                "backend_type": self.backend.backend_type.value,
                "model": self.backend.model,
                "host": self.backend.host,
                "port": self.backend.port,
                "tensor_parallel_size": self.backend.tensor_parallel_size,
                "gpu_memory_utilization": self.backend.gpu_memory_utilization,
                "max_model_len": self.backend.max_model_len,
                "enable_prefix_caching": self.backend.enable_prefix_caching,
            },
            "traffic": {
                "pattern": self.traffic.pattern.value,
                "num_requests": self.traffic.num_requests,
                "concurrency": self.traffic.concurrency,
                "max_tokens": self.traffic.max_tokens,
                "prompt_lengths": self.traffic.prompt_lengths,
                "ramp_up_seconds": self.traffic.ramp_up_seconds,
                "burst_size": self.traffic.burst_size,
                "burst_interval_seconds": self.traffic.burst_interval_seconds,
                "cycle_period_seconds": self.traffic.cycle_period_seconds,
                "spike_multiplier": self.traffic.spike_multiplier,
            },
            "cost": {
                "cost_per_gpu_hour": self.cost.cost_per_gpu_hour,
            },
            "warmup_requests": self.warmup_requests,
            "output_dir": str(self.output_dir),
            "run_name": self.run_name,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
