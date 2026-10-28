from __future__ import annotations

from benchmark.backends.base import ServingBackend
from benchmark.config import BackendConfig


class VLLMBackend(ServingBackend):
    def build_command(self) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len),
        ]

        if self.config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        for key, value in self.config.extra_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        return cmd

    @classmethod
    def from_config(cls, config: BackendConfig) -> VLLMBackend:
        return cls(config)
