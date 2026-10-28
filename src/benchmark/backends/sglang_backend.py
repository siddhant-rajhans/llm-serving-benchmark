from __future__ import annotations

from benchmark.backends.base import ServingBackend
from benchmark.config import BackendConfig


class SGLangBackend(ServingBackend):
    def build_command(self) -> list[str]:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.config.model,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tp", str(self.config.tensor_parallel_size),
            "--mem-fraction-static", str(self.config.gpu_memory_utilization),
        ]

        if self.config.enable_prefix_caching:
            cmd.extend(["--chunked-prefill-size", "8192"])

        for key, value in self.config.extra_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        return cmd

    @classmethod
    def from_config(cls, config: BackendConfig) -> SGLangBackend:
        return cls(config)
