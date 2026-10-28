from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from benchmark.config import BenchmarkConfig, BackendType
from benchmark.backends import VLLMBackend, SGLangBackend, ServingBackend
from benchmark.traffic import Request, get_traffic_generator

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    latency: float
    time_to_first_token: float
    tokens_generated: int
    prompt_tokens: int
    tokens_per_second: float
    success: bool
    error: str | None = None


@dataclass
class BenchmarkResult:
    backend: str
    model: str
    pattern: str
    num_requests: int
    concurrency: int
    wall_clock_seconds: float = 0.0
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len(self.successful) / len(self.results)

    @property
    def latencies(self) -> np.ndarray:
        return np.array([r.latency for r in self.successful])

    @property
    def p50_latency(self) -> float:
        lats = self.latencies
        return float(np.percentile(lats, 50)) if len(lats) > 0 else 0.0

    @property
    def p95_latency(self) -> float:
        lats = self.latencies
        return float(np.percentile(lats, 95)) if len(lats) > 0 else 0.0

    @property
    def p99_latency(self) -> float:
        lats = self.latencies
        return float(np.percentile(lats, 99)) if len(lats) > 0 else 0.0

    @property
    def mean_ttft(self) -> float:
        vals = [r.time_to_first_token for r in self.successful]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def throughput_rps(self) -> float:
        if not self.successful or self.wall_clock_seconds <= 0:
            return 0.0
        return len(self.successful) / self.wall_clock_seconds

    @property
    def tokens_per_second(self) -> float:
        vals = [r.tokens_per_second for r in self.successful]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_generated + r.prompt_tokens for r in self.successful)

    def cost_per_million_tokens(self, cost_per_gpu_hour: float) -> float:
        if not self.successful:
            return 0.0
        total_time_hours = sum(r.latency for r in self.successful) / 3600.0
        total_cost = total_time_hours * cost_per_gpu_hour
        tokens_m = self.total_tokens / 1_000_000
        return total_cost / tokens_m if tokens_m > 0 else 0.0

    def summary(self, cost_per_gpu_hour: float = 2.50) -> dict:
        return {
            "backend": self.backend,
            "model": self.model,
            "pattern": self.pattern,
            "num_requests": self.num_requests,
            "success_rate": f"{self.success_rate:.1%}",
            "p50_latency_s": f"{self.p50_latency:.3f}",
            "p95_latency_s": f"{self.p95_latency:.3f}",
            "p99_latency_s": f"{self.p99_latency:.3f}",
            "mean_ttft_s": f"{self.mean_ttft:.3f}",
            "throughput_rps": f"{self.throughput_rps:.1f}",
            "tokens_per_second": f"{self.tokens_per_second:.1f}",
            "cost_per_M_tokens": f"${self.cost_per_million_tokens(cost_per_gpu_hour):.4f}",
        }


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    def _create_backend(self) -> ServingBackend:
        if self.config.backend.backend_type == BackendType.VLLM:
            return VLLMBackend(self.config.backend)
        elif self.config.backend.backend_type == BackendType.SGLANG:
            return SGLangBackend(self.config.backend)
        raise ValueError(f"Unknown backend: {self.config.backend.backend_type}")

    async def _send_request(self, request: Request, endpoint: str) -> RequestResult:
        payload = {
            "model": self.config.backend.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "stream": True,
        }

        if self._client is None:
            raise RuntimeError("HTTP client not initialized; call run() instead of _send_request()")
        first_token_time: float | None = None
        tokens_generated = 0
        start = time.monotonic()

        try:
            async with self._client.stream("POST", endpoint, json=payload, timeout=120.0) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    tokens_generated += 1

            elapsed = time.monotonic() - start
            ttft = (first_token_time - start) if first_token_time else elapsed
            tps = tokens_generated / elapsed if elapsed > 0 else 0.0

            # Word count approximates token count for offline benchmarking;
            # production use should tokenize with the model's tokenizer.
            return RequestResult(
                latency=elapsed,
                time_to_first_token=ttft,
                tokens_generated=tokens_generated,
                prompt_tokens=len(request.prompt.split()),
                tokens_per_second=tps,
                success=True,
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            return RequestResult(
                latency=elapsed,
                time_to_first_token=0.0,
                tokens_generated=0,
                prompt_tokens=len(request.prompt.split()),
                tokens_per_second=0.0,
                success=False,
                error=str(e),
            )

    async def _warmup(self, endpoint: str) -> None:
        logger.info("Warming up with %d requests...", self.config.warmup_requests)
        for _ in range(self.config.warmup_requests):
            payload = {
                "model": self.config.backend.model,
                "messages": [{"role": "user", "content": "Hello, world!"}],
                "max_tokens": 16,
            }
            if self._client is None:
                raise RuntimeError("HTTP client not initialized")
            try:
                resp = await self._client.post(endpoint, json=payload, timeout=60.0)
                resp.raise_for_status()
            except Exception:
                pass

    async def _run_traffic(self, endpoint: str) -> tuple[list[RequestResult], float]:
        traffic = list(get_traffic_generator(self.config.traffic))
        logger.info(
            "Running %d requests with pattern=%s concurrency=%d",
            len(traffic), self.config.traffic.pattern.value, self.config.traffic.concurrency,
        )

        semaphore = asyncio.Semaphore(self.config.traffic.concurrency)
        results: list[RequestResult] = []

        async def _bounded_request(req: Request) -> RequestResult:
            now = time.monotonic()
            if req.scheduled_at > now:
                await asyncio.sleep(req.scheduled_at - now)
            async with semaphore:
                return await self._send_request(req, endpoint)

        wall_start = time.monotonic()
        tasks = [asyncio.create_task(_bounded_request(r)) for r in traffic]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info("Completed %d/%d requests", i + 1, len(traffic))
        wall_clock = time.monotonic() - wall_start

        return results, wall_clock

    async def run(self) -> BenchmarkResult:
        backend = self._create_backend()
        result = BenchmarkResult(
            backend=self.config.backend.backend_type.value,
            model=self.config.backend.model,
            pattern=self.config.traffic.pattern.value,
            num_requests=self.config.traffic.num_requests,
            concurrency=self.config.traffic.concurrency,
        )

        self._client = httpx.AsyncClient()
        try:
            with backend:
                endpoint = backend.get_endpoint()
                await self._warmup(endpoint)
                result.results, result.wall_clock_seconds = await self._run_traffic(endpoint)
        finally:
            await self._client.aclose()
            self._client = None

        succeeded = len(result.successful)
        failed = len(result.results) - succeeded
        logger.info(
            "Benchmark complete: %d succeeded, %d failed, p95=%.3fs, throughput=%.1f rps",
            succeeded, failed, result.p95_latency, result.throughput_rps,
        )
        return result

    def run_sync(self) -> BenchmarkResult:
        return asyncio.run(self.run())
