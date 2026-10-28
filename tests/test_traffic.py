from __future__ import annotations

import pytest

from benchmark.config import TrafficConfig, TrafficPattern
from benchmark.traffic import (
    Request,
    get_traffic_generator,
    uniform,
    bursty,
    ramp_up,
    cyclic,
    spike,
)


class TestTrafficGenerators:
    def _make_config(self, pattern: TrafficPattern, num_requests: int = 50) -> TrafficConfig:
        return TrafficConfig(pattern=pattern, num_requests=num_requests, concurrency=4)

    def test_uniform_count(self) -> None:
        cfg = self._make_config(TrafficPattern.UNIFORM, 100)
        requests = list(uniform(cfg))
        assert len(requests) == 100

    def test_uniform_scheduling(self) -> None:
        cfg = self._make_config(TrafficPattern.UNIFORM, 20)
        requests = list(uniform(cfg))
        times = [r.scheduled_at for r in requests]
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]

    def test_bursty_count(self) -> None:
        cfg = self._make_config(TrafficPattern.BURSTY, 100)
        cfg.burst_size = 25
        requests = list(bursty(cfg))
        assert len(requests) == 100

    def test_ramp_up_count(self) -> None:
        cfg = self._make_config(TrafficPattern.RAMP_UP, 80)
        requests = list(ramp_up(cfg))
        assert len(requests) == 80

    def test_cyclic_count(self) -> None:
        cfg = self._make_config(TrafficPattern.CYCLIC, 60)
        requests = list(cyclic(cfg))
        assert len(requests) == 60

    def test_spike_count(self) -> None:
        cfg = self._make_config(TrafficPattern.SPIKE, 90)
        requests = list(spike(cfg))
        assert len(requests) == 90

    def test_all_requests_have_content(self) -> None:
        cfg = self._make_config(TrafficPattern.UNIFORM, 10)
        for req in uniform(cfg):
            assert isinstance(req, Request)
            assert len(req.prompt) > 0
            assert req.max_tokens == cfg.max_tokens

    def test_get_traffic_generator_all_patterns(self) -> None:
        for pattern in TrafficPattern:
            cfg = self._make_config(pattern, 10)
            gen = get_traffic_generator(cfg)
            requests = list(gen)
            assert len(requests) == 10

    def test_unknown_pattern_raises(self) -> None:
        cfg = self._make_config(TrafficPattern.UNIFORM, 10)
        cfg.pattern = TrafficPattern("uniform")
        requests = list(get_traffic_generator(cfg))
        assert len(requests) == 10
