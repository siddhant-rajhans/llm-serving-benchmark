from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.config import TrafficConfig


SAMPLE_PROMPTS = [
    "Explain the concept of {topic} in simple terms.",
    "Write a short summary of the following text: {text}",
    "What are the key differences between {a} and {b}?",
    "Generate a Python function that {task}.",
    "Translate the following to French: {text}",
]

TOPICS = [
    "quantum computing", "machine learning", "blockchain",
    "photosynthesis", "general relativity", "neural networks",
    "distributed systems", "compiler design", "operating systems",
]


@dataclass
class Request:
    prompt: str
    max_tokens: int
    scheduled_at: float


def _make_prompt(target_length: int) -> str:
    base = random.choice(SAMPLE_PROMPTS).format(
        topic=random.choice(TOPICS),
        text="Lorem ipsum " * (target_length // 15),
        a=random.choice(TOPICS),
        b=random.choice(TOPICS),
        task="calculates the Fibonacci sequence",
    )
    while len(base.split()) < target_length // 4:
        base += f" Additionally, consider the implications of {random.choice(TOPICS)}."
    return base


def uniform(config: TrafficConfig) -> Iterator[Request]:
    interval = 1.0 / config.concurrency
    start = time.monotonic()
    for i in range(config.num_requests):
        prompt_len = random.choice(config.prompt_lengths)
        yield Request(
            prompt=_make_prompt(prompt_len),
            max_tokens=config.max_tokens,
            scheduled_at=start + i * interval,
        )


def bursty(config: TrafficConfig) -> Iterator[Request]:
    start = time.monotonic()
    sent = 0
    burst_idx = 0
    while sent < config.num_requests:
        burst_start = start + burst_idx * config.burst_interval_seconds
        burst_count = min(config.burst_size, config.num_requests - sent)
        for j in range(burst_count):
            prompt_len = random.choice(config.prompt_lengths)
            yield Request(
                prompt=_make_prompt(prompt_len),
                max_tokens=config.max_tokens,
                scheduled_at=burst_start + j * 0.01,
            )
            sent += 1
        burst_idx += 1


def ramp_up(config: TrafficConfig) -> Iterator[Request]:
    start = time.monotonic()
    total_time = config.ramp_up_seconds
    for i in range(config.num_requests):
        progress = i / max(config.num_requests - 1, 1)
        rate = 1.0 + progress * (config.concurrency - 1)
        delay = i / rate
        prompt_len = random.choice(config.prompt_lengths)
        yield Request(
            prompt=_make_prompt(prompt_len),
            max_tokens=config.max_tokens,
            scheduled_at=start + delay * (total_time / config.num_requests),
        )


def cyclic(config: TrafficConfig) -> Iterator[Request]:
    start = time.monotonic()
    period = config.cycle_period_seconds
    base_rate = config.concurrency / 2
    for i in range(config.num_requests):
        phase = (i / config.num_requests) * 2 * math.pi * 3
        rate = base_rate + base_rate * math.sin(phase)
        rate = max(rate, 1.0)
        delay = i / rate
        prompt_len = random.choice(config.prompt_lengths)
        yield Request(
            prompt=_make_prompt(prompt_len),
            max_tokens=config.max_tokens,
            scheduled_at=start + delay,
        )


def spike(config: TrafficConfig) -> Iterator[Request]:
    start = time.monotonic()
    base_interval = 1.0 / config.concurrency
    spike_start = config.num_requests // 3
    spike_end = spike_start + config.num_requests // 10
    offset = 0.0
    for i in range(config.num_requests):
        if spike_start <= i < spike_end:
            interval = base_interval / config.spike_multiplier
        else:
            interval = base_interval
        offset += interval
        prompt_len = random.choice(config.prompt_lengths)
        yield Request(
            prompt=_make_prompt(prompt_len),
            max_tokens=config.max_tokens,
            scheduled_at=start + offset,
        )


GENERATORS = {
    "uniform": uniform,
    "bursty": bursty,
    "ramp_up": ramp_up,
    "cyclic": cyclic,
    "spike": spike,
}


def get_traffic_generator(config: TrafficConfig) -> Iterator[Request]:
    gen_fn = GENERATORS.get(config.pattern.value)
    if gen_fn is None:
        raise ValueError(f"Unknown traffic pattern: {config.pattern}")
    return gen_fn(config)
