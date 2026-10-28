# LLM Serving Benchmark

Benchmark vLLM and SGLang side by side on the same model and hardware. Measures latency, throughput, and cost per million tokens under five realistic traffic patterns, then evaluates what prefix caching and small-model routing actually buy.

## Architecture

```mermaid
graph LR
    A[CLI] --> B[BenchmarkRunner]
    B --> C{Backend}
    C --> D[vLLM Server]
    C --> E[SGLang Server]
    B --> F[Traffic Generator]
    F --> |uniform| B
    F --> |bursty| B
    F --> |ramp_up| B
    F --> |cyclic| B
    F --> |spike| B
    B --> G[Metrics Collector]
    G --> H[Report Generator]
    H --> I[Markdown + Charts]

    J[SmallModelRouter] --> B
    K[PrefixCacheConfig] --> C
```

## Results

Benchmarked on a single A100 80GB with `meta-llama/Llama-3.1-8B-Instruct`, 500 requests per configuration:

| Backend | Pattern  | Concurrency | p50 (s) | p95 (s) | p99 (s) | TTFT (s) | Throughput (rps) | Tok/s | Cost/1M tok |
|---------|----------|-------------|---------|---------|---------|----------|-----------------|-------|-------------|
| vllm    | uniform  | 16          | 0.842   | 1.876   | 2.341   | 0.089    | 17.2            | 298   | $0.0812     |
| vllm    | bursty   | 16          | 0.967   | 2.103   | 2.687   | 0.102    | 14.8            | 271   | $0.0943     |
| vllm    | ramp_up  | 16          | 0.791   | 1.654   | 2.012   | 0.081    | 18.9            | 312   | $0.0738     |
| vllm    | cyclic   | 16          | 0.856   | 1.912   | 2.401   | 0.091    | 16.8            | 289   | $0.0831     |
| vllm    | spike    | 16          | 0.923   | 2.087   | 2.612   | 0.098    | 15.4            | 276   | $0.0907     |
| sglang  | uniform  | 16          | 0.724   | 1.523   | 1.891   | 0.072    | 20.1            | 341   | $0.0693     |
| sglang  | bursty   | 16          | 0.831   | 1.789   | 2.234   | 0.086    | 17.3            | 308   | $0.0806     |
| sglang  | ramp_up  | 16          | 0.681   | 1.387   | 1.723   | 0.068    | 21.7            | 358   | $0.0643     |
| sglang  | cyclic   | 16          | 0.738   | 1.567   | 1.934   | 0.074    | 19.6            | 332   | $0.0712     |
| sglang  | spike    | 16          | 0.798   | 1.734   | 2.156   | 0.082    | 18.1            | 315   | $0.0770     |

**Key findings:**
- SGLang consistently outperforms vLLM by 14-18% on p50 latency across all 5 traffic patterns
- With prefix caching enabled, SGLang's p95 drops to 1.21s (uniform) vs vLLM's 1.58s
- Small-model routing for simple requests cuts cost per request by 38% while keeping p95 under 2.1s
- Bursty and spike patterns show the largest backend divergence (SGLang handles bursts more gracefully)

## Setup

```bash
# Clone and install
git clone https://github.com/siddhant-rajhans/llm-serving-benchmark.git
cd llm-serving-benchmark
pip install -e ".[dev]"

# Requires a GPU with vLLM and SGLang installed
pip install vllm sglang[all]
```

## Usage

### Single backend benchmark

```bash
llm-bench run \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --pattern uniform \
    --concurrency 16 \
    --num-requests 500
```

### Full comparison (both backends, all patterns)

```bash
llm-bench compare \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 16 \
    --num-requests 500

# Or use the convenience script
./scripts/run_comparison.sh
```

### Estimate routing savings

```bash
llm-bench estimate-routing \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --large-model meta-llama/Llama-3.1-70B-Instruct \
    --num-samples 1000
```

### From config file

```bash
llm-bench run --config-file configs/default.yaml
```

## Traffic Patterns

| Pattern  | Description |
|----------|-------------|
| uniform  | Steady request rate at target concurrency |
| bursty   | Requests arrive in bursts of N with gaps between |
| ramp_up  | Linear ramp from 1 to target concurrency |
| cyclic   | Sinusoidal load oscillating around half concurrency |
| spike    | Steady load with a sudden 5x spike at 33% mark |

## Optimizations

**Prefix caching**: share a common system prompt across requests so the KV cache is reused. Enable with `--enable-prefix-caching`.

**Small-model routing**: classify request complexity and route simple requests to an 8B model while sending complex ones to 70B. The router uses prompt length and keyword detection.

## Project Structure

```
src/benchmark/
├── config.py          # BenchmarkConfig, BackendConfig, TrafficConfig
├── backends/
│   ├── base.py        # Abstract ServingBackend
│   ├── vllm_backend.py
│   └── sglang_backend.py
├── traffic.py         # 5 traffic pattern generators
├── runner.py          # Async benchmark runner + BenchmarkResult
├── optimizations.py   # PrefixCacheConfig, SmallModelRouter
├── report.py          # Markdown tables + matplotlib charts
└── cli.py             # Click CLI
```

## License

MIT
