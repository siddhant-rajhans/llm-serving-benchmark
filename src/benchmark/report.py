from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

if TYPE_CHECKING:
    from benchmark.runner import BenchmarkResult


def generate_markdown_table(results: list[BenchmarkResult], cost_per_gpu_hour: float = 2.50) -> str:
    headers = [
        "Backend", "Pattern", "Concurrency", "p50 (s)", "p95 (s)", "p99 (s)",
        "TTFT (s)", "Throughput (rps)", "Tok/s", "Cost/1M tok", "Success",
    ]
    rows = []
    for r in results:
        s = r.summary(cost_per_gpu_hour)
        rows.append([
            s["backend"], s["pattern"], r.concurrency,
            s["p50_latency_s"], s["p95_latency_s"], s["p99_latency_s"],
            s["mean_ttft_s"], s["throughput_rps"], s["tokens_per_second"],
            s["cost_per_M_tokens"], s["success_rate"],
        ])
    return tabulate(rows, headers=headers, tablefmt="github")


def plot_latency_comparison(
    results: list[BenchmarkResult], output_path: Path
) -> Path:
    backends = sorted(set(r.backend for r in results))
    patterns = sorted(set(r.pattern for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Latency Comparison: vLLM vs SGLang", fontsize=14)

    for ax, (metric, label) in zip(
        axes,
        [("p50_latency", "p50"), ("p95_latency", "p95"), ("p99_latency", "p99")],
    ):
        x = np.arange(len(patterns))
        width = 0.35
        for i, backend in enumerate(backends):
            vals = []
            for pattern in patterns:
                matching = [r for r in results if r.backend == backend and r.pattern == pattern]
                vals.append(getattr(matching[0], metric) if matching else 0.0)
            offset = (i - len(backends) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=backend)

        ax.set_xlabel("Traffic Pattern")
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"{label} Latency")
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    chart_path = output_path / "latency_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def plot_throughput_comparison(
    results: list[BenchmarkResult], output_path: Path
) -> Path:
    backends = sorted(set(r.backend for r in results))
    patterns = sorted(set(r.pattern for r in results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Throughput & Cost Comparison", fontsize=14)

    x = np.arange(len(patterns))
    width = 0.35

    for i, backend in enumerate(backends):
        rps_vals = []
        cost_vals = []
        for pattern in patterns:
            matching = [r for r in results if r.backend == backend and r.pattern == pattern]
            if matching:
                rps_vals.append(matching[0].throughput_rps)
                cost_vals.append(matching[0].cost_per_million_tokens(2.50))
            else:
                rps_vals.append(0.0)
                cost_vals.append(0.0)

        offset = (i - len(backends) / 2 + 0.5) * width
        ax1.bar(x + offset, rps_vals, width, label=backend)
        ax2.bar(x + offset, cost_vals, width, label=backend)

    for ax, ylabel, title in [
        (ax1, "Requests/sec", "Throughput"),
        (ax2, "$/1M tokens", "Cost per Million Tokens"),
    ]:
        ax.set_xlabel("Traffic Pattern")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    chart_path = output_path / "throughput_cost_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def plot_latency_distribution(
    results: list[BenchmarkResult], output_path: Path
) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    fig.suptitle("Latency Distribution", fontsize=14)

    for ax, result in zip(axes, results):
        lats = result.latencies
        if len(lats) > 0:
            ax.hist(lats, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.axvline(result.p50_latency, color="green", linestyle="--", label=f"p50={result.p50_latency:.2f}s")
            ax.axvline(result.p95_latency, color="orange", linestyle="--", label=f"p95={result.p95_latency:.2f}s")
            ax.axvline(result.p99_latency, color="red", linestyle="--", label=f"p99={result.p99_latency:.2f}s")
        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"{result.backend} / {result.pattern}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    chart_path = output_path / "latency_distribution.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def generate_report(
    results: list[BenchmarkResult],
    output_dir: Path,
    cost_per_gpu_hour: float = 2.50,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    table = generate_markdown_table(results, cost_per_gpu_hour)

    latency_chart = plot_latency_comparison(results, output_dir)
    throughput_chart = plot_throughput_comparison(results, output_dir)
    dist_chart = plot_latency_distribution(results, output_dir)

    report_lines = [
        "# LLM Serving Benchmark Report\n",
        "## Summary\n",
        table,
        "\n\n## Latency Comparison\n",
        f"![Latency Comparison]({latency_chart.name})\n",
        "\n## Throughput & Cost\n",
        f"![Throughput & Cost]({throughput_chart.name})\n",
        "\n## Latency Distribution\n",
        f"![Latency Distribution]({dist_chart.name})\n",
    ]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))

    raw_data = output_dir / "results.json"
    raw_data.write_text(json.dumps(
        [r.summary(cost_per_gpu_hour) for r in results],
        indent=2,
    ))

    return report_path
