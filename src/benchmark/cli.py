from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from benchmark.config import BenchmarkConfig, BackendConfig, BackendType, TrafficPattern
from benchmark.runner import BenchmarkRunner, BenchmarkResult
from benchmark.report import generate_report, generate_markdown_table
from benchmark.optimizations import SmallModelRouter

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
def cli() -> None:
    """LLM Serving Benchmark: compare vLLM and SGLang performance."""
    pass


@cli.command()
@click.option("--backend", type=click.Choice(["vllm", "sglang"]), required=True)
@click.option("--model", default="meta-llama/Llama-3.1-8B-Instruct")
@click.option("--pattern", type=click.Choice(["uniform", "bursty", "ramp_up", "cyclic", "spike"]), default="uniform")
@click.option("--concurrency", default=16, type=int)
@click.option("--num-requests", default=500, type=int)
@click.option("--max-tokens", default=256, type=int)
@click.option("--enable-prefix-caching", is_flag=True)
@click.option("--output-dir", default="results", type=click.Path())
@click.option("--config-file", type=click.Path(exists=True), default=None)
def run(
    backend: str,
    model: str,
    pattern: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    enable_prefix_caching: bool,
    output_dir: str,
    config_file: str | None,
) -> None:
    """Run a benchmark against a single backend."""
    if config_file:
        config = BenchmarkConfig.from_yaml(config_file)
    else:
        config = BenchmarkConfig(
            backend=BackendConfig(
                backend_type=BackendType(backend),
                model=model,
                enable_prefix_caching=enable_prefix_caching,
            ),
            traffic=BenchmarkConfig().traffic,
            output_dir=Path(output_dir),
        )
        config.traffic.pattern = TrafficPattern(pattern)
        config.traffic.concurrency = concurrency
        config.traffic.num_requests = num_requests
        config.traffic.max_tokens = max_tokens

    console.print(f"[bold]Running benchmark:[/bold] {backend} / {pattern} / concurrency={concurrency}")

    runner = BenchmarkRunner(config)
    result = runner.run_sync()

    _print_result_table([result])
    console.print(f"\n[green]Done.[/green] {len(result.successful)}/{len(result.results)} requests succeeded.")


@cli.command()
@click.option("--model", default="meta-llama/Llama-3.1-8B-Instruct")
@click.option("--patterns", default="uniform,bursty,ramp_up,cyclic,spike")
@click.option("--concurrency", default=16, type=int)
@click.option("--num-requests", default=500, type=int)
@click.option("--output-dir", default="results", type=click.Path())
def compare(
    model: str,
    patterns: str,
    concurrency: int,
    num_requests: int,
    output_dir: str,
) -> None:
    """Run benchmarks across both backends and all traffic patterns."""
    pattern_list = [p.strip() for p in patterns.split(",")]
    all_results: list[BenchmarkResult] = []
    out = Path(output_dir)

    for backend_type in [BackendType.VLLM, BackendType.SGLANG]:
        for pattern in pattern_list:
            console.print(f"\n[bold cyan]>>> {backend_type.value} / {pattern}[/bold cyan]")
            config = BenchmarkConfig(
                backend=BackendConfig(backend_type=backend_type, model=model),
                output_dir=out,
            )
            config.traffic.pattern = TrafficPattern(pattern)
            config.traffic.concurrency = concurrency
            config.traffic.num_requests = num_requests

            runner = BenchmarkRunner(config)
            result = runner.run_sync()
            all_results.append(result)

    console.print("\n[bold]Full comparison:[/bold]")
    _print_result_table(all_results)

    report_path = generate_report(all_results, out)
    console.print(f"\n[green]Report saved to {report_path}[/green]")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
def report(results_dir: str) -> None:
    """Regenerate report from saved results."""
    import json
    results_path = Path(results_dir) / "results.json"
    if not results_path.exists():
        console.print("[red]No results.json found in the given directory.[/red]")
        raise SystemExit(1)

    raw = json.loads(results_path.read_text())
    console.print(f"Loaded {len(raw)} result entries.")
    for entry in raw:
        console.print(f"  {entry['backend']} / {entry['pattern']}: p95={entry['p95_latency_s']}")


@cli.command()
@click.option("--model", default="meta-llama/Llama-3.1-8B-Instruct")
@click.option("--large-model", default="meta-llama/Llama-3.1-70B-Instruct")
@click.option("--num-samples", default=1000, type=int)
def estimate_routing(model: str, large_model: str, num_samples: int) -> None:
    """Estimate cost savings from small-model routing."""
    from benchmark.traffic import _make_prompt
    import random

    router = SmallModelRouter(large_model=large_model, small_model=model)
    prompts = [_make_prompt(random.choice([64, 128, 256, 512, 1024])) for _ in range(num_samples)]

    large_cost_per_req = 0.015
    small_cost_per_req = 0.002
    stats = router.estimate_cost_savings(prompts, large_cost_per_req, small_cost_per_req)

    table = Table(title="Routing Cost Estimate")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total requests", str(stats["total_requests"]))
    table.add_row("Simple (routed to small)", str(stats["simple_requests"]))
    table.add_row("Complex (kept on large)", str(stats["complex_requests"]))
    table.add_row("Routing rate", f"{stats['routing_rate']:.1%}")
    table.add_row("Baseline cost", f"${stats['baseline_cost']:.2f}")
    table.add_row("Routed cost", f"${stats['routed_cost']:.2f}")
    table.add_row("Savings", f"${stats['savings']:.2f}")
    table.add_row("Savings %", f"{stats['savings_pct']:.1%}")

    console.print(table)


def _print_result_table(results: list[BenchmarkResult]) -> None:
    table = Table(title="Benchmark Results")
    for col in ["Backend", "Pattern", "p50", "p95", "p99", "TTFT", "RPS", "Tok/s", "$/1M tok"]:
        table.add_column(col)

    for r in results:
        s = r.summary()
        table.add_row(
            s["backend"], s["pattern"],
            s["p50_latency_s"], s["p95_latency_s"], s["p99_latency_s"],
            s["mean_ttft_s"], s["throughput_rps"], s["tokens_per_second"],
            s["cost_per_M_tokens"],
        )
    console.print(table)


if __name__ == "__main__":
    cli()
