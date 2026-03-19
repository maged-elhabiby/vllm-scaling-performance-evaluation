"""
Experiment orchestrator. Iterates over the Context Length × Concurrency grid,
runs each cell `iterations` times in randomised order, and saves results to disk.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Adjust import path when run as __main__
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[1]))

from benchmark.load_generator import run_warmup, run_load
from benchmark.metrics import aggregate, RunSummary
from prompts.prompt_builder import build_prompt_cache


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_run_plan(grid: dict, iterations: int, randomize: bool) -> list[tuple[int, int, int]]:
    # full factorial: every (prompt_len, concurrency) pair × every iteration
    plan = [
        (pl, cl, it)
        for pl in grid["prompt_lengths"]
        for cl in grid["concurrency_levels"]
        for it in range(1, iterations + 1)
    ]
    # randomize to avoid ordering bias - e.g. thermal throttling or KV cache effects
    # that could make later runs look slower/faster than earlier ones
    if randomize:
        random.shuffle(plan)
    return plan


def result_path(results_dir: Path, prompt_len: int, concurrency: int, iteration: int) -> Path:
    return results_dir / f"pl{prompt_len}_c{concurrency}_i{iteration}.json"


async def run_experiment(
    cfg: dict,
    grid: dict,
    results_dir: Path,
    dry_run: bool = False,
) -> list[RunSummary]:
    server = cfg["server"]
    model = cfg["model"]
    bench = cfg["benchmark"]

    api_base = server["api_base"]
    model_name = model["name"]
    max_tokens = model["max_tokens"]
    temperature = model["temperature"]
    top_p = model["top_p"]
    warmup_n = bench["warmup_requests"]
    duration = bench["test_duration_sec"]
    timeout = bench["request_timeout_sec"]
    iterations = bench["iterations"]
    inter_sleep = grid.get("inter_run_sleep_sec", 5)

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  vLLM Scaling Benchmark")
    print(f"  Model   : {model_name}")
    print(f"  Server  : {api_base}")
    print(f"  Grid    : {len(grid['prompt_lengths'])} prompt lengths × "
          f"{len(grid['concurrency_levels'])} concurrency levels × "
          f"{iterations} iterations")
    total = len(grid["prompt_lengths"]) * len(grid["concurrency_levels"]) * iterations
    print(f"  Total   : {total} runs × ~{duration}s = ~{total * duration / 60:.0f} min")
    print(f"{'='*60}\n")

    if dry_run:
        print("[dry-run] Building prompt cache to verify tokenizer...")

    print("Building prompt cache...")
    prompt_cache = build_prompt_cache(grid["prompt_lengths"], model_name)
    print("Prompt cache ready.\n")

    plan = build_run_plan(grid, iterations, grid.get("randomize_order", True))
    summaries: list[RunSummary] = []

    for run_idx, (prompt_len, concurrency, iteration) in enumerate(plan, start=1):
        out_path = result_path(results_dir, prompt_len, concurrency, iteration)

        # skip already-completed runs so interrupted experiments can resume
        # without re-running cells we already have data for
        if out_path.exists():
            print(f"[{run_idx:3d}/{len(plan)}] SKIP pl={prompt_len} c={concurrency} "
                  f"iter={iteration}  (already saved)")
            continue

        print(f"[{run_idx:3d}/{len(plan)}] pl={prompt_len:5d}  c={concurrency:2d}  "
              f"iter={iteration}  ", end="", flush=True)

        if dry_run:
            print("(dry-run, skipping)")
            continue

        prompt = prompt_cache[prompt_len]

        # warmup before each cell, not just once at the start - we want the GPU
        # in a consistent state for every (prompt_len, concurrency) pair
        # TODO: figure out if 10 warmup requests is enough for high concurrency cells
        #       might need more at c=32 to actually saturate the scheduler
        await run_warmup(
            api_base, model_name, prompt, max_tokens,
            temperature, top_p, timeout, warmup_n,
        )

        raw_results, elapsed = await run_load(
            api_base, model_name, prompt, prompt_len,
            max_tokens, temperature, top_p, timeout,
            concurrency, duration,
        )

        summary = aggregate(raw_results, prompt_len, concurrency, iteration, elapsed)
        summaries.append(summary)
        summary.to_json(out_path, include_raw=cfg["output"].get("save_raw", True))

        ok = summary.successful_requests
        rps = summary.requests_per_second
        tps = summary.tokens_per_second
        p99 = summary.p99_latency
        err = summary.error_rate * 100
        print(f"ok={ok:4d}  rps={rps:6.2f}  tps={tps:7.1f}  "
              f"p99={p99:7.2f}s  err={err:5.1f}%")

        # sleep between runs to let GPU memory/state settle
        # TODO: is 5s enough? might want to add a health check ping instead of fixed sleep
        if inter_sleep > 0:
            await asyncio.sleep(inter_sleep)

    return summaries


def main():
    parser = argparse.ArgumentParser(description="vLLM scaling benchmark runner")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--grid", default="configs/experiment_grid.yaml")
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan without executing requests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for run-order randomisation")
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = load_config(args.config)
    grid = load_config(args.grid)

    results_dir = Path(args.results_dir or cfg["output"]["results_dir"])

    asyncio.run(run_experiment(cfg, grid, results_dir, dry_run=args.dry_run))
    print("\nDone.")


if __name__ == "__main__":
    main()
