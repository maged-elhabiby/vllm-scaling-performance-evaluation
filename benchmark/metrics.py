"""
Data structures and aggregation functions for benchmark results
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class RequestResult:
    """Stores the outcome of a single HTTP request to the vLLM server."""

    request_id: str
    prompt_tokens: int           # intended token count (from experiment config)
    start_time: float            # perf_counter timestamp at request start
    end_time: float              # perf_counter timestamp at response end
    ttft: Optional[float]        # seconds from send to first token (None if request failed before any token)
    completion_tokens: int       # tokens in the response (0 on error)
    success: bool
    timed_out: bool = False
    error: Optional[str] = None

    @property
    def latency(self) -> float:
        # only meaningful for successful requests - failed ones return 0
        # so they don't pollute the latency distributions
        return self.end_time - self.start_time if self.success else 0.0


@dataclass
class RunSummary:
    """Aggregated metrics for one (prompt_len, concurrency, iteration) cell."""

    prompt_len: int
    concurrency: int
    iteration: int

    # request counts
    total_requests: int
    successful_requests: int
    failed_requests: int
    timed_out_requests: int

    # latency stats in seconds - computed over successful requests only
    # failed/timed-out requests are excluded so they don't skew distributions
    mean_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    std_latency: float

    # TTFT in seconds - None if no requests completed with streaming data
    mean_ttft: Optional[float]
    p95_ttft: Optional[float]
    p99_ttft: Optional[float]

    # throughput over the measurement window
    requests_per_second: float   # successful / elapsed
    tokens_per_second: float     # output tokens / elapsed (not counting prompt tokens)

    # reliability
    error_rate: float            # (failed) / total
    timeout_rate: float          # (timed_out) / total

    elapsed_sec: float           # actual wall time of the measurement window

    raw_results: list[RequestResult] = field(default_factory=list, repr=False)

    def to_dict(self, include_raw: bool = False) -> dict:
        d = asdict(self)
        if not include_raw:
            d.pop("raw_results", None)
        return d

    def to_json(self, path: Path, include_raw: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(include_raw=include_raw), f, indent=2)


def _percentile(sorted_values: list[float], p: float) -> float:
    # linear interpolation between adjacent ranks - same method as numpy percentile default
    if not sorted_values:
        return float("nan")
    idx = (p / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[lo] + (idx - lo) * (sorted_values[hi] - sorted_values[lo])


def aggregate(
    results: list[RequestResult],
    prompt_len: int,
    concurrency: int,
    iteration: int,
    elapsed_sec: float,
) -> RunSummary:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    timed_out = [r for r in results if r.timed_out]

    latencies = sorted(r.latency for r in successful)
    # some requests succeed but never get a token back (empty response) - exclude those from TTFT
    ttfts = sorted(r.ttft for r in successful if r.ttft is not None)

    total_output_tokens = sum(r.completion_tokens for r in successful)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def _std(xs: list[float]) -> float:
        if len(xs) < 2:
            return float("nan")
        m = _mean(xs)
        # sample std dev (n-1) since we're treating each run as a sample
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    return RunSummary(
        prompt_len=prompt_len,
        concurrency=concurrency,
        iteration=iteration,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        timed_out_requests=len(timed_out),
        mean_latency=_mean(latencies),
        p50_latency=_percentile(latencies, 50),
        p95_latency=_percentile(latencies, 95),
        p99_latency=_percentile(latencies, 99),
        min_latency=min(latencies) if latencies else float("nan"),
        max_latency=max(latencies) if latencies else float("nan"),
        std_latency=_std(latencies),
        mean_ttft=_mean(ttfts) if ttfts else None,
        p95_ttft=_percentile(ttfts, 95) if ttfts else None,
        p99_ttft=_percentile(ttfts, 99) if ttfts else None,
        requests_per_second=len(successful) / elapsed_sec if elapsed_sec > 0 else 0.0,
        tokens_per_second=total_output_tokens / elapsed_sec if elapsed_sec > 0 else 0.0,
        error_rate=len(failed) / len(results) if results else 0.0,
        timeout_rate=len(timed_out) / len(results) if results else 0.0,
        elapsed_sec=elapsed_sec,
        raw_results=results,
    )
