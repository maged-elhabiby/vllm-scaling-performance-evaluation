"""
Heatmaps, line plots, and error-bar plots for benchmark results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - needed when running headless on the GPU machine
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.stats import load_results, aggregate_iterations, flag_outliers


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# keep these in sync with experiment_grid.yaml
PROMPT_LENGTHS = [128, 512, 1024, 2048, 4096, 8192]
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]


def _pivot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = f"{metric}_mean"
    return summary.pivot(index="prompt_len", columns="concurrency", values=col)


def plot_heatmap(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    fmt: str = ".2f",
    cmap: str = "YlOrRd",
    out_path: Path | None = None,
) -> plt.Figure:
    grid = _pivot(summary, metric)
    fig, ax = plt.subplots(figsize=(9, 5))

    # origin="lower" puts short prompts at the bottom which reads more naturally
    im = ax.imshow(grid.values, aspect="auto", cmap=cmap, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Prompt Length (tokens)")
    ax.set_title(title)

    # annotate each cell with its value so the heatmap is readable without the colorbar
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            val = grid.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_vs_concurrency(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path | None = None,
    log_y: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(PROMPT_LENGTHS)))

    for color, pl in zip(color_cycle, PROMPT_LENGTHS):
        sub = summary[summary["prompt_len"] == pl].sort_values("concurrency")
        if sub.empty:
            continue
        mean_col = f"{metric}_mean"
        lo_col = f"{metric}_lo"
        hi_col = f"{metric}_hi"

        ax.plot(sub["concurrency"], sub[mean_col], marker="o", label=f"{pl} tok", color=color)
        # shade the 95% CI band if we have it - helps see where the 3 iterations disagree
        if lo_col in sub.columns and hi_col in sub.columns:
            ax.fill_between(
                sub["concurrency"],
                sub[lo_col], sub[hi_col],
                alpha=0.15, color=color,
            )

    # log2 x-axis because concurrency levels are powers of 2 - linear would bunch them all left
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Prompt Length", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def plot_error_heatmap(summary: pd.DataFrame, out_path: Path | None = None) -> plt.Figure:
    grid = _pivot(summary, "error_rate") * 100  # convert fraction to %
    fig, ax = plt.subplots(figsize=(9, 5))

    # clamp vmax to at least 1 so an all-zero grid doesn't collapse the colormap
    im = ax.imshow(grid.values, aspect="auto", cmap="Reds", origin="lower",
                   vmin=0, vmax=max(grid.values.max(), 1))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error Rate (%)")

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Prompt Length (tokens)")
    ax.set_title("Error Rate (%) — Prompt Length × Concurrency")

    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            val = grid.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig
def plot_ttft_vs_length(
        summary: pd.DataFrame,
        out_path: Path | None = None,

) -> plt.Figure:
    fig, ax = plt.subplots(figsize = (9, 5))
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(CONCURRENCY_LEVELS)))
    for color, cl in zip(color_cycle, CONCURRENCY_LEVELS):
        sub = summary[summary["concurrency"] == cl].sort_values("prompt_len")
        
        if sub.empty or "mean_ttft_mean" not in sub.columns:
            continue
        ax.plot(sub["prompt_len"], sub["mean_ttft_mean"], marker="o", label=f"{cl} conc", color=color)
        if "mean_ttft_lo" in sub.columns and "mean_ttft_hi" in sub.columns:

            ax.fill_between(
                sub["prompt_len"],
                sub["mean_ttft_lo"], sub["mean_ttft_hi"],
                alpha=0.15, color=color,
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks(PROMPT_LENGTHS)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Mean TTFT (s)")
    ax.set_title("Mean TTFT vs Prompt Length")
    ax.legend(title="Concurrency", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig
    

def generate_all(results_dir: str | Path, out_dir: str | Path, confidence: float = 0.95) -> None:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    df = load_results(results_dir)
    summary = aggregate_iterations(df, confidence=confidence)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"  {len(df)} runs aggregated into {len(summary)} cells.")

    print("  Plotting TTFT vs prompt length...")
    plot_ttft_vs_length(summary, out_path=out_dir / "line_ttft_vs_prompt_length.png")
    specs = [
        # (metric, title, fmt, cmap, ylabel, log_y)
        ("mean_latency",        "Mean E2E Latency (s)",     ".2f", "YlOrRd", "Latency (s)", False),
        ("p95_latency",         "P95 E2E Latency (s)",      ".2f", "OrRd",   "Latency (s)", False),
        ("p99_latency",         "P99 E2E Latency (s)",      ".2f", "Reds",   "Latency (s)", False),
        ("mean_ttft",           "Mean TTFT (s)",             ".3f", "YlGnBu", "TTFT (s)",    False),
        ("requests_per_second", "Requests / Second (RPS)",   ".2f", "YlGn",  "RPS",         False),
        ("tokens_per_second",   "Tokens / Second (TPS)",     ".0f", "Greens", "TPS",         False),
    ]

    for metric, title, fmt, cmap, ylabel, log_y in specs:
        if f"{metric}_mean" not in summary.columns:
            print(f"  [skip] {metric} not in results")
            continue

        print(f"  Plotting {metric}...")
        plot_heatmap(summary, metric, f"{title} — Prompt Length × Concurrency",
                     fmt=fmt, cmap=cmap,
                     out_path=out_dir / f"heatmap_{metric}.png")
        plot_vs_concurrency(summary, metric, f"{title} vs Concurrency", ylabel=ylabel,
                            log_y=log_y,
                            out_path=out_dir / f"line_{metric}_vs_concurrency.png")

    plot_error_heatmap(summary, out_path=out_dir / "heatmap_error_rate.png")
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark visualisations")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    generate_all(args.results_dir, args.out_dir, args.confidence)
