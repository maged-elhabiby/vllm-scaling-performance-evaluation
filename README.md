# vLLM Scaling Benchmark

Performance evaluation of [vLLM](https://github.com/vllm-project/vllm) under variable workloads.
Measures latency, throughput, and reliability across a full context length x concurrency grid.

## Project Structure

```
configs/
  base.yaml              # server URL, model, generation params, benchmark settings
  experiment_grid.yaml   # prompt lengths, concurrency levels, run options
prompts/
  templates.py           # padding text used to fill prompts to target length
  prompt_builder.py      # builds prompts to exact token counts via HF tokenizer
benchmark/
  metrics.py             # RequestResult dataclass + aggregation helpers
  load_generator.py      # async HTTP load generator (streaming, TTFT measurement)
  runner.py              # experiment orchestrator (full factorial, randomised order)
analysis/
  stats.py               # confidence intervals, outlier detection, CSV aggregation
  visualize.py           # heatmaps, line plots, error-bar plots
results/                 # output directory for raw JSON + aggregated CSV
figures/                 # output directory for generated plots
```

## Experimental Design

| Factor | Levels |
|---|---|
| Prompt length (tokens) | 128, 512, 1024, 2048, 4096, 8192 |
| Concurrency | 1, 2, 4, 8, 16, 32 |
| Iterations per cell | 3 (randomised run order) |
| Fixed output tokens | 128 |

Total runs: 6 x 6 x 3 = 108

## Metrics

| Category | Metric |
|---|---|
| Latency | mean, P50, P95, P99 end-to-end |
| TTFT | mean, P95, P99 time-to-first-token |
| Throughput | requests/sec (RPS), tokens/sec (TPS) |
| Reliability | error rate, timeout rate |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

<!-- TODO: requirements.txt still needs to be committed -->

### 2. Configure the server

Edit [configs/base.yaml](configs/base.yaml):
- Set `server.api_base` to your vLLM server address
- Set `model.name` to match the exact model name loaded by vLLM

### 3. Start vLLM (on the GPU machine)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 10000
```

### 4. Verify prompt lengths (optional)

```bash
python -m prompts.prompt_builder
```

## Running the Benchmark

```bash
# dry run - prints the plan without sending any requests
python -m benchmark.runner --dry-run

# full run
python -m benchmark.runner
```

Results are saved to `results/pl{N}_c{C}_i{I}.json` per cell. If the run is interrupted, re-running will skip already-saved cells.

## Analysis

```bash
# compute confidence intervals and flag outliers
python -m analysis.stats --results-dir results --out results/summary.csv

# generate all heatmaps and line plots
python -m analysis.visualize --results-dir results --out-dir figures
```

Figures saved to `figures/`: heatmaps for each metric + line plots vs concurrency per metric.

## Infrastructure Notes

- vLLM server should run on the GPU machine
- load generator (`benchmark/runner.py`) should run on a separate machine to avoid CPU/memory interference
- restart the server between run sessions to prevent KV-cache residuals from contaminating measurements

<!-- TODO: add notes on which GPU/machine we actually used once hardware is confirmed -->
