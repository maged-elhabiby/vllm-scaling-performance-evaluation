# vLLM Scaling Benchmark

Performance evaluation of [vLLM](https://github.com/vllm-project/vllm) under variable prompt length and request concurrency.

This project measures latency, throughput, and reliability across a full prompt length x concurrency grid to characterize how vLLM inference performance changes under increasing workload pressure. The benchmark was designed for the SENG 533 final project and evaluates the interaction between prompt length, concurrency, KV-cache pressure, and GPU resource contention.

## Project Structure

```text
configs/
  base.yaml              # server URL, model, generation params, benchmark settings
  experiment_grid.yaml   # prompt lengths, concurrency levels, run options

prompts/
  templates.py           # padding text used to fill prompts to target length
  prompt_builder.py      # builds prompts to exact token counts via HF tokenizer

benchmark/
  metrics.py             # RequestResult dataclass + aggregation helpers
  load_generator.py      # async HTTP load generator with streaming + TTFT measurement
  runner.py              # experiment orchestrator with full factorial + randomized order

analysis/
  stats.py               # confidence intervals, outlier detection, CSV aggregation
  visualize.py           # heatmaps, line plots, error-bar plots

results/                 # raw JSON outputs, summary.csv, heatmaps, line plots (*.png)
```

## Experimental Design

The benchmark uses a full factorial experiment over prompt length and request concurrency.

| Factor | Levels |
|---|---|
| Prompt length (tokens) | 128, 512, 1024, 2048, 4096, 8192 |
| Concurrency | 1, 2, 4, 8, 16, 32 |
| Iterations per cell | 3 |
| Fixed output tokens | 128 |
| Decoding strategy | Greedy decoding |
| Temperature | 0.0 |
| Run order | Randomized |
| Warmup | 10 requests before each measurement cell |

Total runs:

```text
6 prompt lengths x 6 concurrency levels x 3 iterations = 108 runs
```

## Metrics

| Category | Metric |
|---|---|
| Latency | mean, P50, P95, P99 end-to-end latency |
| TTFT | mean, P95, P99 time-to-first-token |
| Throughput | requests/sec (RPS), tokens/sec (TPS) |
| Reliability | error rate, timeout rate |

## Infrastructure

The benchmark was run using a two-instance AWS setup:

| Component | Configuration |
|---|---|
| vLLM server | AWS EC2 g5.xlarge |
| GPU | NVIDIA A10G |
| GPU memory | 24 GB VRAM |
| Workload generator | Separate CPU-based EC2 instance |
| Network | Private IP communication inside shared AWS VPC |
| Model | Qwen/Qwen2-7B-Instruct |
| Precision | bfloat16 |
| Max model length | 10,000 tokens |

The vLLM server was hosted on the GPU-backed EC2 instance, while the load generator was run on a separate instance to avoid CPU, memory, and network interference with inference execution.

## Key Findings

The benchmark identified three operating regions:

1. **Stable region:** concurrency 1-4
   - 0% error rate across all prompt lengths
   - near-linear throughput scaling
   - tightly bounded latency and tail latency

2. **Transition region:** concurrency 8
   - failures begin to appear
   - system approaches its practical operating limit
   - TTFT begins showing stronger pressure effects

3. **Unstable region:** concurrency 16-32
   - high error rates in several configurations
   - reliability breakdown becomes the dominant failure mode
   - latency metrics become less meaningful when most requests fail

A major conclusion is that the practical limit of the deployment is defined more by reliability breakdown than by gradual latency growth. TTFT was the earliest observable pressure indicator, while P95 and P99 latency provided limited warning before failure.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the server

Edit [`configs/base.yaml`](configs/base.yaml):

```yaml
server:
  api_base: "http://<VLLM_SERVER_PRIVATE_IP>:8000/v1"

model:
  name: "Qwen/Qwen2-7B-Instruct"
```

Make sure the model name matches the exact model loaded by the vLLM server.

### 3. Start vLLM on the GPU machine

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 10000
```

### 4. Verify prompt lengths

```bash
python -m prompts.prompt_builder
```

This step confirms that the prompt builder can generate prompts at the target token lengths using the Hugging Face tokenizer.

## Running the Benchmark

```bash
# Dry run: prints the planned experiment without sending requests
python -m benchmark.runner --dry-run

# Full benchmark run
python -m benchmark.runner
```

Results are saved as individual JSON files:

```text
results/pl{PROMPT_LENGTH}_c{CONCURRENCY}_i{ITERATION}.json
```

If the benchmark is interrupted, re-running the command will skip already completed result files.

## Analysis

```bash
# Compute confidence intervals, aggregate results, and flag outliers
python -m analysis.stats --results-dir results --out results/summary.csv

# Generate heatmaps and line plots
python -m analysis.visualize --results-dir results --out-dir results
```

Generated figures are saved into `results/` alongside the raw JSON outputs. This includes heatmaps for latency, TTFT, RPS, TPS, and error rate, along with line plots showing how each metric changes with concurrency.

## Reliability and Statistical Notes

Each experimental cell is repeated three times. Because the sample size is small, confidence intervals are computed using the t-distribution.

Cells where most runs fail should not be interpreted as reliable average performance values. In those cases, the error rate is the primary result, and latency or throughput values are treated as descriptive only.

## Recommended Run Practices

- Run the vLLM server on the GPU machine.
- Run the benchmark load generator on a separate machine.
- Use the private IP address between AWS instances to reduce network variability.
- Restart the vLLM server between large run sessions to reduce KV-cache residual effects.
- Keep model, precision, output length, and decoding settings fixed across all runs.
- Use randomized run order to reduce bias from thermal effects, resource drift, or run ordering.
- Save raw JSON results for reproducibility and rerun support.

## Repository Outputs

After a full benchmark, the repository should contain:

```text
results/
  summary.csv     # aggregated metrics across all runs
  heatmap_*.png   # one per metric
  line_*.png      # metric vs. concurrency / prompt length
```

## Future Work

Future work could repeat this benchmark across different models, inference platforms, and GPU configurations. Testing smaller models, larger models, and different instruction-tuned architectures would help determine how model size affects the stable operating boundary.

The benchmark could also be extended to compare vLLM with TensorRT-LLM, Hugging Face TGI, and llama.cpp. Additional GPU configurations, including T4, L4, A100, H100, and multi-GPU systems, would help determine how memory capacity and compute performance shift the failure boundary.

Further extensions could vary output length, decoding strategy, and workload pattern to better represent real-world LLM traffic.