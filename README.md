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

results/                 # raw JSON outputs + aggregated CSV summaries
figures/                 # generated heatmaps and plots
docs/                    # docs
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