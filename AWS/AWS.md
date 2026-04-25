# AWS Benchmark Run Guide

This project evaluates vLLM inference performance across prompt length and request concurrency using a two-instance AWS setup.

## AWS Setup

Use two EC2 instances in the same VPC:

1. **GPU server**
   - Instance: `g5.xlarge`
   - GPU: NVIDIA A10G, ~24 GB VRAM
   - AMI: AWS Deep Learning AMI with CUDA/NVIDIA drivers
   - Role: runs the vLLM OpenAI-compatible API server

2. **Load generator**
   - CPU-based EC2 instance
   - Role: runs the benchmark client, collects results, and generates figures

Security groups should allow:
- SSH only from your IP
- Port `8000` on the GPU instance only from the load-generator instance/security group

## Start vLLM on GPU Instance

SSH into the GPU instance:

```bash
cd ~/vllm-scaling-performance-evaluation
source .venv/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen2-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 10000 \
  --enforce-eager