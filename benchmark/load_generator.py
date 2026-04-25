"""
Async load generator for the vLLM OpenAI-compatible /v1/completions endpoint.
Uses streaming so TTFT can be captured per-request.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import aiohttp

from benchmark.metrics import RequestResult


async def _send_request(
    session: aiohttp.ClientSession,
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_sec: float,
    request_id: str,
    prompt_tokens: int,
) -> RequestResult:
    url = f"{api_base}/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,  # must be True to capture TTFT - non-streaming gives us nothing until the end
    }

    start = time.perf_counter()
    ttft: Optional[float] = None
    completion_tokens = 0
    error_msg: Optional[str] = None
    timed_out = False

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                end = time.perf_counter()
                return RequestResult(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    start_time=start,
                    end_time=end,
                    ttft=None,
                    completion_tokens=0,
                    success=False,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            # parse the SSE stream line by line
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                token_text = choices[0].get("text", "")

                # first non-empty token = TTFT
                if token_text and ttft is None:
                    ttft = time.perf_counter() - start

                # vLLM sends a usage block on the last chunk but not always - fall back to counting chunks
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                elif token_text:
                    completion_tokens += 1  # fallback: count chunks, not accurate but close enough

    except asyncio.TimeoutError:
        end = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            start_time=start,
            end_time=end,
            ttft=ttft,
            completion_tokens=completion_tokens,
            success=False,
            timed_out=True,
            error="timeout",
        )
    except aiohttp.ClientError as exc:
        end = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            start_time=start,
            end_time=end,
            ttft=ttft,
            completion_tokens=completion_tokens,
            success=False,
            error=str(exc),
        )

    end = time.perf_counter()
    return RequestResult(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        start_time=start,
        end_time=end,
        ttft=ttft,
        completion_tokens=completion_tokens,
        success=True,
    )


async def run_warmup(
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_sec: float,
    n_warmup: int,
) -> None:
    # sequential warmup - we don't want concurrent warmup requests inflating
    # the KV cache before the measurement window starts
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(n_warmup):
            await _send_request(
                session, api_base, model_name, prompt, max_tokens,
                temperature, top_p, timeout_sec,
                request_id=f"warmup-{i}", prompt_tokens=0,
            )


async def run_load(
    api_base: str,
    model_name: str,
    prompt: str,
    prompt_tokens: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_sec: float,
    concurrency: int,
    test_duration_sec: float,
) -> tuple[list[RequestResult], float]:
    # Closed-loop load model: concurrency is the fixed input, RPS is the measured output.
    # Each worker fires the next request the instant the previous one returns, keeping
    # exactly `concurrency` requests in flight at all times. This differs from an open-loop
    # model (Poisson arrivals at a fixed rate) where RPS is the input and queue depth is
    # the output. Implication: at low latency the server sustains high RPS; as latency
    # rises (larger prompts, heavier concurrency), workers stall waiting for responses and
    # RPS drops naturally — that degradation curve is what the benchmark is designed to capture.
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    stop_event = asyncio.Event()
    request_counter = 0

    connector = aiohttp.TCPConnector(limit=0)

    async def worker():
        nonlocal request_counter
        while not stop_event.is_set():
            async with semaphore:
                # re-check after acquiring - stop_event may have fired while we waited
                if stop_event.is_set():
                    break
                req_id = str(uuid.uuid4())
                request_counter += 1
                result = await _send_request(
                    session, api_base, model_name, prompt,
                    max_tokens, temperature, top_p, timeout_sec,
                    req_id, prompt_tokens,
                )
                results.append(result)

    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

        wall_start = time.perf_counter()
        await asyncio.sleep(test_duration_sec)
        stop_event.set()

        # wait for any in-flight requests to finish before we return
        # this means elapsed will be slightly longer than test_duration_sec
        await asyncio.gather(*workers, return_exceptions=True)
        wall_end = time.perf_counter()

    return results, wall_end - wall_start
