"""
Build synthetic prompts to exact token counts for the benchmark

Uses the HuggingFace tokenizer for the target model so the counts are precise.
Falls back to a rough character-based estimate if the tokenizer isn't available.
"""

import sys
from typing import Optional

from prompts.templates import BASE_INSTRUCTION, QUESTION_SUFFIX, PADDING_PASSAGE_LONG


def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=".tokenizer_cache")
    except Exception:
        return None


def _build_with_tokenizer(target: int, tokenizer) -> str:
    base = BASE_INSTRUCTION + QUESTION_SUFFIX
    base_len = len(tokenizer.encode(base))

    # edge case: target is smaller than the fixed instruction frame
    if target <= base_len:
        tokens = tokenizer.encode(base)[:target]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    # Binary search over how many characters of padding to include
    # We're searching character indices into PADDING_PASSAGE_LONG, not token counts directly
    lo, hi = 0, len(PADDING_PASSAGE_LONG)
    best = base

    for _ in range(30):
        mid = (lo + hi) // 2
        candidate = BASE_INSTRUCTION + PADDING_PASSAGE_LONG[:mid] + QUESTION_SUFFIX
        n = len(tokenizer.encode(candidate))

        if n == target:
            return candidate
        elif n < target:
            lo = mid
            best = candidate
        else:
            hi = mid

        if hi - lo <= 1:
            break

    # Binary search gets us close but char boundaries don't map 1:1 to tokens, so nudge forward one char at a time until we hit or exceed the target
    for extra in range(1, 200):
        candidate = BASE_INSTRUCTION + PADDING_PASSAGE_LONG[:mid + extra] + QUESTION_SUFFIX
        if len(tokenizer.encode(candidate)) >= target:
            tokens = tokenizer.encode(candidate)[:target]
            return tokenizer.decode(tokens, skip_special_tokens=True)

    # fallback: just truncate whatever best we found
    tokens = tokenizer.encode(best)[:target]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _build_heuristic(target: int) -> str:
    # ~4 chars per token is a reasonable estimate for English text
    base = BASE_INSTRUCTION + QUESTION_SUFFIX
    needed = max(0, target * 4 - len(base))
    return BASE_INSTRUCTION + PADDING_PASSAGE_LONG[:needed] + QUESTION_SUFFIX


def build_prompt(target: int, model_name: str, tokenizer=None, cache: Optional[dict] = None) -> str:
    if cache is not None and target in cache:
        return cache[target]

    if tokenizer is None:
        raise RuntimeError(
            f'Tokenizer not available for model "{model_name}", cannot build prompt with exact token count. '
        )

    prompt = _build_with_tokenizer(target, tokenizer)

    if cache is not None:
        cache[target] = prompt
    return prompt


def build_prompt_cache(token_lengths: list[int], model_name: str) -> dict[int, str]:
    """Build and return prompts for all target lengths. Call once before the experiment."""
    tokenizer = _load_tokenizer(model_name)
    if tokenizer is None:
        raise RuntimeError(
        f"Tokenizer could not be loaded for model '{model_name}'. A working tokenizer is required for exact token-count control."
    )

    cache: dict[int, str] = {}
    for length in token_lengths:
        prompt = build_prompt(length, model_name, tokenizer=tokenizer, cache=cache)
        actual = len(tokenizer.encode(prompt)) if tokenizer else "~heuristic"
        print(f"  target={length:5d}  actual={actual}")
    return cache


if __name__ == "__main__":
    import yaml, pathlib

    cfg = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "configs/base.yaml"))
    grid = yaml.safe_load(open(pathlib.Path(__file__).parents[1] / "configs/experiment_grid.yaml"))

    print("Building prompt cache...")
    cache = build_prompt_cache(grid["prompt_lengths"], cfg["model"]["name"])
    for length, prompt in cache.items():
        print(f"\n--- {length} tokens ---")
        print(prompt[:200], "...")
