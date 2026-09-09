"""Thin, rate-limited client for the Groq API (OpenAI-compatible endpoint).

Handles the two things that matter for reliability at free-tier limits:
  1. Proactive throttling against a measured ~8000 tokens/min cap (see
     DECISION_LOG.md) — we target LLM_TOKEN_BUDGET_PER_MIN with headroom.
  2. Structured JSON output via Groq's `json_schema` strict mode, with
     `reasoning_effort: "low"` + `include_reasoning: false` so gpt-oss models
     don't burn their max_tokens budget on hidden reasoning before emitting
     content (this really happens — see scripts/README notes).
"""
from __future__ import annotations

import json
import re
import threading
import time

import requests

from agent import config

_lock = threading.Lock()


class RateLimiter:
    """Sliding-window token budget, shared process-wide."""

    def __init__(self, tokens_per_min: int):
        self.budget = tokens_per_min
        self.window_start = time.time()
        self.used = 0

    def wait_for(self, est_tokens: int) -> None:
        with _lock:
            now = time.time()
            elapsed = now - self.window_start
            if elapsed > 60:
                self.window_start = now
                self.used = 0
                elapsed = 0
            if self.used + est_tokens > self.budget:
                sleep_s = max(60 - elapsed + 0.5, 0)
            else:
                sleep_s = 0
                self.used += est_tokens
        if sleep_s > 0:
            if sleep_s > 3:
                # Visible, flushed, so a long-running batch script's log shows
                # *why* progress paused instead of just going quiet — a silent
                # multi-minute sleep here once looked indistinguishable from a
                # hang (see DECISION_LOG.md).
                print(f"    [rate-limit] sleeping {sleep_s:.1f}s (token budget)", flush=True)
            time.sleep(sleep_s)
            with _lock:
                self.window_start = time.time()
                self.used = est_tokens


_rate_limiter = RateLimiter(config.LLM_TOKEN_BUDGET_PER_MIN)
USAGE_LOG: list[dict] = []


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _safe_json_parse(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from model output: {content[:300]!r}")


class LLMError(RuntimeError):
    pass


def chat_structured(
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    schema_name: str,
    model: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    retries: int = config.LLM_MAX_RETRIES,
) -> dict:
    if not config.has_llm():
        raise LLMError("GROQ_API_KEY is not set — see .env.example")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    }
    est_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt) + max_tokens

    last_err = None
    for attempt in range(retries):
        _rate_limiter.wait_for(est_tokens)
        try:
            resp = requests.post(
                f"{config.GROQ_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60,
            )
        except requests.RequestException as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            print(f"    [network error] {e!r} -- retrying in {wait}s (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = _safe_json_parse(content)
            usage = data.get("usage", {})
            USAGE_LOG.append({"model": model, "schema": schema_name, **usage})
            parsed["_usage"] = usage
            parsed["_model"] = model
            return parsed

        if resp.status_code == 429:
            # Cap the wait: a runaway Retry-After (e.g. a daily-quota reset
            # hours away, as opposed to the usual per-minute throttle) should
            # fail fast into the retry-count limit rather than silently sleep
            # for a very long time on one call.
            retry_after = min(float(resp.headers.get("Retry-After", 2 ** attempt)), 30.0)
            print(f"    [429] retrying in {retry_after:.1f}s (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(retry_after + 0.5)
            continue

        if resp.status_code >= 500:
            wait = min(2 ** attempt, 20)
            print(f"    [{resp.status_code}] server error -- retrying in {wait}s (attempt {attempt + 1}/{retries})", flush=True)
            time.sleep(wait)
            continue

        raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:500]}")

    raise LLMError(f"Groq API: exceeded {retries} retries; last error: {last_err}")


def summarize_usage() -> dict:
    if not USAGE_LOG:
        return {"n_calls": 0, "total_tokens": 0}
    total_prompt = sum(u.get("prompt_tokens", 0) for u in USAGE_LOG)
    total_completion = sum(u.get("completion_tokens", 0) for u in USAGE_LOG)
    return {
        "n_calls": len(USAGE_LOG),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }
