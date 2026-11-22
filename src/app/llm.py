"""
One function, complete(), over two providers. Messages use
a simple shared format:
{"role": "system" | "user" | "assistant", "content": "..."}.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import boto3
import httpx


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    model: str


def to_bedrock(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Bedrock wants system prompts separately and content as a list of blocks.
    """
    system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
    convo = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    return system, convo


def _ollama(messages: list[dict], model: str, temperature: float) -> Response:
    start = time.perf_counter()
    r = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    return Response(
        d["message"]["content"],
        d.get("prompt_eval_count", 0),
        d.get("eval_count", 0),
        (time.perf_counter() - start) * 1000,
        "ollama",
        model,
    )


@lru_cache(maxsize=1)
def _bedrock_client():
    return boto3.client("bedrock-runtime")  # Region comes from my AWS profile


def _bedrock(messages: list[dict], model: str, temperature: float) -> Response:
    system, convo = to_bedrock(messages)
    kwargs = {
        "modelId": model,
        "messages": convo,
        "inferenceConfig": {"maxTokens": 1024, "temperature": temperature},
    }
    if system:
        kwargs["system"] = system
    start = time.perf_counter()
    r = _bedrock_client().converse(**kwargs)
    return Response(
        r["output"]["message"]["content"][0]["text"],
        r["usage"]["inputTokens"],
        r["usage"]["outputTokens"],
        (time.perf_counter() - start) * 1000,
        "bedrock",
        model,
    )


DEFAULT_MODELS = {"ollama": "qwen3:4b-instruct-2507-q4_K_M", "bedrock": "eu.amazon.nova-micro-v1:0"}


def complete(
    messages: list[dict],
    provider: str = "ollama",
    model: str | None = None,
    temperature: float = 0.2,
) -> Response:
    model = model or DEFAULT_MODELS[provider]
    if provider == "ollama":
        return _ollama(messages, model, temperature)
    if provider == "bedrock":
        return _bedrock(messages, model, temperature)
    raise ValueError(f"unknown provider: {provider}")
