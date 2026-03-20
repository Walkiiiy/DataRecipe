"""Stage 1: Data loading and CDT capability distillation.

This script:
1) Loads samples from a JSONL dataset.
2) Calls an OpenAI-compatible LLM API asynchronously (e.g., DeepSeek).
3) Produces one dense CDT description per sample (~30-50 English words).
4) Writes output JSONL asynchronously, one line per processed sample.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import aiofiles
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = (
    "You are a capability distillation engine for instruction data.\n"
    "Given one sample with instruction, input, and output, generate exactly one English paragraph "
    "named CDT_description (about 30-50 words).\n"
    "The paragraph must be highly abstract and de-entityized (remove concrete names, places, products, IDs).\n"
    "It must strictly cover four explicit dimensions:\n"
    "1) Cognition: thinking mode (reasoning, extraction, planning, creativity, etc.)\n"
    "2) Domain: knowledge area (computer science, physics, medicine, law, daily conversation, etc.)\n"
    "3) Task: operation objective (debugging, summarization, translation, classification, etc.)\n"
    "4) Topic: the concrete content theme being handled by the task in abstract form\n"
    "Output rules:\n"
    "- Return plain text only, no JSON, no bullets, no markdown.\n"
    "- Keep density high and avoid filler words.\n"
    "- Keep all four dimensions explicit in one coherent paragraph.\n"
    "- Task must mention what is being processed and in which domain.\n"
    "- For summarization tasks, explicitly state what topic is summarized and its domain."
)

DEFAULT_INPUT = Path("data/alpaca-gpt4-data-en/train.jsonl")
DEFAULT_OUTPUT = Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl")
DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass
class Config:
    input_path: Path
    output_path: Path
    model: str
    api_key: str
    base_url: str
    max_samples: int | None
    concurrency: int
    timeout: int
    max_retries: int
    retry_base_delay: float
    max_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: CDT capability distillation")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--api-key", type=str, default=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Only process first N lines. Use -1 to process the entire file.",
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    return parser.parse_args()


def sanitize_description(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return (
            "Cognition: structured instruction following with reasoning and information transformation. "
            "Domain: broad open-domain knowledge abstraction. "
            "Task: produce a concise response aligned to requested constraints and output intent, with explicit content focus. "
            "Topic: the central content theme of the sample expressed in abstract terms."
        )
    return cleaned


def build_user_prompt(sample: dict[str, Any]) -> str:
    instruction = str(sample.get("instruction", "")).strip()
    input_text = str(sample.get("input", "")).strip()
    output_text = str(sample.get("output", "")).strip()
    return (
        "Read the following single instruction-tuning sample and distill capability.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{input_text}\n\n"
        f"Output:\n{output_text}\n\n"
        "Return only one English paragraph CDT_description (30-50 words), highly abstract, de-entityized, "
        "and explicitly covering Cognition, Domain, Task, and Topic. "
        "Task must state what content is being handled and in which domain. "
        "If the task is summarization, state the summarized topic and domain explicitly."
    )


async def load_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        async for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


async def request_cdt(
    session: aiohttp.ClientSession,
    cfg: Config,
    sample: dict[str, Any],
) -> str:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(sample)},
        ],
        "temperature": 0.0,
        "max_tokens": cfg.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=cfg.timeout)

    for attempt in range(1, cfg.max_retries + 1):
        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                body = await resp.text()
                if resp.status in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"retryable HTTP {resp.status}: {body[:300]}")
                if resp.status >= 400:
                    raise RuntimeError(f"non-retryable HTTP {resp.status}: {body[:300]}")

                data = json.loads(body)
                content = data["choices"][0]["message"]["content"]
                return sanitize_description(str(content))
        except Exception:
            if attempt == cfg.max_retries:
                break
            backoff = cfg.retry_base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.25 * backoff)
            await asyncio.sleep(backoff + jitter)

    return (
        "Cognition: generic instruction understanding with basic transformation and response planning. "
        "Domain: mixed open-domain context without fixed specialized grounding. "
        "Task: generate a concise answer that matches requested format and communicative goal, with explicit content focus. "
        "Topic: the underlying content theme inferred from the sample context."
    )


async def process_one(
    idx: int,
    row: dict[str, Any],
    session: aiohttp.ClientSession,
    cfg: Config,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        cdt_text = await request_cdt(session=session, cfg=cfg, sample=row)
    return {
        "id": idx,
        "instruction": row.get("instruction", ""),
        "input": row.get("input", ""),
        "output": row.get("output", ""),
        "CDT_description": cdt_text,
    }


async def run(cfg: Config) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    rows = await load_jsonl(cfg.input_path, cfg.max_samples)
    semaphore = asyncio.Semaphore(cfg.concurrency)
    connector = aiohttp.TCPConnector(limit=max(100, cfg.concurrency * 2))

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(process_one(i, row, session, cfg, semaphore))
            for i, row in enumerate(rows)
        ]

        async with aiofiles.open(cfg.output_path, "w", encoding="utf-8") as writer:
            for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Stage1 CDT"):
                item = await fut
                await writer.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    max_samples = None if args.max_samples == -1 else max(0, args.max_samples)
    cfg = Config(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_samples=max_samples,
        concurrency=max(1, args.concurrency),
        timeout=max(1, args.timeout),
        max_retries=max(1, args.max_retries),
        retry_base_delay=max(0.1, args.retry_base_delay),
        max_tokens=max(32, args.max_tokens),
    )
    if not cfg.api_key:
        raise ValueError("Missing API key. Set --api-key or DEEPSEEK_API_KEY/OPENAI_API_KEY.")
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
