"""Stage 1: Atomic capability profiling for Alpaca instructions.

Features:
- Load `tatsu-lab/alpaca` with HuggingFace `datasets`
- Support quick testing via `--max-samples` (default: 1000)
- Extract rich zero-shot capability tags for each sample
- Asynchronous concurrent batched API calls with retries/backoff
- Progress bar and JSONL persistence

Example:
    export OPENAI_API_KEY="..."
    python src/4.1/stage1_atomic_profile.py \
        --model gpt-4o-mini \
        --max-samples 1000 \
        --output data/alpaca_with_tags.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an expert at instruction-data capability profiling. "
    "Your task is to produce rich, fine-grained capability tags for one instruction sample. "
    "Return MANY tags when appropriate; do not restrict to 3 tags. "
    "The tags must jointly reflect both COMMON abilities and DISTINCTIVE/SPECIAL traits of this sample. "
    "Output format: one comma-separated line of tags only.\n\n"
    "Tagging principles:\n"
    "1) Cover common/general capabilities:\n"
    "   - task family (e.g., QA, Summarization, Translation, Coding, Classification)\n"
    "   - reasoning style (e.g., LogicalReasoning, CausalReasoning, ComparativeReasoning)\n"
    "   - language skills (e.g., Rewriting, ToneControl, Grammar)\n"
    "   - knowledge type (e.g., WorldKnowledge, DomainKnowledge, MathKnowledge)\n"
    "2) Cover distinctive/specific traits:\n"
    "   - domain/topic (e.g., Medicine, Finance, Law, Education, Biology)\n"
    "   - constraints (e.g., LengthConstraint, FormatConstraint, MultiStepConstraint)\n"
    "   - required output form (e.g., JSONOutput, BulletListOutput, CodeSnippetOutput)\n"
    "   - data modality clues (e.g., TableUnderstanding, NumericComputation, DialogueGeneration)\n"
    "   - edge/special challenge (e.g., AmbiguityResolution, ErrorCorrection, SafetySensitive)\n"
    "3) Prefer atomic, reusable tags; avoid full-sentence tags.\n"
    "4) Use English tags in PascalCase or concise CamelCase.\n"
    "5) Remove duplicates and near-duplicates.\n"
    "6) Typically return 8-20 tags when the sample is information-rich; fewer only if truly simple.\n"
    "7) Do not output any explanation, JSON, numbering, or extra commentary."
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class Config:
    dataset_name: str
    split: str
    output: Path
    max_samples: int | None
    model: str
    api_key: str
    base_url: str
    concurrency: int
    request_timeout: int
    max_retries: int
    retry_base_delay: float
    batch_size: int
    max_tokens: int


class OpenAICompatibleClient:
    """Minimal async OpenAI-compatible client over aiohttp."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._timeout = aiohttp.ClientTimeout(total=cfg.request_timeout)

    async def chat_completion(self, session: aiohttp.ClientSession, messages: list[dict[str, str]]) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.cfg.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        async with session.post(url, headers=headers, json=payload, timeout=self._timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")

            try:
                data = json.loads(text)
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Invalid API response: {text[:500]}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Atomic profile tag generation")
    parser.add_argument("--dataset-name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="data/alpaca_with_tags.jsonl")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Limit number of rows for quick testing. Use -1 for full split.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of samples per API call. Increase to reduce call count.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max completion tokens for each API call.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def normalize_tags(raw: str) -> str:
    # Normalize separators and keep unique tags while preserving rich detail.
    parts = [p.strip() for p in re.split(r"[,，;/\\|]+", raw) if p.strip()]
    uniq: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return ", ".join(uniq) if uniq else "General"


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def truncate_text(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def parse_batch_tags(raw: str, expected_ids: list[int]) -> dict[int, str]:
    cleaned = strip_code_fence(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Batch response is not a JSON object.")

    out: dict[int, str] = {}
    for sid in expected_ids:
        val = data.get(str(sid))
        if val is None:
            val = data.get(sid)

        if val is None:
            continue

        if isinstance(val, list):
            merged = ", ".join(str(x).strip() for x in val if str(x).strip())
        else:
            merged = str(val).strip()
        out[sid] = normalize_tags(merged)
    return out


async def extract_tags(
    instruction: str,
    input_text: str,
    output_text: str,
    client: OpenAICompatibleClient,
    session: aiohttp.ClientSession,
    cfg: Config,
) -> str:
    """Extract rich capability tags from one sample with retry/backoff."""
    user_prompt = (
        "Instruction sample:\n"
        f"Instruction: {instruction}\n"
        f"Input: {input_text}\n"
        f"Output: {output_text}\n\n"
        "Return only capability tags as a comma-separated list. "
        "Include both common and distinctive tags; prefer rich coverage."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, cfg.max_retries + 1):
        try:
            raw = await client.chat_completion(session=session, messages=messages)
            return normalize_tags(raw)
        except Exception as exc:  # noqa: BLE001
            if attempt == cfg.max_retries:
                logging.error("Tag extraction failed after retries: %s", exc)
                return "General"

            backoff = cfg.retry_base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.3 * backoff)
            sleep_s = backoff + jitter
            logging.warning(
                "extract_tags retry %s/%s due to error: %s; sleep %.2fs",
                attempt,
                cfg.max_retries,
                exc,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)

    return "General"


async def extract_tags_batch(
    batch_items: list[tuple[int, dict[str, Any]]],
    client: OpenAICompatibleClient,
    session: aiohttp.ClientSession,
    cfg: Config,
) -> dict[int, str]:
    expected_ids = [idx for idx, _ in batch_items]
    lines: list[str] = []
    for idx, row in batch_items:
        instruction = truncate_text(str(row.get("instruction", "")))
        input_text = truncate_text(str(row.get("input", "")))
        output_text = truncate_text(str(row.get("output", "")))
        lines.append(
            f"ID={idx}\n"
            f"Instruction: {instruction}\n"
            f"Input: {input_text}\n"
            f"Output: {output_text}\n"
        )

    user_prompt = (
        "You will label multiple instruction samples.\n"
        "Return STRICT JSON only: an object mapping sample ID(string) -> comma-separated tags string.\n"
        "No markdown, no explanation.\n"
        f"Sample IDs: {expected_ids}\n\n"
        "Samples:\n"
        + "\n".join(lines)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, cfg.max_retries + 1):
        try:
            raw = await client.chat_completion(session=session, messages=messages)
            parsed = parse_batch_tags(raw, expected_ids)
            if not parsed:
                raise ValueError("Parsed empty batch response.")
            return parsed
        except Exception as exc:  # noqa: BLE001
            if attempt == cfg.max_retries:
                logging.error("Batch extraction failed after retries: %s", exc)
                return {}
            backoff = cfg.retry_base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.3 * backoff)
            sleep_s = backoff + jitter
            logging.warning(
                "extract_tags_batch retry %s/%s due to error: %s; sleep %.2fs",
                attempt,
                cfg.max_retries,
                exc,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)

    return {}


def chunk_items(items: list[tuple[int, dict[str, Any]]], batch_size: int) -> list[list[tuple[int, dict[str, Any]]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def process_batch(
    batch_items: list[tuple[int, dict[str, Any]]],
    semaphore: asyncio.Semaphore,
    client: OpenAICompatibleClient,
    session: aiohttp.ClientSession,
    cfg: Config,
) -> list[tuple[int, dict[str, Any]]]:
    async with semaphore:
        batch_tags = await extract_tags_batch(batch_items, client, session, cfg)

        # Fallback to single-sample extraction when some IDs are missing.
        results: list[tuple[int, dict[str, Any]]] = []
        for idx, row in batch_items:
            tags = batch_tags.get(idx)
            if tags is None:
                instruction = str(row.get("instruction", ""))
                input_text = str(row.get("input", ""))
                output_text = str(row.get("output", ""))
                tags = await extract_tags(instruction, input_text, output_text, client, session, cfg)
            out = dict(row)
            out["tags"] = tags
            results.append((idx, out))
        return results


async def run_pipeline(cfg: Config) -> None:
    logging.info("Loading dataset: %s[%s]", cfg.dataset_name, cfg.split)
    ds = load_dataset(cfg.dataset_name, split=cfg.split)

    if cfg.max_samples is not None and cfg.max_samples >= 0:
        sample_n = min(cfg.max_samples, len(ds))
        ds = ds.select(range(sample_n))

    rows = [dict(r) for r in ds]
    total = len(rows)
    logging.info("Total rows to process: %s", total)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    indexed_rows = list(enumerate(rows))
    batches = chunk_items(indexed_rows, cfg.batch_size)
    logging.info("Batch size: %s, API calls (estimated): %s", cfg.batch_size, len(batches))

    semaphore = asyncio.Semaphore(cfg.concurrency)
    client = OpenAICompatibleClient(cfg)

    connector = aiohttp.TCPConnector(limit=max(cfg.concurrency * 2, 100))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(process_batch(batch, semaphore, client, session, cfg))
            for batch in batches
        ]

        results: list[dict[str, Any] | None] = [None] * total
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Tagging(Batches)"):
            batch_result = await fut
            for idx, tagged_row in batch_result:
                results[idx] = tagged_row

    with cfg.output.open("w", encoding="utf-8") as f:
        for row in results:
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info("Saved tagged dataset to: %s", cfg.output.resolve())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    max_samples = None if args.max_samples == -1 else args.max_samples
    cfg = Config(
        dataset_name=args.dataset_name,
        split=args.split,
        output=Path(args.output),
        max_samples=max_samples,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=max(1, args.concurrency),
        request_timeout=max(1, args.request_timeout),
        max_retries=max(1, args.max_retries),
        retry_base_delay=max(0.1, args.retry_base_delay),
        batch_size=max(1, args.batch_size),
        max_tokens=max(64, args.max_tokens),
    )

    if not cfg.api_key:
        raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")

    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
