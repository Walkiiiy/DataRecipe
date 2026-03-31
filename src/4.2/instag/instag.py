#!/usr/bin/env python3
"""InSTAG scoring script.

This script only performs tag scoring (labeling) and writes intermediate
results to a score-aligned JSON/JSONL file.
Sampling is intentionally decoupled into `instag_sample.py`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SYSTEM_PROMPT = "You are a helpful JSON-outputting tagging assistant."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InStagTagger:
    """Tag scorer using DeepSeek chat completion API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        request_timeout: int = 120,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.request_timeout = int(request_timeout)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.endpoint = self._normalize_endpoint(base_url)
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    @staticmethod
    def _normalize_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @staticmethod
    def _choose_row_id(row: Dict[str, Any], fallback_idx: int) -> Any:
        for key in ("id", "data_id", "uid", "idx", "index"):
            if key in row and row[key] is not None:
                return row[key]
        return fallback_idx

    @staticmethod
    def _load_json_or_jsonl(path: str) -> Tuple[List[Dict[str, Any]], str]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return [], "jsonl"

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data, "json"
            raise ValueError("Input JSON must be a list of samples.")
        except json.JSONDecodeError:
            lines = [line for line in text.splitlines() if line.strip()]
            return [json.loads(line) for line in lines], "jsonl"

    @staticmethod
    def _compose_text(row: Dict[str, Any], text_field: str) -> str:
        direct = str(row.get(text_field, "") or "").strip()
        if direct:
            return direct

        instruction = str(row.get("instruction", "") or "").strip()
        input_text = str(row.get("input", "") or "").strip()
        output_text = str(row.get("output", row.get("response", "")) or "").strip()
        parts = [p for p in [instruction, input_text, output_text] if p]
        return "\n".join(parts)

    @staticmethod
    def _build_prompt(instruction_text: str) -> str:
        # Keep the original InSTAG prompt design.
        return f"""You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant.
Below is an instruction:

[begin]
{instruction_text}
[end]

Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction.
Your answer should be a list including titles of tags and a brief explanation of each tag.
Your response have to strictly follow this JSON format: [{{"tag": str, "explanation": str}}]. Please response in English."""

    @staticmethod
    def _extract_json_array(raw_text: str) -> List[Dict[str, Any]]:
        text = str(raw_text or "").strip()
        if not text:
            return []

        def _coerce(obj: Any) -> List[Dict[str, Any]]:
            if isinstance(obj, list):
                out: List[Dict[str, Any]] = []
                for it in obj:
                    if isinstance(it, dict):
                        out.append(it)
                    elif isinstance(it, str):
                        out.append({"tag": it, "explanation": ""})
                return out
            if isinstance(obj, dict):
                for _, value in obj.items():
                    if isinstance(value, list):
                        return _coerce(value)
            return []

        try:
            parsed = json.loads(text)
            return _coerce(parsed)
        except json.JSONDecodeError:
            pass

        match_obj = re.search(r"\{[\s\S]*\}", text)
        if match_obj:
            try:
                parsed = json.loads(match_obj.group(0))
                arr = _coerce(parsed)
                if arr:
                    return arr
            except json.JSONDecodeError:
                pass

        match_arr = re.search(r"\[[\s\S]*\]", text)
        if match_arr:
            try:
                parsed = json.loads(match_arr.group(0))
                arr = _coerce(parsed)
                if arr:
                    return arr
            except json.JSONDecodeError:
                pass

        return []

    @staticmethod
    def _normalize_tags(tag_items: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
        tags: List[str] = []
        tag_details: List[Dict[str, str]] = []
        seen = set()

        for item in tag_items:
            tag = str(item.get("tag", "") or "").strip()
            exp = str(item.get("explanation", "") or "").strip()
            if not tag:
                continue

            tag_norm = tag.lower()
            if tag_norm in seen:
                continue

            seen.add(tag_norm)
            tags.append(tag_norm)
            tag_details.append({"tag": tag_norm, "explanation": exp})

        return tags, tag_details

    def _call_deepseek_text(self, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        use_response_format = True
        for attempt in range(1, self.max_retries + 1):
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": False,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}

            try:
                session = self._get_session()
                resp = session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )

                if resp.status_code == 400 and use_response_format and "response_format" in resp.text.lower():
                    use_response_format = False
                    continue

                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(
                        f"DeepSeek temporary error {resp.status_code}: {resp.text}",
                        response=resp,
                    )

                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                return str(choices[0].get("message", {}).get("content", "")).strip()

            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek tag request failed after retries: {exc}") from exc
                time.sleep(self.retry_sleep * attempt)

        return ""

    def _score_one_sample(
        self,
        sample: Dict[str, Any],
        sample_index: int,
        text_field: str,
        dry_run: bool,
        temperature: float,
        max_tokens: int,
        include_prompt: bool,
    ) -> Dict[str, Any]:
        sample_id = self._choose_row_id(sample, sample_index)
        text = self._compose_text(sample, text_field=text_field)
        prompt = self._build_prompt(text)

        if dry_run:
            raw_response = '[{"tag":"dry_run_tag","explanation":"dry run"}]'
        else:
            raw_response = self._call_deepseek_text(prompt, temperature=temperature, max_tokens=max_tokens)

        parsed_items = self._extract_json_array(raw_response)
        tags, tag_details = self._normalize_tags(parsed_items)

        out = {
            "id": sample_id,
            "instag_index": int(sample_index),
            "tags": tags,
            "labels": tags,
            "tags_with_explanations": tag_details,
            "text": text,
            "score_type": "instag_tags",
        }
        if include_prompt:
            out["instag_prompt"] = prompt
            out["instag_raw_response"] = raw_response
        return out

    def score_dataset(
        self,
        data: List[Dict[str, Any]],
        text_field: str,
        starting_sample: int,
        max_samples: int,
        concurrency: int,
        dry_run: bool,
        temperature: float,
        max_tokens: int,
        include_prompt: bool,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        total = len(data)
        start = max(0, int(starting_sample))
        if start >= total:
            raise ValueError(f"starting_sample={start} out of range for dataset size {total}.")

        if max_samples is None or int(max_samples) < 0:
            end = total
        else:
            end = min(start + int(max_samples), total)

        selected = data[start:end]
        logger.info("Loaded %d rows, tagging rows [%d, %d) -> %d", total, start, end, len(selected))

        if concurrency < 1:
            raise ValueError("--concurrency must be >= 1")

        if concurrency == 1:
            iterator = tqdm(selected, total=len(selected), desc="InSTAG tagging") if tqdm else selected
            out: List[Dict[str, Any]] = []
            for local_idx, sample in enumerate(iterator):
                out.append(
                    self._score_one_sample(
                        sample=sample,
                        sample_index=start + local_idx,
                        text_field=text_field,
                        dry_run=dry_run,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        include_prompt=include_prompt,
                    )
                )
            return out, start, end

        results: List[Optional[Dict[str, Any]]] = [None] * len(selected)

        def worker(local_idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
            return self._score_one_sample(
                sample=row,
                sample_index=start + local_idx,
                text_field=text_field,
                dry_run=dry_run,
                temperature=temperature,
                max_tokens=max_tokens,
                include_prompt=include_prompt,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(worker, idx, sample): idx
                for idx, sample in enumerate(selected)
            }

            progress = tqdm(total=len(futures), desc="InSTAG tagging") if tqdm else None
            try:
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    if progress:
                        progress.update(1)
            finally:
                if progress:
                    progress.close()

        return [x for x in results if x is not None], start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InSTAG scoring script (tag generation only).")
    parser.add_argument("--data_path", type=str, required=True, help="Input dataset path (JSON or JSONL).")
    parser.add_argument("--output_path", type=str, required=True, help="Output tag-score path (JSON or JSONL).")
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Preferred field used as instruction text. Falls back to instruction/input/output if empty.",
    )
    parser.add_argument("--api_key", type=str, default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--starting_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--concurrency",
        "--concurrancy",
        dest="concurrency",
        type=int,
        default=1,
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--include_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to persist prompt/raw response for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.api_key:
        raise ValueError("Missing API key. Use --api_key or set DEEPSEEK_API_KEY.")

    tagger = InStagTagger(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        request_timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )

    data, input_format = tagger._load_json_or_jsonl(args.data_path)
    logger.info("Loaded %d samples from %s", len(data), args.data_path)
    if args.max_samples >= 0:
        logger.info("Tagging first %d samples (--max_samples)", min(args.max_samples, len(data)))

    tagged_rows, start, end = tagger.score_dataset(
        data=data,
        text_field=args.text_field,
        starting_sample=args.starting_sample,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
        dry_run=bool(args.dry_run),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_prompt=bool(args.include_prompt),
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_is_jsonl = input_format == "jsonl" or args.output_path.lower().endswith(".jsonl")
    with open(args.output_path, "w", encoding="utf-8") as f:
        if output_is_jsonl:
            for row in tagged_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(tagged_rows, f, ensure_ascii=False, indent=2)

    logger.info("Saved InSTAG tags rows [%d, %d) to %s", start, end, args.output_path)


if __name__ == "__main__":
    main()
