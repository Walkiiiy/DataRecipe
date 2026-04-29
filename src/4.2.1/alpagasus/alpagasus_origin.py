import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from urllib3.util import Timeout as Urllib3Timeout

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SYSTEM_PROMPT = "You are a helpful assistant."

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekAlpagasusScorer:
    """Alpagasus origin scorer (scalar only, no capability-cluster scoring)."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        request_timeout: int = 120,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.request_timeout = int(request_timeout)
        self.connect_timeout = max(0.1, float(connect_timeout))
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
    def generate_prompt(row: Dict[str, Any]) -> str:
        instruction = str(row.get("instruction", ""))
        input_data = str(row.get("input", ""))
        output_data = str(row.get("output", row.get("response", "")))

        # Keep the same scoring prompt logic as original tool_automatic_check.py.
        if input_data != "" or input_data == "Noinput":
            return (
                "Following the format <yes/no>||<explanation why yes or no>. "
                f"Given the following instruction: {instruction} and the following "
                f"input: {input_data}, is the output '{output_data}' correct?"
            )
        return (
            "Following the format <yes/no>||<explanation why yes or no>. "
            f"Given the following instruction: {instruction}, "
            f"is the output '{output_data}' correct?"
        )

    @staticmethod
    def parse_judge_response(response_text: str) -> Tuple[bool, float, str]:
        text = (response_text or "").strip()
        if not text:
            return False, 0.0, ""

        if "||" in text:
            first, second = text.split("||", 1)
        else:
            parts = re.split(r"[,\.]", text, maxsplit=1)
            first = parts[0]
            second = parts[1] if len(parts) > 1 else ""

        first_lower = first.lower()
        is_yes = "yes" in first_lower
        score = 1.0 if is_yes else 0.0
        return is_yes, score, second.strip()

    def _build_http_timeout(self) -> Urllib3Timeout:
        total_timeout = max(1.0, float(self.request_timeout))
        connect_timeout = min(self.connect_timeout, total_timeout)
        return Urllib3Timeout(total=total_timeout, connect=connect_timeout, read=total_timeout)

    def _compute_retry_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        base = max(0.0, self.retry_sleep * attempt)
        jitter = random.uniform(0.0, max(0.1, self.retry_sleep))
        delay = base + jitter
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        return delay

    @staticmethod
    def _short_error_text(text: str, max_len: int = 240) -> str:
        clean = str(text or "").replace("\n", " ").replace("\r", " ").strip()
        if len(clean) <= max_len:
            return clean
        return clean[:max_len].rstrip() + "..."

    def call_deepseek(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                session = self._get_session()
                resp = session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._build_http_timeout(),
                )

                if resp.status_code >= 500 or resp.status_code in (408, 429):
                    status = resp.status_code
                    body = self._short_error_text(resp.text)
                    retry_after = resp.headers.get("Retry-After")
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"DeepSeek temporary error {status} after retries: {body}"
                        )
                    sleep_s = self._compute_retry_delay(attempt, retry_after)
                    logger.warning(
                        "DeepSeek temporary status=%s at attempt=%d/%d, sleeping %.1fs then retry.",
                        status,
                        attempt,
                        self.max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                return str(choices[0].get("message", {}).get("content", "")).strip()

            except requests.Timeout as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"DeepSeek request timeout after retries (timeout={self.request_timeout}s total): {exc}"
                    ) from exc
                sleep_s = self._compute_retry_delay(attempt, retry_after=None)
                logger.warning(
                    "DeepSeek timeout at attempt=%d/%d, sleeping %.1fs then retry.",
                    attempt,
                    self.max_retries,
                    sleep_s,
                )
                time.sleep(sleep_s)
            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"DeepSeek API request failed after retries: {exc}") from exc
                sleep_s = self._compute_retry_delay(attempt, retry_after=None)
                logger.warning(
                    "DeepSeek request error at attempt=%d/%d: %s; sleeping %.1fs then retry.",
                    attempt,
                    self.max_retries,
                    self._short_error_text(str(exc)),
                    sleep_s,
                )
                time.sleep(sleep_s)

        return ""

    def _score_one_sample(
        self,
        sample: Dict[str, Any],
        sample_index: int,
        dry_run: bool,
        temperature: float,
        max_tokens: int,
        include_prompt: bool,
    ) -> Dict[str, Any]:
        prompt = self.generate_prompt(sample)
        raw_response = "yes||dry run" if dry_run else self.call_deepseek(prompt, temperature, max_tokens)

        judge_pass, judge_score, judge_reason = self.parse_judge_response(raw_response)

        item = dict(sample)
        sample_id = self._choose_row_id(sample, sample_index)
        item["id"] = sample_id
        item["judge_model"] = self.model
        item["judge_response"] = raw_response
        item["judge_pass"] = bool(judge_pass)
        item["judge_score"] = float(judge_score)
        item["judge_reason"] = judge_reason
        item["judge_index"] = int(sample_index)
        item["score"] = float(judge_score)
        item["score_type"] = "alpagasus_origin_scalar"
        if include_prompt:
            item["judge_prompt"] = prompt
        return item

    def _build_error_row(
        self,
        sample: Dict[str, Any],
        sample_index: int,
        include_prompt: bool,
        error: Exception,
    ) -> Dict[str, Any]:
        prompt = self.generate_prompt(sample)
        sample_id = self._choose_row_id(sample, sample_index)
        item = dict(sample)
        item["id"] = sample_id
        item["judge_model"] = self.model
        item["judge_response"] = ""
        item["judge_pass"] = False
        item["judge_score"] = 0.0
        item["judge_reason"] = ""
        item["judge_index"] = int(sample_index)
        item["score"] = 0.0
        item["score_type"] = "alpagasus_origin_scalar"
        item["judge_error"] = self._short_error_text(str(error), max_len=500)
        if include_prompt:
            item["judge_prompt"] = prompt
        return item

    def score_dataset(
        self,
        data: List[Dict[str, Any]],
        starting_sample: int,
        max_samples: int,
        concurrency: int,
        dry_run: bool,
        temperature: float,
        max_tokens: int,
        include_prompt: bool,
        fail_fast: bool,
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
        logger.info("Loaded %d rows, scoring rows [%d, %d) -> %d", total, start, end, len(selected))

        if concurrency < 1:
            raise ValueError("--concurrency must be >= 1")

        if concurrency == 1:
            iterator = tqdm(selected, total=len(selected), desc="Scoring samples") if tqdm else selected
            out: List[Dict[str, Any]] = []
            for local_idx, sample in enumerate(iterator):
                global_idx = start + local_idx
                try:
                    out.append(
                        self._score_one_sample(
                            sample=sample,
                            sample_index=global_idx,
                            dry_run=dry_run,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            include_prompt=include_prompt,
                        )
                    )
                except Exception as exc:
                    if fail_fast:
                        raise
                    logger.warning(
                        "Sample idx=%d failed, writing zero-score row and continuing: %s",
                        global_idx,
                        self._short_error_text(str(exc)),
                    )
                    out.append(
                        self._build_error_row(
                            sample=sample,
                            sample_index=global_idx,
                            include_prompt=include_prompt,
                            error=exc,
                        )
                    )
            return out, start, end

        results: List[Optional[Dict[str, Any]]] = [None] * len(selected)

        def worker(local_idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
            global_idx = start + local_idx
            return self._score_one_sample(
                sample=row,
                sample_index=global_idx,
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

            progress = tqdm(total=len(futures), desc="Scoring samples") if tqdm else None
            try:
                for future in as_completed(futures):
                    idx = futures[future]
                    global_idx = start + idx
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        if fail_fast:
                            raise
                        logger.warning(
                            "Sample idx=%d failed, writing zero-score row and continuing: %s",
                            global_idx,
                            self._short_error_text(str(exc)),
                        )
                        results[idx] = self._build_error_row(
                            sample=selected[idx],
                            sample_index=global_idx,
                            include_prompt=include_prompt,
                            error=exc,
                        )
                    if progress:
                        progress.update(1)
            finally:
                if progress:
                    progress.close()

        return [x for x in results if x is not None], start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score Alpaca-style data with DeepSeek API (origin scalar mode). "
            "No capability-cluster scoring is applied."
        )
    )
    parser.add_argument(
        "--data_path",
        "--input-path",
        dest="data_path",
        type=str,
        required=True,
        help="Input dataset path (JSON or JSONL).",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        dest="output_path",
        type=str,
        required=True,
        help="Output path for scored JSON or JSONL.",
    )
    parser.add_argument(
        "--api_key",
        "--api-key",
        dest="api_key",
        type=str,
        default=os.getenv("DEEPSEEK_API_KEY", ""),
        help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY.",
    )
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument(
        "--base_url",
        "--base-url",
        dest="base_url",
        type=str,
        default="https://api.deepseek.com",
        help="DeepSeek base URL.",
    )
    parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature.")
    parser.add_argument("--max_tokens", "--max-tokens", dest="max_tokens", type=int, default=200)
    parser.add_argument(
        "--max_samples",
        "--num-examples",
        dest="max_samples",
        type=int,
        default=-1,
        help="Number of samples to process; -1 means all.",
    )
    parser.add_argument(
        "--starting_sample",
        "--starting-sample",
        dest="starting_sample",
        type=int,
        default=0,
        help="Start index in dataset.",
    )
    parser.add_argument(
        "--max_retries",
        "--max-attempts",
        dest="max_retries",
        type=int,
        default=3,
        help="Retry attempts per API call.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--connect_timeout", type=float, default=10.0, help="HTTP connect timeout in seconds.")
    parser.add_argument(
        "--retry_sleep",
        "--retry-sleep",
        dest="retry_sleep",
        type=float,
        default=2.0,
        help="Base sleep seconds between retries.",
    )
    parser.add_argument(
        "--concurrency",
        "--concurrancy",
        dest="concurrency",
        type=int,
        default=1,
        help="Number of concurrent worker threads.",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true", dest="dry_run")
    parser.add_argument(
        "--fail_fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, stop immediately when any sample fails. Default is continue with zero-score row.",
    )
    parser.add_argument(
        "--include_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include the full judge prompt in outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.api_key:
        raise ValueError("Missing API key. Use --api_key or set DEEPSEEK_API_KEY.")

    scorer = DeepSeekAlpagasusScorer(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        request_timeout=args.timeout,
        connect_timeout=args.connect_timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )

    data, input_format = scorer._load_json_or_jsonl(args.data_path)
    logger.info("Loaded %d samples from %s", len(data), args.data_path)
    if args.max_samples >= 0:
        logger.info("Scoring first %d samples (--max_samples)", min(args.max_samples, len(data)))

    scored_data, start, end = scorer.score_dataset(
        data=data,
        starting_sample=args.starting_sample,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
        dry_run=bool(args.dry_run),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_prompt=bool(args.include_prompt),
        fail_fast=bool(args.fail_fast),
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_is_jsonl = input_format == "jsonl" or args.output_path.lower().endswith(".jsonl")
    with open(args.output_path, "w", encoding="utf-8") as f:
        if output_is_jsonl:
            for item in scored_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(scored_data, f, ensure_ascii=False, indent=2)

    logger.info("Saved scored rows [%d, %d) to %s", start, end, args.output_path)


if __name__ == "__main__":
    main()
