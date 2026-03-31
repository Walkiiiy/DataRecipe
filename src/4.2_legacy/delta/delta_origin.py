import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekDeltaScorer:
    """DEITA delta scorer with DeepSeek API backend.

    Adapted for OpenAI-compatible APIs by requesting strict numeric outputs (1-6)
    and parsing them directly. Falls back to logprob-based expectation when needed.
    """

    complexity_template = (
        "You are a strict grader.\n"
        "Evaluate the complexity of the following user query on a 1-6 scale.\n"
        "Return ONLY one digit: 1, 2, 3, 4, 5, or 6.\n"
        "No words, no explanation, no markdown, no punctuation.\n"
        "##Query: {instruction}\n"
        "##Complexity:"
    )

    quality_template = (
        "You are a strict grader.\n"
        "Evaluate the response quality for the given question on a 1-6 scale.\n"
        "Return ONLY one digit: 1, 2, 3, 4, 5, or 6.\n"
        "No words, no explanation, no markdown, no punctuation.\n"
        "#Question#:\n"
        "{instruction}\n"
        "#Response#:\n"
        "{output}\n"
        "##Quality:"
    )

    score_template = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        request_timeout: int = 60,
        max_retries: int = 5,
        retry_backoff: float = 1.5,
    ) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required. Set --api_key or DEEPSEEK_API_KEY.")

        self.api_key = api_key
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

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
    def _softmax(values: List[float]) -> List[float]:
        max_logit = max(values)
        shifted = [v - max_logit for v in values]
        exp_logits = [math.exp(v) for v in shifted]
        exp_sum = sum(exp_logits)
        return [v / exp_sum for v in exp_logits]

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
    def _merge_instruction_and_input(instruction: str, input_text: str) -> str:
        instruction = (instruction or "").strip()
        input_text = (input_text or "").strip()
        if instruction and input_text:
            return f"{instruction}\n{input_text}"
        return instruction or input_text

    @staticmethod
    def _extract_turns(sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Support both ShareGPT and train.jsonl-style samples."""
        if "conversations" in sample and isinstance(sample.get("conversations"), list):
            conversations = sample.get("conversations", [])
            turns: List[Dict[str, str]] = []

            for idx in range(len(conversations)):
                if idx % 2 != 0:
                    continue

                instruction = conversations[idx].get("value", "")
                if idx != len(conversations) - 1:
                    response = conversations[idx + 1].get("value", "")
                else:
                    response = ""

                turns.append({"instruction": instruction, "response": response})

            return turns

        merged_instruction = DeepSeekDeltaScorer._merge_instruction_and_input(
            sample.get("instruction", ""),
            sample.get("input", ""),
        )
        return [{"instruction": merged_instruction, "response": sample.get("output", "")}]

    @staticmethod
    def _extract_top_logprobs(choice_logprobs: Any) -> Optional[List[Dict[str, float]]]:
        """Support common OpenAI-compatible logprobs response shapes."""
        if not choice_logprobs:
            return None

        content = None
        if isinstance(choice_logprobs, dict):
            content = choice_logprobs.get("content")
        else:
            content = getattr(choice_logprobs, "content", None)

        if not content:
            return None

        first_token_info = content[0]
        if isinstance(first_token_info, dict):
            top_logprobs = first_token_info.get("top_logprobs")
        else:
            top_logprobs = getattr(first_token_info, "top_logprobs", None)

        if not top_logprobs:
            return None

        normalized: List[Dict[str, float]] = []

        # Some providers may return dict[token] = logprob instead of list entries.
        if isinstance(top_logprobs, dict):
            for token, logprob in top_logprobs.items():
                if token is None or logprob is None:
                    continue
                normalized.append({"token": str(token), "logprob": float(logprob)})
            return normalized if normalized else None

        for item in top_logprobs:
            if isinstance(item, dict):
                token = item.get("token")
                logprob = item.get("logprob")
            else:
                token = getattr(item, "token", None)
                logprob = getattr(item, "logprob", None)

            if token is None or logprob is None:
                continue

            normalized.append({"token": str(token), "logprob": float(logprob)})

        return normalized if normalized else None

    @staticmethod
    def _score_from_top_logprobs(top_logprobs: List[Dict[str, float]]) -> float:
        """Replicate Scorer.infer_score with token IDs replaced by digit tokens."""
        digit_to_logprob: Dict[str, Optional[float]] = {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None,
            "6": None,
        }

        for item in top_logprobs:
            token = item["token"].strip()
            if token in digit_to_logprob and digit_to_logprob[token] is None:
                digit_to_logprob[token] = item["logprob"]

        # Keep DEITA scorer fallback behavior when target token not available.
        if any(v is None for v in digit_to_logprob.values()):
            return 3.0

        logits = [digit_to_logprob[str(i)] for i in range(1, 7)]
        probs = DeepSeekDeltaScorer._softmax(logits)
        score = float(sum(p * s for p, s in zip(probs, DeepSeekDeltaScorer.score_template)))
        return score

    @staticmethod
    def _extract_digit_score(text: str) -> Optional[float]:
        """Extract a single-digit score from model text output."""
        if not text:
            return None

        cleaned = text.strip()
        if not cleaned:
            return None

        # Fast path: exact one-digit output.
        if cleaned in {"1", "2", "3", "4", "5", "6"}:
            return float(cleaned)

        # Allow wrappers like "**4**" or "Score: 4".
        match = re.search(r"(?<!\d)([1-6])(?!\d)", cleaned)
        if match:
            return float(match.group(1))

        return None

    def _call_deepseek(self, user_input: str) -> float:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                session = self._get_session()
                resp = session.post(
                    # Each worker thread uses its own session.
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(
                        f"DeepSeek temporary error {resp.status_code}: {resp.text}",
                        response=resp,
                    )

                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    logger.warning("Empty choices from DeepSeek, fallback score=3.0")
                    return 3.0

                message_content = choices[0].get("message", {}).get("content", "")
                parsed_score = self._extract_digit_score(str(message_content))
                if parsed_score is not None:
                    return parsed_score

                top_logprobs = self._extract_top_logprobs(choices[0].get("logprobs"))
                if not top_logprobs:
                    logger.warning("Missing logprobs from DeepSeek, fallback score=3.0")
                    return 3.0

                return self._score_from_top_logprobs(top_logprobs)

            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == self.max_retries:
                    logger.warning("DeepSeek request failed after retries, fallback score=3.0: %s", exc)
                    return 3.0

                sleep_seconds = self.retry_backoff ** (attempt - 1)
                time.sleep(sleep_seconds)

        return 3.0

    def _score_one_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        turns = self._extract_turns(sample)
        delta_scores: List[float] = []

        for turn in turns:
            complexity_score = self.infer_complexity(turn["instruction"])
            quality_score = self.infer_quality(turn["instruction"], turn["response"])

            complexity_score = float(complexity_score)
            quality_score = float(quality_score)
            delta_score = float(complexity_score * quality_score)
            delta_scores.append(delta_score)

        sample["score"] = delta_scores
        return sample

    def score_dataset(
        self,
        data: List[Dict[str, Any]],
        concurrency: int = 1,
        max_samples: int = -1,
    ) -> List[Dict[str, Any]]:
        """Compute DEITA delta score per sample.

        delta_score = sum_i (complexity_scores[i] * quality_scores[i])
        """
        if max_samples is not None and max_samples >= 0:
            data = data[:max_samples]

        if concurrency < 1:
            raise ValueError("--concurrency must be >= 1")

        # Keep the simple single-thread path for deterministic debugging.
        if concurrency == 1:
            iterator = tqdm(data, total=len(data), desc="Scoring samples") if tqdm else data
            for idx, sample in enumerate(iterator):
                data[idx] = self._score_one_sample(sample)
            return data

        results: List[Optional[Dict[str, Any]]] = [None] * len(data)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self._score_one_sample, sample): idx
                for idx, sample in enumerate(data)
            }

            progress = tqdm(total=len(futures), desc="Scoring samples") if tqdm else None
            try:
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    if progress:
                        progress.update(1)
            finally:
                if progress:
                    progress.close()

        return [item for item in results if item is not None]

    def infer_complexity(self, input_text: str) -> float:
        user_input = self.complexity_template.format(instruction=input_text)
        return self._call_deepseek(user_input)

    def infer_quality(self, input_text: str, resp_text: str) -> float:
        user_input = self.quality_template.format(instruction=input_text, output=resp_text)
        return self._call_deepseek(user_input)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DEITA delta scores with DeepSeek API.")
    parser.add_argument("--data_path", type=str, required=True, help="Input dataset path (JSON or JSONL).")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for scored JSON or JSONL.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("DEEPSEEK_API_KEY"),
        help="DeepSeek API key. Defaults to env DEEPSEEK_API_KEY.",
    )
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name.")
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com",
        help="DeepSeek base url, e.g. https://api.deepseek.com",
    )
    parser.add_argument("--request_timeout", type=int, default=60, help="HTTP request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=5, help="Retry attempts per request.")
    parser.add_argument("--retry_backoff", type=float, default=1.5, help="Exponential retry backoff base.")
    parser.add_argument(
        "--concurrency",
        "--concurrancy",
        dest="concurrency",
        type=int,
        default=1,
        help="Number of concurrent worker threads.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Only score first N samples. Use -1 to score all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scorer = DeepSeekDeltaScorer(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    data, input_format = scorer._load_json_or_jsonl(args.data_path)
    logger.info("Loaded %d samples from %s", len(data), args.data_path)
    if args.max_samples >= 0:
        logger.info("Scoring first %d samples (--max_samples).", min(args.max_samples, len(data)))
    logger.info("Using concurrency=%d", args.concurrency)

    scored_data = scorer.score_dataset(
        data,
        concurrency=args.concurrency,
        max_samples=args.max_samples,
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

    logger.info("Saved scored dataset to %s", args.output_path)


if __name__ == "__main__":
    main()
