import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RoutedItem:
    row_index: int
    sample_id: Any
    top_k_indices: List[int]
    top_k_scores: List[float]
    top_k_node_ids: List[str]
    top_k_node_names: List[str]
    top_k_node_paths: List[str]


class DeepSeekDeltaScorer:
    """DEITA delta scorer with DeepSeek API backend.

    Origin scalar delta is preserved as `delta_scalar`, and then mapped into an
    m-dim capability vector using routed top-k indices.
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
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return default

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
            # NOTE:
            # Avoid splitlines(): certain unicode separators (e.g. U+0085 NEL)
            # can appear inside JSON string values and would break one JSON row
            # into invalid fragments.
            rows: List[Dict[str, Any]] = []
            for line_no, raw in enumerate(text.split("\n"), start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid JSONL object at {path}:{line_no}: expected dict.")
                rows.append(row)
            return rows, "jsonl"

    @staticmethod
    def _extract_optional_list(row: Dict[str, Any], key: str) -> Optional[List[Any]]:
        value = row.get(key)
        if value is None:
            return None
        if not isinstance(value, list):
            return None
        return value

    @staticmethod
    def _load_routing_jsonl(
        routing_path: str,
        max_top_k: int,
    ) -> Tuple[Dict[str, RoutedItem], int]:
        items: Dict[str, RoutedItem] = {}
        inferred_m = 0

        with open(routing_path, "r", encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)

                sample_id = row.get("id", row_index)
                top_k_indices_raw = row.get("top_k_indices")
                top_k_scores_raw = row.get("top_k_scores")
                if not isinstance(top_k_indices_raw, list) or not isinstance(top_k_scores_raw, list):
                    raise ValueError(
                        f"Invalid routing row at line={row_index + 1}: missing list top_k_indices/top_k_scores"
                    )
                if len(top_k_indices_raw) != len(top_k_scores_raw):
                    raise ValueError(
                        f"Routing row length mismatch at line={row_index + 1}: "
                        f"top_k_indices={len(top_k_indices_raw)} vs top_k_scores={len(top_k_scores_raw)}"
                    )

                limit = len(top_k_indices_raw)
                if max_top_k > 0:
                    limit = min(limit, max_top_k)

                top_k_node_ids_raw = DeepSeekDeltaScorer._extract_optional_list(row, "top_k_node_ids")
                top_k_node_names_raw = DeepSeekDeltaScorer._extract_optional_list(row, "top_k_node_names")
                top_k_node_paths_raw = DeepSeekDeltaScorer._extract_optional_list(row, "top_k_node_paths")

                top_k_indices: List[int] = []
                top_k_scores: List[float] = []
                top_k_node_ids: List[str] = []
                top_k_node_names: List[str] = []
                top_k_node_paths: List[str] = []

                for rank in range(limit):
                    idx = DeepSeekDeltaScorer._to_int(top_k_indices_raw[rank], -1)
                    if idx < 0:
                        continue
                    score = DeepSeekDeltaScorer._to_float(top_k_scores_raw[rank], 0.0)

                    node_id = ""
                    if top_k_node_ids_raw is not None and rank < len(top_k_node_ids_raw):
                        node_id = str(top_k_node_ids_raw[rank] or "").strip()

                    node_name = ""
                    if top_k_node_names_raw is not None and rank < len(top_k_node_names_raw):
                        node_name = str(top_k_node_names_raw[rank] or "").strip()

                    node_path = ""
                    if top_k_node_paths_raw is not None and rank < len(top_k_node_paths_raw):
                        node_path = str(top_k_node_paths_raw[rank] or "").strip()

                    top_k_indices.append(idx)
                    top_k_scores.append(score)
                    top_k_node_ids.append(node_id)
                    top_k_node_names.append(node_name)
                    top_k_node_paths.append(node_path)
                    inferred_m = max(inferred_m, idx + 1)

                if not top_k_indices:
                    logger.warning("Routing row id=%s has no valid candidates and will be ignored.", sample_id)
                    continue

                key = str(sample_id)
                if key in items:
                    logger.warning("Duplicate routing id=%s found; keeping the last row.", key)

                items[key] = RoutedItem(
                    row_index=row_index,
                    sample_id=sample_id,
                    top_k_indices=top_k_indices,
                    top_k_scores=top_k_scores,
                    top_k_node_ids=top_k_node_ids,
                    top_k_node_names=top_k_node_names,
                    top_k_node_paths=top_k_node_paths,
                )

        return items, inferred_m

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

        if cleaned in {"1", "2", "3", "4", "5", "6"}:
            return float(cleaned)

        match = re.search(r"(?<!\d)([1-6])(?!\d)", cleaned)
        if match:
            return float(match.group(1))

        return None

    @staticmethod
    def _aggregate_turn_scores(scores: List[float], mode: str) -> float:
        if not scores:
            return 0.0
        if mode == "sum":
            return float(sum(scores))
        if mode == "mean":
            return float(sum(scores) / len(scores))
        if mode == "max":
            return float(max(scores))
        raise ValueError(f"Unsupported turn aggregation: {mode}")

    @staticmethod
    def _normalize_weights(weights: List[float], mode: str) -> List[float]:
        if not weights:
            return []
        if mode == "uniform":
            w = 1.0 / len(weights)
            return [w for _ in weights]

        clipped = [max(0.0, float(x)) for x in weights]
        total = float(sum(clipped))
        if total <= 0:
            w = 1.0 / len(weights)
            return [w for _ in weights]
        return [x / total for x in clipped]

    @staticmethod
    def _resolve_sample_routing(
        sample: Dict[str, Any],
        fallback_idx: int,
        routing_map: Dict[str, RoutedItem],
    ) -> Tuple[Any, RoutedItem]:
        sample_id = DeepSeekDeltaScorer._choose_row_id(sample, fallback_idx)
        routed = routing_map.get(str(sample_id))
        if routed is None and str(sample_id) != str(fallback_idx):
            routed = routing_map.get(str(fallback_idx))
        if routed is None:
            raise ValueError(
                f"Missing routing row for sample id={sample_id} (fallback_idx={fallback_idx}). "
                "Check --routing_path alignment with --data_path."
            )
        return sample_id, routed

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

    def _score_one_sample(
        self,
        sample: Dict[str, Any],
        sample_id: Any,
        routed_item: RoutedItem,
        m_dimensions: int,
        turn_aggregation: str,
        routing_weight_mode: str,
        store_turn_scores: bool,
        attach_routing_meta: bool,
    ) -> Dict[str, Any]:
        turns = self._extract_turns(sample)
        delta_scores: List[float] = []

        for turn in turns:
            complexity_score = self.infer_complexity(turn["instruction"])
            quality_score = self.infer_quality(turn["instruction"], turn["response"])

            complexity_score = float(complexity_score)
            quality_score = float(quality_score)
            delta_score = float(complexity_score * quality_score)
            delta_scores.append(delta_score)

        delta_scalar = self._aggregate_turn_scores(delta_scores, mode=turn_aggregation)
        mapped_vector = [0.0] * m_dimensions

        weights = self._normalize_weights(routed_item.top_k_scores, mode=routing_weight_mode)
        for idx, weight in zip(routed_item.top_k_indices, weights):
            if 0 <= idx < m_dimensions:
                mapped_vector[idx] += float(delta_scalar * weight)

        sample["id"] = sample_id
        sample["delta_scalar"] = delta_scalar
        sample["mapped_vector"] = mapped_vector
        sample["score"] = mapped_vector
        sample["score_type"] = "delta_origin_mapped_vector"
        sample["score_turn_aggregation"] = turn_aggregation

        if store_turn_scores:
            sample["score_by_turn_scalar"] = delta_scores

        if attach_routing_meta:
            sample["top_k_indices"] = routed_item.top_k_indices
            sample["top_k_scores"] = routed_item.top_k_scores
            sample["top_k_node_ids"] = routed_item.top_k_node_ids
            sample["top_k_node_names"] = routed_item.top_k_node_names
            sample["top_k_node_paths"] = routed_item.top_k_node_paths

        return sample

    def score_dataset(
        self,
        data: List[Dict[str, Any]],
        routing_map: Dict[str, RoutedItem],
        m_dimensions: int,
        turn_aggregation: str,
        routing_weight_mode: str,
        concurrency: int = 1,
        max_samples: int = -1,
        store_turn_scores: bool = False,
        attach_routing_meta: bool = True,
    ) -> List[Dict[str, Any]]:
        if max_samples is not None and max_samples >= 0:
            data = data[:max_samples]

        if concurrency < 1:
            raise ValueError("--concurrency must be >= 1")

        if concurrency == 1:
            iterator = tqdm(data, total=len(data), desc="Scoring samples") if tqdm else data
            for idx, sample in enumerate(iterator):
                sample_id, routed_item = self._resolve_sample_routing(sample, idx, routing_map)
                data[idx] = self._score_one_sample(
                    sample=sample,
                    sample_id=sample_id,
                    routed_item=routed_item,
                    m_dimensions=m_dimensions,
                    turn_aggregation=turn_aggregation,
                    routing_weight_mode=routing_weight_mode,
                    store_turn_scores=store_turn_scores,
                    attach_routing_meta=attach_routing_meta,
                )
            return data

        results: List[Optional[Dict[str, Any]]] = [None] * len(data)

        def worker(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
            sample_id, routed_item = self._resolve_sample_routing(row, index, routing_map)
            return self._score_one_sample(
                sample=row,
                sample_id=sample_id,
                routed_item=routed_item,
                m_dimensions=m_dimensions,
                turn_aggregation=turn_aggregation,
                routing_weight_mode=routing_weight_mode,
                store_turn_scores=store_turn_scores,
                attach_routing_meta=attach_routing_meta,
            )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(worker, idx, sample): idx
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
    parser.add_argument(
        "--routing_path",
        type=str,
        required=True,
        help="Coarse routing jsonl path (e.g. train_coarse_topk5.jsonl).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output path for scored JSON or JSONL.")
    parser.add_argument(
        "--m_dimensions",
        type=int,
        default=None,
        help="Capability dimension m. If omitted, infer from routing top_k_indices.",
    )
    parser.add_argument(
        "--max_top_k",
        type=int,
        default=5,
        help="Use at most top-k routed clusters for mapping; <=0 means all.",
    )
    parser.add_argument(
        "--turn_aggregation",
        type=str,
        choices=["sum", "mean", "max"],
        default="sum",
        help="How to aggregate multi-turn scalar delta into one sample-level scalar.",
    )
    parser.add_argument(
        "--routing_weight_mode",
        type=str,
        choices=["coarse", "uniform"],
        default="coarse",
        help="coarse: use normalized top_k_scores as weights; uniform: equal weights.",
    )
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
    parser.add_argument(
        "--store_turn_scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to store score_by_turn_scalar (larger output).",
    )
    parser.add_argument(
        "--attach_routing_meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to store routed top-k metadata alongside scores.",
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
    routing_map, inferred_m = scorer._load_routing_jsonl(
        routing_path=args.routing_path,
        max_top_k=int(args.max_top_k),
    )

    if not routing_map:
        raise ValueError("No valid routed rows loaded. Check --routing_path.")

    if args.m_dimensions is None:
        m_dimensions = int(inferred_m)
        logger.info("m_dimensions inferred from routing indices: %d", m_dimensions)
    else:
        if args.m_dimensions <= 0:
            raise ValueError("--m_dimensions must be > 0")
        m_dimensions = int(args.m_dimensions)
        if m_dimensions < inferred_m:
            raise ValueError(
                f"--m_dimensions={m_dimensions} is too small; routing requires >= {inferred_m}."
            )

    logger.info("Loaded %d samples from %s", len(data), args.data_path)
    logger.info("Loaded %d routed rows from %s", len(routing_map), args.routing_path)
    if args.max_samples >= 0:
        logger.info("Scoring first %d samples (--max_samples).", min(args.max_samples, len(data)))
    logger.info("Using concurrency=%d", args.concurrency)
    logger.info("Turn aggregation=%s", args.turn_aggregation)
    logger.info("Routing weight mode=%s", args.routing_weight_mode)

    scored_data = scorer.score_dataset(
        data,
        routing_map=routing_map,
        m_dimensions=m_dimensions,
        turn_aggregation=args.turn_aggregation,
        routing_weight_mode=args.routing_weight_mode,
        concurrency=args.concurrency,
        max_samples=args.max_samples,
        store_turn_scores=bool(args.store_turn_scores),
        attach_routing_meta=bool(args.attach_routing_meta),
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
