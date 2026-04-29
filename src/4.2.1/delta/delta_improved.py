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
class RoutedCandidate:
    index: int
    coarse_score: float
    node_id: str
    node_name: str
    node_path: str


@dataclass(frozen=True)
class RoutedItem:
    row_index: int
    sample_id: Any
    top_k_indices: List[int]
    top_k_scores: List[float]
    top_k_node_ids: List[str]
    top_k_node_names: List[str]
    top_k_node_paths: List[str]
    candidates: List[RoutedCandidate]


class DeepSeekDeltaArrayScorer:
    """Top-k routed DEITA delta scorer with DeepSeek API backend.

    For each turn:
      delta_k = complexity(turn) * quality(turn, routed_cluster_k)
    Then aggregate over turns and map scores into an m-dim dense vector.
    """

    complexity_template = (
        "You are a strict grader.\n"
        "Evaluate the complexity of the following user query on a 1-6 scale.\n"
        "Return ONLY one digit: 1, 2, 3, 4, 5, or 6.\n"
        "No words, no explanation, no markdown, no punctuation.\n"
        "##Query: {instruction}\n"
        "##Complexity:"
    )

    quality_cluster_template = (
        "You are a strict grader.\n"
        "Given one candidate capability cluster from a capability tree, evaluate how well the response fits "
        "this capability for the given question on a 1-6 scale.\n"
        "Return ONLY one digit: 1, 2, 3, 4, 5, or 6.\n"
        "No words, no explanation, no markdown, no punctuation.\n"
        "#CapabilityClusterName#:\n"
        "{capability_name}\n"
        "#CapabilityNodeId#:\n"
        "{capability_node_id}\n"
        "#CapabilityPath#:\n"
        "{capability_path}\n"
        "#Question#:\n"
        "{instruction}\n"
        "#Response#:\n"
        "{output}\n"
        "##Quality:"
    )

    quality_batch_template = (
        "You are a strict grader.\n"
        "For the given question-response pair, rate response quality for EACH listed capability cluster on a 1-6 scale.\n"
        "Return ONLY one JSON object with this exact schema:\n"
        '{{"scores":[d1,d2,...,dN]}}\n'
        "Constraints:\n"
        "- Each d_i must be an integer from 1 to 6.\n"
        "- N must equal the number of clusters provided.\n"
        "- No explanation, no markdown, no extra keys.\n"
        "#Question#:\n"
        "{instruction}\n"
        "#Response#:\n"
        "{output}\n"
        "#CandidateClusters#:\n"
        "{cluster_block}\n"
        "##JSON:"
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

                top_k_node_ids_raw = DeepSeekDeltaArrayScorer._extract_optional_list(row, "top_k_node_ids")
                top_k_node_names_raw = DeepSeekDeltaArrayScorer._extract_optional_list(row, "top_k_node_names")
                top_k_node_paths_raw = DeepSeekDeltaArrayScorer._extract_optional_list(row, "top_k_node_paths")

                candidates: List[RoutedCandidate] = []
                top_k_indices: List[int] = []
                top_k_scores: List[float] = []
                top_k_node_ids: List[str] = []
                top_k_node_names: List[str] = []
                top_k_node_paths: List[str] = []

                for rank in range(limit):
                    idx = DeepSeekDeltaArrayScorer._to_int(top_k_indices_raw[rank], -1)
                    if idx < 0:
                        continue
                    score = DeepSeekDeltaArrayScorer._to_float(top_k_scores_raw[rank], 0.0)

                    node_id = ""
                    if top_k_node_ids_raw is not None and rank < len(top_k_node_ids_raw):
                        node_id = str(top_k_node_ids_raw[rank] or "").strip()

                    node_name = ""
                    if top_k_node_names_raw is not None and rank < len(top_k_node_names_raw):
                        node_name = str(top_k_node_names_raw[rank] or "").strip()
                    if not node_name:
                        node_name = node_id or f"cluster_{idx}"

                    node_path = ""
                    if top_k_node_paths_raw is not None and rank < len(top_k_node_paths_raw):
                        node_path = str(top_k_node_paths_raw[rank] or "").strip()

                    candidate = RoutedCandidate(
                        index=idx,
                        coarse_score=score,
                        node_id=node_id,
                        node_name=node_name,
                        node_path=node_path,
                    )
                    candidates.append(candidate)
                    top_k_indices.append(idx)
                    top_k_scores.append(score)
                    top_k_node_ids.append(node_id)
                    top_k_node_names.append(node_name)
                    top_k_node_paths.append(node_path)
                    inferred_m = max(inferred_m, idx + 1)

                if not candidates:
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
                    candidates=candidates,
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

        merged_instruction = DeepSeekDeltaArrayScorer._merge_instruction_and_input(
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
        probs = DeepSeekDeltaArrayScorer._softmax(logits)
        return float(sum(p * s for p, s in zip(probs, DeepSeekDeltaArrayScorer.score_template)))

    @staticmethod
    def _extract_digit_score(text: str) -> Optional[float]:
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
    def _clip_text(text: str, max_chars: int) -> str:
        x = str(text or "").strip()
        if len(x) <= max_chars:
            return x
        return x[: max(0, max_chars - 3)] + "..."

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _normalize_batch_scores(
        values: List[Any],
        expected_len: int,
    ) -> List[float]:
        out: List[float] = []
        for v in values:
            try:
                x = int(v)
            except Exception:  # noqa: BLE001
                continue
            if x < 1 or x > 6:
                continue
            out.append(float(x))

        if len(out) < expected_len:
            out.extend([3.0] * (expected_len - len(out)))
        return out[:expected_len]

    @staticmethod
    def _resolve_sample_routing(
        sample: Dict[str, Any],
        fallback_idx: int,
        routing_map: Dict[str, RoutedItem],
    ) -> Tuple[Any, RoutedItem]:
        sample_id = DeepSeekDeltaArrayScorer._choose_row_id(sample, fallback_idx)
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

    def _call_deepseek_text(self, user_input: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_input}],
            "max_tokens": max(64, max_tokens),
            "temperature": 0,
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
                    raise ValueError("Empty choices in DeepSeek response.")

                message_content = str(choices[0].get("message", {}).get("content", "")).strip()
                if not message_content:
                    raise ValueError("Empty message content in DeepSeek response.")
                return message_content

            except (requests.RequestException, ValueError, KeyError) as exc:
                if attempt == self.max_retries:
                    logger.warning("DeepSeek text call failed after retries: %s", exc)
                    return ""
                sleep_seconds = self.retry_backoff ** (attempt - 1)
                time.sleep(sleep_seconds)

        return ""

    def infer_complexity(self, input_text: str) -> float:
        user_input = self.complexity_template.format(instruction=input_text)
        return self._call_deepseek(user_input)

    def infer_quality_for_candidate(self, input_text: str, resp_text: str, candidate: RoutedCandidate) -> float:
        user_input = self.quality_cluster_template.format(
            capability_name=candidate.node_name or (candidate.node_id or f"cluster_{candidate.index}"),
            capability_node_id=candidate.node_id or "N/A",
            capability_path=candidate.node_path or "N/A",
            instruction=input_text,
            output=resp_text,
        )
        return self._call_deepseek(user_input)

    def infer_quality_for_candidate_batch(
        self,
        input_text: str,
        resp_text: str,
        candidates: List[RoutedCandidate],
        cluster_path_max_chars: int = 180,
        quality_max_tokens: int = 512,
    ) -> List[float]:
        if not candidates:
            return []

        lines: List[str] = []
        for i, candidate in enumerate(candidates, start=1):
            path = self._clip_text(candidate.node_path or "N/A", cluster_path_max_chars)
            name = candidate.node_name or (candidate.node_id or f"cluster_{candidate.index}")
            lines.append(
                f"{i}. index={candidate.index} | name={name} | node_id={candidate.node_id or 'N/A'} | path={path}"
            )

        cluster_block = "\n".join(lines)
        user_input = self.quality_batch_template.format(
            instruction=input_text,
            output=resp_text,
            cluster_block=cluster_block,
        )

        raw = self._call_deepseek_text(user_input, max_tokens=quality_max_tokens)
        if not raw:
            return [3.0] * len(candidates)

        obj = self._extract_json_object(raw)
        if obj is not None:
            scores = obj.get("scores")
            if isinstance(scores, list):
                return self._normalize_batch_scores(scores, expected_len=len(candidates))

        digits = re.findall(r"(?<!\d)([1-6])(?!\d)", raw)
        if digits:
            return self._normalize_batch_scores(digits, expected_len=len(candidates))

        return [3.0] * len(candidates)

    @staticmethod
    def _aggregate_turn_vectors(vectors: List[List[float]], mode: str) -> List[float]:
        if not vectors:
            return []
        width = len(vectors[0])
        if width == 0:
            return []

        if mode == "sum":
            return [float(sum(v[i] for v in vectors)) for i in range(width)]
        if mode == "mean":
            return [float(sum(v[i] for v in vectors) / len(vectors)) for i in range(width)]
        if mode == "max":
            return [float(max(v[i] for v in vectors)) for i in range(width)]
        raise ValueError(f"Unsupported turn aggregation: {mode}")

    def _score_one_sample(
        self,
        sample: Dict[str, Any],
        sample_id: Any,
        routed_item: RoutedItem,
        m_dimensions: int,
        turn_aggregation: str,
        store_turn_scores: bool,
        attach_routing_meta: bool,
        quality_mode: str,
        cluster_batch_size: int,
        cluster_path_max_chars: int,
        quality_max_tokens: int,
    ) -> Dict[str, Any]:
        turns = self._extract_turns(sample)
        turn_vectors: List[List[float]] = []

        for turn in turns:
            instruction = str(turn.get("instruction", ""))
            response = str(turn.get("response", ""))

            complexity_score = float(self.infer_complexity(instruction))
            delta_vector: List[float] = [0.0] * m_dimensions

            if quality_mode == "single":
                for candidate in routed_item.candidates:
                    if candidate.index < 0 or candidate.index >= m_dimensions:
                        continue
                    quality_score = float(self.infer_quality_for_candidate(instruction, response, candidate))
                    delta_vector[candidate.index] += float(complexity_score * quality_score)
            else:
                bs = max(1, cluster_batch_size)
                start = 0
                while start < len(routed_item.candidates):
                    end = min(start + bs, len(routed_item.candidates))
                    chunk = routed_item.candidates[start:end]
                    quality_scores = self.infer_quality_for_candidate_batch(
                        input_text=instruction,
                        resp_text=response,
                        candidates=chunk,
                        cluster_path_max_chars=cluster_path_max_chars,
                        quality_max_tokens=quality_max_tokens,
                    )
                    for j, quality_score in enumerate(quality_scores):
                        idx = chunk[j].index
                        if idx < 0 or idx >= m_dimensions:
                            continue
                        delta_vector[idx] += float(complexity_score * float(quality_score))
                    start = end

            turn_vectors.append(delta_vector)

        mapped_vector = self._aggregate_turn_vectors(turn_vectors, turn_aggregation)
        if len(mapped_vector) != m_dimensions:
            mapped_vector = (mapped_vector + [0.0] * m_dimensions)[:m_dimensions]

        sample["id"] = sample_id
        sample["mapped_vector"] = mapped_vector
        sample["score"] = mapped_vector
        sample["score_type"] = "delta_array_mapped_vector"
        sample["score_turn_aggregation"] = turn_aggregation

        if store_turn_scores:
            sample["score_by_turn"] = turn_vectors

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
        concurrency: int = 1,
        max_samples: int = -1,
        turn_aggregation: str = "sum",
        store_turn_scores: bool = False,
        attach_routing_meta: bool = True,
        quality_mode: str = "batch",
        cluster_batch_size: int = 5,
        cluster_path_max_chars: int = 180,
        quality_max_tokens: int = 512,
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
                    store_turn_scores=store_turn_scores,
                    attach_routing_meta=attach_routing_meta,
                    quality_mode=quality_mode,
                    cluster_batch_size=cluster_batch_size,
                    cluster_path_max_chars=cluster_path_max_chars,
                    quality_max_tokens=quality_max_tokens,
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
                store_turn_scores=store_turn_scores,
                attach_routing_meta=attach_routing_meta,
                quality_mode=quality_mode,
                cluster_batch_size=cluster_batch_size,
                cluster_path_max_chars=cluster_path_max_chars,
                quality_max_tokens=quality_max_tokens,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute top-k routed DEITA delta score vectors with DeepSeek API."
    )
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
        help="Use at most top-k routed clusters per sample for quality prompts; <=0 means all.",
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
        "--turn_aggregation",
        type=str,
        choices=["sum", "mean", "max"],
        default="sum",
        help="How to aggregate multi-turn vectors into one sample-level vector.",
    )
    parser.add_argument(
        "--quality_mode",
        type=str,
        choices=["batch", "single"],
        default="batch",
        help="batch: one request scores all top-k clusters by chunk; single: one request per cluster.",
    )
    parser.add_argument(
        "--cluster_batch_size",
        type=int,
        default=5,
        help="How many routed clusters to score per quality request when --quality_mode=batch.",
    )
    parser.add_argument(
        "--cluster_path_max_chars",
        type=int,
        default=180,
        help="Max chars of cluster path included in batch quality prompt.",
    )
    parser.add_argument(
        "--quality_max_tokens",
        type=int,
        default=512,
        help="max_tokens for batch quality response generation.",
    )
    parser.add_argument(
        "--store_turn_scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to store score_by_turn (larger output).",
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

    scorer = DeepSeekDeltaArrayScorer(
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
    logger.info(
        "Quality mode=%s, cluster_batch_size=%d, m_dimensions=%d",
        args.quality_mode,
        max(1, args.cluster_batch_size),
        m_dimensions,
    )

    scoring_data = data if args.max_samples < 0 else data[: args.max_samples]
    turn_count = sum(len(scorer._extract_turns(sample)) for sample in scoring_data)
    max_top_k = max(1, int(args.max_top_k)) if int(args.max_top_k) != 0 else 1
    if args.quality_mode == "batch":
        calls_per_turn = 1 + math.ceil(max_top_k / max(1, args.cluster_batch_size))
    else:
        calls_per_turn = 1 + max_top_k
    estimated_calls = turn_count * calls_per_turn
    logger.info(
        "Estimated API calls: %d (turns=%d, approx_calls_per_turn=%d, top_k<=%d)",
        estimated_calls,
        turn_count,
        calls_per_turn,
        max_top_k,
    )

    scored_data = scorer.score_dataset(
        data=data,
        routing_map=routing_map,
        m_dimensions=m_dimensions,
        concurrency=args.concurrency,
        max_samples=args.max_samples,
        turn_aggregation=args.turn_aggregation,
        store_turn_scores=bool(args.store_turn_scores),
        attach_routing_meta=bool(args.attach_routing_meta),
        quality_mode=args.quality_mode,
        cluster_batch_size=max(1, args.cluster_batch_size),
        cluster_path_max_chars=max(32, args.cluster_path_max_chars),
        quality_max_tokens=max(128, args.quality_max_tokens),
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
