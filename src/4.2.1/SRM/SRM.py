#!/usr/bin/env python3
"""
Fine Deep Verification via LLM & Late Fusion (SRM stage)

Given coarse routing outputs:
  - top_k_indices
  - top_k_scores (S_emb^(k))

For each sample x:
  1) Ask LLM ONCE to verify relation for all retained top-k clusters
  2) Parse per-candidate gate W_llm in {0, w, 1}
  3) Late fusion: v_{x,k} = S_emb^(k) * W_llm^(k)

Output is dense mapped_vector over m dimensions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import aiohttp
    AIOHTTP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_IMPORT_ERROR = exc

try:
    from tqdm.asyncio import tqdm_asyncio
except Exception:  # noqa: BLE001
    tqdm_asyncio = None


SYSTEM_PROMPT = (
    "You are a strict capability relation judge.\n"
    "Given one input text and multiple candidate capability descriptions, decide relation for EACH candidate.\n"
    "Each relation must be exactly one of:\n"
    "1) strongly related\n"
    "2) weakly related\n"
    "3) unrelated\n"
    "Return ONLY valid JSON in this format:\n"
    "{\"relations\": [{\"rank\": 0, \"relation\": \"strongly related\"}]}"
)

REL_STRONG = "strongly related"
REL_WEAK = "weakly related"
REL_NONE = "unrelated"
REL_SET = {REL_STRONG, REL_WEAK, REL_NONE}


@dataclass(frozen=True)
class RoutedItem:
    row_index: int
    sample_id: Any
    text: str
    top_k_indices: list[int]
    top_k_scores: list[float]
    top_k_node_ids: list[str] | None = None
    top_k_node_names: list[str] | None = None
    top_k_node_paths: list[str] | None = None


@dataclass(frozen=True)
class VerifyItemJob:
    job_index: int
    row_index: int
    sample_id: Any
    sample_text: str
    top_k_indices: list[int]
    top_k_scores: list[float]
    capability_texts: list[str]


@dataclass(frozen=True)
class VerifyItemResult:
    relations: list[str]
    gates: list[float]
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SRM: Fine deep verification and late fusion.")
    parser.add_argument("--input_path", type=Path, required=True, help="Stage-1 coarse routing jsonl path.")
    parser.add_argument(
        "--capabilities_path",
        type=Path,
        default=None,
        help="Optional capability descriptions path (.json or .jsonl). If omitted, build candidate texts from routing row metadata.",
    )
    parser.add_argument(
        "--m_dimensions",
        type=int,
        default=None,
        help="Total capability dimension m. If omitted, infer as max(capabilities, routed indices).",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Raw data jsonl path used to build sample text; defaults to sibling train.jsonl.",
    )
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--damping_w", type=float, default=0.1, help="w for weakly related.")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_concurrent_requests", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_base_delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument(
        "--dense_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated. Dense vector output is always used; --no-dense_output is ignored.",
    )
    parser.add_argument("--include_debug_fields", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def ensure_dependencies() -> None:
    if aiohttp is None:
        raise RuntimeError(f"Missing required dependency: aiohttp ({AIOHTTP_IMPORT_ERROR})")


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def normalize_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def infer_data_path(input_path: Path) -> Path:
    return input_path.parent / "train.jsonl"


def compose_sample_text(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction"))
    inp = clean_text(row.get("input"))
    out = clean_text(row.get("output") or row.get("response"))
    chunks = [x for x in [instruction, inp, out] if x]
    return "\n".join(chunks)


def choose_row_id(row: dict[str, Any], fallback_idx: int) -> Any:
    for k in ("id", "data_id", "uid", "idx", "index"):
        if k in row and row[k] is not None:
            return row[k]
    return fallback_idx


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid jsonl at {path}:{line_no}: {exc}") from exc
    return out


def load_data_text_map(data_path: Path) -> dict[str, str]:
    rows = load_jsonl(data_path)
    text_map: dict[str, str] = {}
    for i, row in enumerate(rows):
        row_id = choose_row_id(row, i)
        text = compose_sample_text(row)
        text_map[str(row_id)] = text
    return text_map


def parse_top_k_list(row: dict[str, Any], key: str) -> list[Any]:
    val = row.get(key)
    if not isinstance(val, list):
        raise ValueError(f"Missing/invalid '{key}' in routing row: {row}")
    return val


def parse_optional_top_k_list(row: dict[str, Any], key: str, expected_len: int, row_index: int) -> list[Any] | None:
    if key not in row or row.get(key) is None:
        return None
    val = row.get(key)
    if not isinstance(val, list):
        logging.warning("Ignore invalid optional field %s at routed row=%d (not a list).", key, row_index)
        return None
    if len(val) != expected_len:
        logging.warning(
            "Ignore optional field %s at routed row=%d due to length mismatch: got=%d, expected=%d.",
            key,
            row_index,
            len(val),
            expected_len,
        )
        return None
    return val


def load_routed_items(input_path: Path, text_map: dict[str, str]) -> list[RoutedItem]:
    rows = load_jsonl(input_path)
    items: list[RoutedItem] = []
    for i, row in enumerate(rows):
        sample_id = row.get("id", i)
        top_k_indices_raw = parse_top_k_list(row, "top_k_indices")
        top_k_scores_raw = parse_top_k_list(row, "top_k_scores")
        if len(top_k_indices_raw) != len(top_k_scores_raw):
            raise ValueError(f"top_k_indices/top_k_scores length mismatch at row={i}")

        top_k_indices = [int(x) for x in top_k_indices_raw]
        top_k_scores = [float(x) for x in top_k_scores_raw]
        top_k_node_ids_raw = parse_optional_top_k_list(row, "top_k_node_ids", len(top_k_indices), i)
        top_k_node_names_raw = parse_optional_top_k_list(row, "top_k_node_names", len(top_k_indices), i)
        top_k_node_paths_raw = parse_optional_top_k_list(row, "top_k_node_paths", len(top_k_indices), i)

        text = clean_text(row.get("text"))
        if not text:
            text = text_map.get(str(sample_id), "")
        if not text:
            raise ValueError(
                f"Cannot find sample text for id={sample_id}. "
                "Provide --data_path with instruction/input/output fields."
            )

        items.append(
            RoutedItem(
                row_index=i,
                sample_id=sample_id,
                text=text,
                top_k_indices=top_k_indices,
                top_k_scores=top_k_scores,
                top_k_node_ids=[clean_text(x) for x in top_k_node_ids_raw] if top_k_node_ids_raw is not None else None,
                top_k_node_names=[clean_text(x) for x in top_k_node_names_raw] if top_k_node_names_raw is not None else None,
                top_k_node_paths=[clean_text(x) for x in top_k_node_paths_raw] if top_k_node_paths_raw is not None else None,
            )
        )
    return items


def infer_m_dimensions_from_routed_items(items: list[RoutedItem]) -> int:
    max_idx = -1
    for item in items:
        for idx in item.top_k_indices:
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def build_capability_text(obj: dict[str, Any], idx: int) -> str:
    name = clean_text(obj.get("capability_name"))
    name_zh = clean_text(obj.get("capability_name_zh"))
    definition = clean_text(obj.get("capability_definition"))
    node_id = clean_text(obj.get("node_id"))

    keyword_items = obj.get("query_keywords_topk", [])
    keywords: list[str] = []
    if isinstance(keyword_items, list):
        for it in keyword_items[:5]:
            if isinstance(it, dict):
                kw = clean_text(it.get("keyword"))
                if kw:
                    keywords.append(kw)

    lines = [f"Capability Index: {idx}"]
    if node_id:
        lines.append(f"Node ID: {node_id}")
    if name:
        lines.append(f"Name (EN): {name}")
    if name_zh:
        lines.append(f"Name (ZH): {name_zh}")
    if definition:
        lines.append(f"Definition: {definition}")
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    return "\n".join(lines)


def load_capabilities_jsonl(path: Path, m: int) -> tuple[list[str], dict[str, int]]:
    desc = [f"Capability Index: {i}" for i in range(m)]
    node_id_to_index: dict[str, int] = {}
    rows = load_jsonl(path)
    for line_idx, obj in enumerate(rows):
        idx = int(obj.get("order", line_idx))
        if idx < 0 or idx >= m:
            continue
        desc[idx] = build_capability_text(obj, idx)
        node_id = clean_text(obj.get("node_id"))
        if node_id:
            node_id_to_index[node_id] = idx
    return desc, node_id_to_index


def load_capabilities_json(path: Path, m: int) -> tuple[list[str], dict[str, int]]:
    desc = [f"Capability Index: {i}" for i in range(m)]
    node_id_to_index: dict[str, int] = {}
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                idx = int(item.get("index", item.get("order", i)))
                if 0 <= idx < m:
                    desc[idx] = build_capability_text(item, idx)
                    node_id = clean_text(item.get("node_id"))
                    if node_id:
                        node_id_to_index[node_id] = idx
            elif isinstance(item, str) and i < m:
                desc[i] = f"Capability Index: {i}\nDescription: {item}"
        return desc, node_id_to_index

    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if not (0 <= idx < m):
                continue
            if isinstance(v, dict):
                desc[idx] = build_capability_text(v, idx)
                node_id = clean_text(v.get("node_id"))
                if node_id:
                    node_id_to_index[node_id] = idx
            else:
                desc[idx] = f"Capability Index: {idx}\nDescription: {clean_text(v)}"
        return desc, node_id_to_index

    raise ValueError(f"Unsupported json structure in capabilities file: {path}")


def load_capabilities(capabilities_path: Path, m: int) -> tuple[list[str], dict[str, int]]:
    suffix = capabilities_path.suffix.lower()
    if suffix == ".jsonl":
        return load_capabilities_jsonl(capabilities_path, m)
    if suffix == ".json":
        return load_capabilities_json(capabilities_path, m)
    raise ValueError(f"Unsupported capabilities format: {capabilities_path}")


def infer_m_dimensions(capabilities_path: Path) -> int:
    suffix = capabilities_path.suffix.lower()
    if suffix == ".jsonl":
        rows = load_jsonl(capabilities_path)
        if not rows:
            raise ValueError(f"No capability rows found in {capabilities_path}")
        max_idx = -1
        for i, obj in enumerate(rows):
            try:
                idx = int(obj.get("order", i))
            except Exception:
                idx = i
            max_idx = max(max_idx, idx)
        return max_idx + 1

    if suffix == ".json":
        obj = json.loads(capabilities_path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            if not obj:
                raise ValueError(f"Empty capability list in {capabilities_path}")
            max_idx = -1
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    try:
                        idx = int(item.get("index", item.get("order", i)))
                    except Exception:
                        idx = i
                else:
                    idx = i
                max_idx = max(max_idx, idx)
            return max_idx + 1
        if isinstance(obj, dict):
            int_keys: list[int] = []
            for k in obj.keys():
                try:
                    int_keys.append(int(k))
                except Exception:
                    continue
            if not int_keys:
                raise ValueError(f"Cannot infer m from json dict keys in {capabilities_path}")
            return max(int_keys) + 1
        raise ValueError(f"Unsupported json structure in {capabilities_path}")

    raise ValueError(f"Unsupported capabilities format: {capabilities_path}")


def build_batch_prompt(sample_text: str, top_k_indices: list[int], capability_texts: list[str]) -> str:
    blocks: list[str] = []
    for rank, (cap_idx, cap_text) in enumerate(zip(top_k_indices, capability_texts)):
        blocks.append(
            f"[Candidate rank={rank}, capability_index={cap_idx}]\n"
            f"{cap_text}"
        )
    candidate_block = "\n\n".join(blocks)
    return (
        "Task: Determine relation between ONE sample text and EACH candidate capability.\n"
        "For every candidate rank, output one relation label.\n"
        "Allowed labels: strongly related | weakly related | unrelated.\n"
        "Return ONLY JSON in this format:\n"
        "{\"relations\": [{\"rank\": 0, \"relation\": \"strongly related\"}, ...]}\n\n"
        "[Sample Text]\n"
        f"{sample_text}\n\n"
        "[Candidates]\n"
        f"{candidate_block}\n"
    )


def extract_json_object(text: str) -> str | None:
    raw = clean_text(text)
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return match.group(0)
    return None


def parse_relation(content: str) -> str:
    raw = clean_text(content).lower()
    if not raw:
        return REL_NONE

    obj_text = extract_json_object(raw)
    if obj_text:
        try:
            obj = json.loads(obj_text)
            rel = clean_text(obj.get("relation")).lower()
            if rel in REL_SET:
                return rel
        except Exception:  # noqa: BLE001
            pass

    if "strongly related" in raw or '"strong"' in raw:
        return REL_STRONG
    if "weakly related" in raw or '"weak"' in raw:
        return REL_WEAK
    if "unrelated" in raw or "not related" in raw:
        return REL_NONE
    return REL_NONE


def relation_to_gate(relation: str, damping_w: float) -> float:
    if relation == REL_STRONG:
        return 1.0
    if relation == REL_WEAK:
        return float(damping_w)
    return 0.0


def parse_relations(content: str, expected_k: int) -> list[str]:
    raw = clean_text(content)
    if expected_k <= 0:
        return []

    parsed: list[str] = []
    obj_text = extract_json_object(raw.lower())
    if obj_text:
        try:
            obj = json.loads(obj_text)
            rel_obj = obj.get("relations")
            if isinstance(rel_obj, list):
                rank_to_rel: dict[int, str] = {}
                sequential_rel: list[str] = []
                for item in rel_obj:
                    if isinstance(item, str):
                        sequential_rel.append(parse_relation(item))
                        continue
                    if isinstance(item, dict):
                        rel = parse_relation(str(item.get("relation", "")))
                        rank = item.get("rank")
                        if isinstance(rank, int):
                            rank_to_rel[rank] = rel
                        else:
                            sequential_rel.append(rel)
                if rank_to_rel:
                    parsed = [rank_to_rel.get(i, REL_NONE) for i in range(expected_k)]
                else:
                    parsed = sequential_rel[:expected_k]
            elif isinstance(rel_obj, dict):
                rank_to_rel: dict[int, str] = {}
                for k, v in rel_obj.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    rank_to_rel[idx] = parse_relation(str(v))
                parsed = [rank_to_rel.get(i, REL_NONE) for i in range(expected_k)]
        except Exception:  # noqa: BLE001
            parsed = []

    if not parsed:
        labels = re.findall(r"strongly related|weakly related|unrelated", raw.lower())
        parsed = [parse_relation(x) for x in labels[:expected_k]]

    if len(parsed) < expected_k:
        parsed.extend([REL_NONE] * (expected_k - len(parsed)))
    if len(parsed) > expected_k:
        parsed = parsed[:expected_k]
    return parsed


def extract_assistant_content(resp_json: dict[str, Any]) -> str:
    choices = resp_json.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return clean_text(message.get("content"))


async def verify_one_job(
    job: VerifyItemJob,
    session: aiohttp.ClientSession,
    endpoint: str,
    api_key: str,
    model_name: str,
    damping_w: float,
    semaphore: asyncio.Semaphore,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    temperature: float,
    max_tokens: int,
) -> VerifyItemResult:
    prompt = build_batch_prompt(job.sample_text, job.top_k_indices, job.capability_texts)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    use_response_format = True
    last_error: str | None = None

    for attempt in range(max_retries):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with semaphore:
                req_timeout = aiohttp.ClientTimeout(total=timeout)
                async with session.post(endpoint, headers=headers, json=payload, timeout=req_timeout) as resp:
                    text = await resp.text()
                    status = resp.status

            if status == 400 and use_response_format and "response_format" in text.lower():
                use_response_format = False
                continue

            if status in (429, 500, 502, 503, 504):
                raise RuntimeError(f"retryable status={status}, body={text[:400]}")
            if status >= 400:
                fallback_rel = [REL_NONE] * len(job.top_k_indices)
                fallback_gate = [0.0] * len(job.top_k_indices)
                return VerifyItemResult(relations=fallback_rel, gates=fallback_gate, error=f"non-retryable status={status}")

            try:
                resp_json = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"response json decode failed: {exc}; body={text[:400]}") from exc

            content = extract_assistant_content(resp_json)
            relations = parse_relations(content, expected_k=len(job.top_k_indices))
            gates = [relation_to_gate(rel, damping_w) for rel in relations]
            return VerifyItemResult(relations=relations, gates=gates, error=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            last_error = str(exc)
            if attempt >= max_retries - 1:
                break
            sleep_s = retry_base_delay * (2**attempt) + random.uniform(0, 0.2)
            await asyncio.sleep(sleep_s)

    fallback_rel = [REL_NONE] * len(job.top_k_indices)
    fallback_gate = [0.0] * len(job.top_k_indices)
    return VerifyItemResult(relations=fallback_rel, gates=fallback_gate, error=last_error)


async def run_verification(
    jobs: list[VerifyItemJob],
    endpoint: str,
    api_key: str,
    model_name: str,
    damping_w: float,
    max_concurrent_requests: int,
    timeout: float,
    max_retries: int,
    retry_base_delay: float,
    temperature: float,
    max_tokens: int,
) -> list[VerifyItemResult]:
    semaphore = asyncio.Semaphore(max(1, max_concurrent_requests))

    async with aiohttp.ClientSession() as session:
        coros = [
            verify_one_job(
                job=job,
                session=session,
                endpoint=endpoint,
                api_key=api_key,
                model_name=model_name,
                damping_w=damping_w,
                semaphore=semaphore,
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for job in jobs
        ]
        if tqdm_asyncio is not None:
            return await tqdm_asyncio.gather(*coros, total=len(coros), desc="LLM verification (1 request / sample)")
        return await asyncio.gather(*coros)


def routed_fallback_capability_text(item: RoutedItem, rank: int, mapped_idx: int) -> str:
    lines = [f"Capability Index: {mapped_idx}"]
    if item.top_k_node_ids is not None and rank < len(item.top_k_node_ids):
        node_id = clean_text(item.top_k_node_ids[rank])
        if node_id:
            lines.append(f"Node ID: {node_id}")
    if item.top_k_node_names is not None and rank < len(item.top_k_node_names):
        name = clean_text(item.top_k_node_names[rank])
        if name:
            lines.append(f"Name (EN): {name}")
    if item.top_k_node_paths is not None and rank < len(item.top_k_node_paths):
        path = clean_text(item.top_k_node_paths[rank])
        if path:
            lines.append(f"Tree Path: {path}")
    lines.append("Definition: Derived from coarse routing metadata.")
    return "\n".join(lines)


def build_jobs(
    items: list[RoutedItem],
    capabilities: list[str] | None,
    m: int,
    node_id_to_index: dict[str, int] | None,
) -> list[VerifyItemJob]:
    node_id_map = node_id_to_index or {}
    use_capability_file = capabilities is not None
    jobs: list[VerifyItemJob] = []
    j = 0
    mapped_by_node_id = 0
    missing_node_id_mapping = 0
    fallback_to_index = 0
    dropped_invalid_index = 0
    routed_text_only = 0
    for item in items:
        cap_indices: list[int] = []
        cap_scores: list[float] = []
        cap_texts: list[str] = []
        use_node_id = use_capability_file and item.top_k_node_ids is not None and len(item.top_k_node_ids) == len(item.top_k_indices)
        for rank, (cap_idx, score) in enumerate(zip(item.top_k_indices, item.top_k_scores)):
            mapped_idx: int | None = None
            used_fallback_text = False
            if use_capability_file and use_node_id:
                node_id = clean_text(item.top_k_node_ids[rank])
                if node_id:
                    mapped_idx = node_id_map.get(node_id)
                if mapped_idx is None:
                    missing_node_id_mapping += 1
                    if cap_idx < 0 or cap_idx >= m:
                        dropped_invalid_index += 1
                        continue
                    mapped_idx = cap_idx
                    fallback_to_index += 1
                    used_fallback_text = True
                else:
                    mapped_by_node_id += 1
            else:
                if cap_idx < 0 or cap_idx >= m:
                    dropped_invalid_index += 1
                    continue
                mapped_idx = cap_idx
                fallback_to_index += 1
                if not use_capability_file:
                    used_fallback_text = True

            cap_indices.append(mapped_idx)
            cap_scores.append(float(score))
            if used_fallback_text or not use_capability_file:
                cap_texts.append(routed_fallback_capability_text(item, rank, mapped_idx))
                routed_text_only += 1
            else:
                assert capabilities is not None
                cap_texts.append(capabilities[mapped_idx])

        jobs.append(
            VerifyItemJob(
                job_index=j,
                row_index=item.row_index,
                sample_id=item.sample_id,
                sample_text=item.text,
                top_k_indices=cap_indices,
                top_k_scores=cap_scores,
                capability_texts=cap_texts,
            )
        )
        j += 1
    logging.info(
        "Job mapping stats: mapped_by_node_id=%d, fallback_to_index=%d, missing_node_id_mapping=%d, dropped_invalid_index=%d, routed_text_only=%d",
        mapped_by_node_id,
        fallback_to_index,
        missing_node_id_mapping,
        dropped_invalid_index,
        routed_text_only,
    )
    return jobs


def fuse_results(
    items: list[RoutedItem],
    jobs: list[VerifyItemJob],
    results: list[VerifyItemResult],
    m: int,
    dense_output: bool,
    include_debug: bool,
) -> list[dict[str, Any]]:
    if len(jobs) != len(results):
        raise ValueError("jobs/results length mismatch")

    if not dense_output:
        logging.warning("Sparse output mode is deprecated and ignored; forcing dense output.")

    per_row_dense: list[list[float]] = [[0.0] * m for _ in items]
    per_row_rel: list[list[str]] = [[] for _ in items]
    per_row_gate: list[list[float]] = [[] for _ in items]
    per_row_fused: list[list[float]] = [[] for _ in items]
    per_row_used_indices: list[list[int]] = [[] for _ in items]
    per_row_used_scores: list[list[float]] = [[] for _ in items]

    for job, res in zip(jobs, results):
        row_idx = job.row_index
        for rank, (cap_idx, emb_score) in enumerate(zip(job.top_k_indices, job.top_k_scores)):
            gate = float(res.gates[rank]) if rank < len(res.gates) else 0.0
            rel = str(res.relations[rank]) if rank < len(res.relations) else REL_NONE
            fused = float(emb_score) * gate

            per_row_dense[row_idx][cap_idx] += fused

            per_row_used_indices[row_idx].append(int(cap_idx))
            per_row_used_scores[row_idx].append(float(emb_score))

            if include_debug:
                per_row_rel[row_idx].append(rel)
                per_row_gate[row_idx].append(gate)
                per_row_fused[row_idx].append(fused)

    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        mapped_vector: list[float] = per_row_dense[i]

        row = {
            "id": item.sample_id,
            "mapped_vector": mapped_vector,
            "score": mapped_vector,
        }
        if include_debug:
            row["top_k_indices"] = per_row_used_indices[i]
            row["top_k_scores"] = per_row_used_scores[i]
            row["raw_top_k_indices"] = item.top_k_indices
            row["raw_top_k_scores"] = item.top_k_scores
            if item.top_k_node_ids is not None:
                row["top_k_node_ids"] = item.top_k_node_ids
            if item.top_k_node_names is not None:
                row["top_k_node_names"] = item.top_k_node_names
            if item.top_k_node_paths is not None:
                row["top_k_node_paths"] = item.top_k_node_paths
            row["llm_relations"] = per_row_rel[i]
            row["llm_gates"] = per_row_gate[i]
            row["fused_top_k"] = per_row_fused[i]
        out.append(row)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def async_main(args: argparse.Namespace) -> None:
    if args.damping_w < 0 or args.damping_w > 1:
        raise ValueError("--damping_w should be in [0, 1]")
    if args.max_retries <= 0:
        raise ValueError("--max_retries must be > 0")
    if not args.dense_output:
        logging.warning("--no-dense_output is deprecated and ignored; forcing dense output.")

    endpoint = normalize_endpoint(args.base_url)
    data_path = args.data_path or infer_data_path(args.input_path)
    capabilities_path: Path | None = args.capabilities_path

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")
    if capabilities_path is not None and not capabilities_path.exists():
        raise FileNotFoundError(f"capabilities_path not found: {capabilities_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    logging.info("Loading data text map: %s", data_path)
    text_map = load_data_text_map(data_path)
    logging.info("Loading routed candidates: %s", args.input_path)
    items = load_routed_items(args.input_path, text_map=text_map)

    inferred_m_cap = infer_m_dimensions(capabilities_path) if capabilities_path is not None else 0
    inferred_m_routed = infer_m_dimensions_from_routed_items(items)
    inferred_m = max(inferred_m_cap, inferred_m_routed)
    if args.m_dimensions is None:
        m_dimensions = inferred_m
        logging.info(
            "m_dimensions inferred: max(capabilities=%d, routed=%d) => %d",
            inferred_m_cap,
            inferred_m_routed,
            m_dimensions,
        )
    else:
        if args.m_dimensions <= 0:
            raise ValueError("--m_dimensions must be > 0")
        m_dimensions = int(args.m_dimensions)
        if m_dimensions < inferred_m_cap or m_dimensions < inferred_m_routed:
            raise ValueError(
                f"--m_dimensions={m_dimensions} is too small; requires >= max(capabilities={inferred_m_cap}, routed={inferred_m_routed})."
            )

    capabilities: list[str] | None = None
    node_id_to_index: dict[str, int] | None = None
    if capabilities_path is not None:
        logging.info("Loading capability descriptions: %s", capabilities_path)
        capabilities, node_id_to_index = load_capabilities(capabilities_path, m=m_dimensions)
        logging.info("Capability node-id mapping loaded: %d entries", len(node_id_to_index))
    else:
        logging.info(
            "No --capabilities_path provided; candidate texts will be built from routed metadata only (node_id/name/path)."
        )

    jobs = build_jobs(
        items=items,
        capabilities=capabilities,
        m=m_dimensions,
        node_id_to_index=node_id_to_index,
    )
    logging.info(
        "SRM verification jobs prepared: samples=%d, requests=%d (1 request/sample), endpoint=%s, model=%s, concurrency=%d",
        len(items),
        len(jobs),
        endpoint,
        args.model_name,
        args.max_concurrent_requests,
    )

    results = await run_verification(
        jobs=jobs,
        endpoint=endpoint,
        api_key=args.api_key,
        model_name=args.model_name,
        damping_w=args.damping_w,
        max_concurrent_requests=args.max_concurrent_requests,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    fused_rows = fuse_results(
        items=items,
        jobs=jobs,
        results=results,
        m=m_dimensions,
        dense_output=args.dense_output,
        include_debug=args.include_debug_fields,
    )
    write_jsonl(args.output_path, fused_rows)

    err_cnt = sum(1 for x in results if x.error)
    strong_cnt = sum(1 for x in results for rel in x.relations if rel == REL_STRONG)
    weak_cnt = sum(1 for x in results for rel in x.relations if rel == REL_WEAK)
    none_cnt = sum(1 for x in results for rel in x.relations if rel == REL_NONE)
    logging.info(
        "SRM done: output=%s, rows=%d, requests=%d, strong=%d, weak=%d, unrelated=%d, errors=%d",
        args.output_path,
        len(fused_rows),
        len(results),
        strong_cnt,
        weak_cnt,
        none_cnt,
        err_cnt,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ensure_dependencies()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
