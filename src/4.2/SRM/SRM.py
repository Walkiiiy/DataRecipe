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

Output is sparse/dense mapped_vector over m dimensions.
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
        required=True,
        help="Capability descriptions path (.json or .jsonl).",
    )
    parser.add_argument(
        "--m_dimensions",
        type=int,
        default=None,
        help="Total capability dimension m. If omitted, infer from --capabilities_path.",
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
    parser.add_argument("--dense_output", action="store_true", help="Write dense vector of length m instead of sparse dict.")
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
            )
        )
    return items


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


def load_capabilities_jsonl(path: Path, m: int) -> list[str]:
    desc = [f"Capability Index: {i}" for i in range(m)]
    rows = load_jsonl(path)
    for line_idx, obj in enumerate(rows):
        idx = int(obj.get("order", line_idx))
        if idx < 0 or idx >= m:
            continue
        desc[idx] = build_capability_text(obj, idx)
    return desc


def load_capabilities_json(path: Path, m: int) -> list[str]:
    desc = [f"Capability Index: {i}" for i in range(m)]
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                idx = int(item.get("index", item.get("order", i)))
                if 0 <= idx < m:
                    desc[idx] = build_capability_text(item, idx)
            elif isinstance(item, str) and i < m:
                desc[i] = f"Capability Index: {i}\nDescription: {item}"
        return desc

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
            else:
                desc[idx] = f"Capability Index: {idx}\nDescription: {clean_text(v)}"
        return desc

    raise ValueError(f"Unsupported json structure in capabilities file: {path}")


def load_capabilities(capabilities_path: Path, m: int) -> list[str]:
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


def build_jobs(items: list[RoutedItem], capabilities: list[str], m: int) -> list[VerifyItemJob]:
    jobs: list[VerifyItemJob] = []
    j = 0
    for item in items:
        cap_indices: list[int] = []
        cap_scores: list[float] = []
        cap_texts: list[str] = []
        for cap_idx, score in zip(item.top_k_indices, item.top_k_scores):
            if cap_idx < 0 or cap_idx >= m:
                continue
            cap_indices.append(cap_idx)
            cap_scores.append(float(score))
            cap_texts.append(capabilities[cap_idx])

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

    per_row_dense: list[list[float] | None] = [([0.0] * m if dense_output else None) for _ in items]
    per_row_sparse: list[dict[str, float] | None] = [({} if not dense_output else None) for _ in items]
    per_row_rel: list[list[str]] = [[] for _ in items]
    per_row_gate: list[list[float]] = [[] for _ in items]
    per_row_fused: list[list[float]] = [[] for _ in items]

    for job, res in zip(jobs, results):
        row_idx = job.row_index
        for rank, (cap_idx, emb_score) in enumerate(zip(job.top_k_indices, job.top_k_scores)):
            gate = float(res.gates[rank]) if rank < len(res.gates) else 0.0
            rel = str(res.relations[rank]) if rank < len(res.relations) else REL_NONE
            fused = float(emb_score) * gate

            if dense_output:
                assert per_row_dense[row_idx] is not None
                per_row_dense[row_idx][cap_idx] += fused
            else:
                assert per_row_sparse[row_idx] is not None
                if fused != 0.0:
                    key = str(cap_idx)
                    per_row_sparse[row_idx][key] = float(per_row_sparse[row_idx].get(key, 0.0) + fused)

            if include_debug:
                per_row_rel[row_idx].append(rel)
                per_row_gate[row_idx].append(gate)
                per_row_fused[row_idx].append(fused)

    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        mapped_vector: Any
        if dense_output:
            mapped_vector = per_row_dense[i]
        else:
            mapped_vector = per_row_sparse[i]

        row = {
            "id": item.sample_id,
            "mapped_vector": mapped_vector,
        }
        if include_debug:
            row["top_k_indices"] = item.top_k_indices
            row["top_k_scores"] = item.top_k_scores
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

    endpoint = normalize_endpoint(args.base_url)
    data_path = args.data_path or infer_data_path(args.input_path)

    if not args.input_path.exists():
        raise FileNotFoundError(f"input_path not found: {args.input_path}")
    if not args.capabilities_path.exists():
        raise FileNotFoundError(f"capabilities_path not found: {args.capabilities_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    logging.info("Loading data text map: %s", data_path)
    text_map = load_data_text_map(data_path)
    logging.info("Loading routed candidates: %s", args.input_path)
    items = load_routed_items(args.input_path, text_map=text_map)

    inferred_m = infer_m_dimensions(args.capabilities_path)
    if args.m_dimensions is None:
        m_dimensions = inferred_m
        logging.info("m_dimensions inferred from capabilities: %d", m_dimensions)
    else:
        if args.m_dimensions <= 0:
            raise ValueError("--m_dimensions must be > 0")
        m_dimensions = int(args.m_dimensions)
        if m_dimensions != inferred_m:
            raise ValueError(
                f"--m_dimensions={m_dimensions} mismatches inferred capability dimension={inferred_m} "
                f"from {args.capabilities_path}"
            )

    logging.info("Loading capability descriptions: %s", args.capabilities_path)
    capabilities = load_capabilities(args.capabilities_path, m=m_dimensions)

    jobs = build_jobs(items=items, capabilities=capabilities, m=m_dimensions)
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
