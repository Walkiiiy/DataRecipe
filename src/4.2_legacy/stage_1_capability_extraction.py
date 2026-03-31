"""Stage 4.2 / Stage 1: Extract capability names for tree clusters with DeepSeek.

This script links:
1) `capability_tree_final_pruned.json` (cluster membership by `data_ids`)
2) `*_cdt_profile.jsonl` (sample-level `CDT_description`)

Then it calls DeepSeek Chat Completions to name each capability cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    torch = None
    SentenceTransformer = None

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TREE_JSON = Path("data/alpaca-gpt4-data-en/capability_tree_final_pruned.json")
DEFAULT_PROFILE_JSONL = Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl")

SYSTEM_PROMPT = (
    "You are an expert in instruction-data capability taxonomy.\n"
    "Given representative samples from ONE capability cluster, infer the latent shared capability.\n"
    "Return strict JSON only with keys:\n"
    '{\n'
    '  "capability_name": "<short English name>",\n'
    '  "capability_name_zh": "<short Chinese name>",\n'
    '  "capability_definition": "<one-sentence definition>"\n'
    '}\n'
    "Rules:\n"
    "- capability_name: 2-6 words, concise, reusable across datasets.\n"
    "- capability_name_zh: concise Chinese phrase, no punctuation.\n"
    "- capability_definition: <= 35 English words, must summarize cognition + domain + task.\n"
    "- Avoid concrete entities, brands, dates, and dataset-specific terms.\n"
    "- No markdown, no extra keys, no explanation outside JSON."
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "within",
    "task",
    "tasks",
    "sample",
    "samples",
    "processing",
    "domain",
    "topic",
    "performing",
    "involves",
    "demonstrates",
    "through",
    "using",
    "based",
    "general",
    "abstract",
    "content",
    "classify",
    "classification",
    "intent",
    "correct",
    "category",
    "conversation",
    "banking",
    "router",
    "following",
    "query",
    "queries",
    "user",
    "users",
    "please",
    "need",
    "help",
    "can",
    "could",
    "would",
    "should",
    "will",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "you",
    "your",
    "yours",
    "get",
    "got",
    "have",
    "has",
    "had",
    "not",
    "still",
    "there",
    "their",
    "them",
    "they",
    "our",
    "ours",
    "are",
    "is",
    "was",
    "were",
    "been",
    "being",
    "but",
    "very",
    "also",
    "just",
    "than",
    "then",
    "too",
    "much",
    "many",
    "any",
    "all",
    "some",
    "one",
    "two",
}

GENERIC_NAME_PATTERNS = [
    r"^\s*(financial|banking|conversational|conversation|dialogue|service)?\s*(intent|query)?\s*classification\s*$",
    r"^\s*(financial|banking|conversational|conversation|dialogue)\s+(intent|query)\s+classification\s*$",
]

GENERIC_ZH_NAMES = {
    "金融意图分类",
    "对话意图分类",
    "金融对话意图分类",
    "金融交易意图分类",
    "银行意图分类",
    "能力簇",
}

LABEL_ZH_OVERRIDES = {
    "pin_blocked": "PIN码被锁定",
    "passcode_forgotten": "忘记登录密码",
    "unable_to_verify_identity": "身份验证失败",
    "card_arrival": "卡片寄送查询",
    "lost_or_stolen_phone": "手机丢失或被盗",
    "get_physical_card": "获取实体卡",
    "wrong_amount_of_cash_received": "取现金额错误",
    "top_up_failed": "充值失败",
    "transfer_not_received_by_recipient": "收款方未到账",
    "extra_charge_on_statement": "账单额外扣费",
    "receiving_money": "收款到账",
    "activate_my_card": "卡片激活",
    "visa_or_mastercard": "Visa或Mastercard咨询",
    "declined_cash_withdrawal": "取现被拒",
    "card_swallowed": "吞卡处理",
    "refund_not_showing_up": "退款未到账",
    "card_payment_not_recognised": "卡支付记录不识别",
    "compromised_card": "卡片疑似被盗刷",
    "cash_withdrawal_not_recognised": "取现记录不识别",
    "reverted_card_payment": "卡支付被撤销",
    "balance_not_updated_after_cheque_or_cash_deposit": "存款后余额未更新",
    "wrong_exchange_rate_for_cash_withdrawal": "取现汇率错误",
    "card_payment_wrong_exchange_rate": "卡支付汇率错误",
    "transfer_timing": "转账时效",
    "card_linking": "卡片绑定",
    "transaction_charged_twice": "重复扣费",
    "getting_spare_card": "申请备用卡",
    "exchange_rate": "汇率咨询",
    "pending_transfer": "转账处理中",
    "country_support": "国家支持范围",
}

TOKEN_ZH_MAP = {
    "card": "卡",
    "payment": "支付",
    "transfer": "转账",
    "pending": "处理中",
    "pin": "PIN码",
    "passcode": "密码",
    "forgotten": "忘记",
    "lost": "丢失",
    "stolen": "被盗",
    "phone": "手机",
    "verify": "验证",
    "identity": "身份",
    "cash": "现金",
    "withdrawal": "取现",
    "exchange": "汇率",
    "rate": "汇率",
    "top": "充值",
    "failed": "失败",
    "declined": "被拒",
    "arrival": "寄送",
    "linking": "绑定",
    "link": "绑定",
    "spare": "备用",
    "country": "国家",
    "support": "支持",
    "recipient": "收款方",
    "received": "到账",
    "receiving": "收款",
    "money": "资金",
    "balance": "余额",
    "cheque": "支票",
    "deposit": "存款",
    "transaction": "交易",
    "charged": "扣费",
    "twice": "重复",
    "reverted": "撤销",
    "visa": "Visa",
    "mastercard": "Mastercard",
    "timing": "时效",
    "recognised": "识别",
    "wrong": "错误",
    "unable": "无法",
    "physical": "实体",
    "swallowed": "吞卡",
    "refund": "退款",
    "showing": "显示",
}


@dataclass(frozen=True)
class ClusterSpec:
    order: int
    node_id: str
    depth: int
    data_ids: list[str]
    children_count: int
    subtree_size: int
    leaf_payload_size: int


@dataclass(frozen=True)
class Config:
    tree_json: Path
    profile_jsonl: Path
    output_jsonl: Path
    output_summary_json: Path
    cluster_scope: str
    min_cluster_size: int
    max_clusters: int | None
    samples_per_cluster: int
    include_data_ids: bool
    include_raw_response: bool
    seed: int
    api_key: str
    model: str
    base_url: str
    timeout: int
    concurrency: int
    max_retries: int
    retry_base_delay: float
    temperature: float
    max_tokens: int
    include_center_vector: bool
    center_vector_precision: int
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    force_rebuild_center_vector: bool
    resume: bool
    print_named_tree: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract capability names for capability-tree clusters with DeepSeek."
    )
    parser.add_argument("--tree-json", type=Path, default=DEFAULT_TREE_JSON)
    parser.add_argument("--profile-jsonl", type=Path, default=DEFAULT_PROFILE_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    parser.add_argument(
        "--cluster-scope",
        type=str,
        choices=["leaf", "data-bearing", "subtree"],
        default="leaf",
        help=(
            "leaf: only leaf nodes with data_ids; "
            "data-bearing: nodes that directly contain data_ids; "
            "subtree: all nodes using full subtree data_ids."
        ),
    )
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--max-clusters", type=int, default=-1, help="-1 means all clusters.")
    parser.add_argument("--samples-per-cluster", type=int, default=12)
    parser.add_argument("--include-data-ids", action="store_true")
    parser.add_argument("--include-raw-response", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--include-center-vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to compute and save each cluster centroid vector in output jsonl.",
    )
    parser.add_argument(
        "--center-vector-precision",
        type=int,
        default=6,
        help="Round centroid vector to N decimals; -1 means no rounding.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model used to compute cluster centroid vectors.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="auto",
        help="Embedding device: auto/cpu/cuda",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=256,
        help="Batch size for embedding encoding when building centroid vectors.",
    )
    parser.add_argument(
        "--force-rebuild-center-vector",
        action="store_true",
        help="Force rebuild center_vector for all clusters (useful when switching embedding model).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip node_id already existing in output jsonl.")
    parser.add_argument(
        "--print-named-tree",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print tree structure to CLI after main flow, using processed cluster names.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def normalize_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def clean_text(text: Any) -> str:
    raw = str(text or "").strip()
    return " ".join(raw.split())


def humanize_label(label: str) -> str:
    x = clean_text(label).replace("_", " ").replace("-", " ")
    x = re.sub(r"[^A-Za-z0-9\s]", " ", x)
    return " ".join(x.split())


def is_generic_name(name: str) -> bool:
    x = clean_text(name).lower()
    if not x:
        return True
    for pat in GENERIC_NAME_PATTERNS:
        if re.match(pat, x):
            return True
    tokens = re.findall(r"[a-z]+", x)
    if "intent" in tokens and "classification" in tokens:
        generic_vocab = {
            "financial",
            "banking",
            "conversational",
            "conversation",
            "dialogue",
            "service",
            "customer",
            "query",
            "queries",
            "transaction",
            "transactions",
            "intent",
            "classification",
        }
        non_generic = [t for t in tokens if t not in generic_vocab]
        if len(non_generic) == 0:
            return True
    return False


def clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def unique_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def extract_query_text(row: dict[str, Any]) -> str:
    """Extract semantic query text and strip boilerplate routing prompt."""
    instruction = str(row.get("instruction", "") or "")
    input_text = str(row.get("input", "") or "")
    merged = "\n".join([x for x in [instruction, input_text] if x]).strip()
    if not merged:
        return ""

    # Banking77-style router boilerplate removal.
    merged = re.sub(
        r"you are\s+[^.\n]*banking[^.\n]*router[^.\n]*\.?\s*category[^.\n]*conversation\s*",
        "",
        merged,
        flags=re.IGNORECASE,
    )
    merged = re.sub(
        r"classify\s+the\s+banking\s+conversation\s+into\s+the\s+correct\s+intent\s+category\.?",
        "",
        merged,
        flags=re.IGNORECASE,
    )

    lines = [ln.strip() for ln in merged.splitlines() if ln.strip()]
    if lines:
        # Heuristic: user query usually appears near the tail after routing template.
        merged = lines[-1]
    return clean_text(merged)


def extract_keyword_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in re.findall(r"[a-z]{3,}", text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def build_cluster_signature(
    spec: ClusterSpec,
    profile_map: dict[str, dict[str, Any]],
    label_top_k: int = 5,
    keyword_top_k: int = 10,
) -> dict[str, Any]:
    label_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    query_samples: list[str] = []

    matched_count = 0
    for rid in spec.data_ids:
        row = profile_map.get(rid)
        if row is None:
            continue
        matched_count += 1

        output_label = clean_text(row.get("output"))
        if output_label:
            label_counter[output_label] += 1

        query_text = extract_query_text(row)
        if query_text:
            query_samples.append(query_text)
            keyword_counter.update(extract_keyword_tokens(query_text))

    top_labels = label_counter.most_common(label_top_k)
    top_keywords = keyword_counter.most_common(keyword_top_k)
    dominant_label = top_labels[0][0] if top_labels else ""
    dominant_count = int(top_labels[0][1]) if top_labels else 0
    dominant_ratio = (dominant_count / matched_count) if matched_count > 0 else 0.0

    return {
        "matched_count": matched_count,
        "unique_label_count": len(label_counter),
        "dominant_label": dominant_label,
        "dominant_label_count": dominant_count,
        "dominant_label_ratio": round(dominant_ratio, 6),
        "top_labels": [{"label": x, "count": int(y)} for x, y in top_labels],
        "top_keywords": [{"keyword": x, "count": int(y)} for x, y in top_keywords],
        "query_examples": [clip_text(x, 120) for x in query_samples[:4]],
    }


def resolve_embedding_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DenseVectorizer:
    def __init__(self, model_name: str, device: str):
        if SentenceTransformer is None or torch is None:
            raise RuntimeError(
                "sentence-transformers/torch is required for center_vector computation, "
                "but is not installed in current environment."
            )
        self.model = SentenceTransformer(model_name, device=resolve_embedding_device(device))

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=max(1, batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def row_to_embedding_text(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction"))
    inp = clean_text(row.get("input"))
    out = clean_text(row.get("output") or row.get("response"))
    chunks = [x for x in [instruction, inp, out] if x]
    return "\n".join(chunks).strip()


def build_profile_vector_store(
    profile_map: dict[str, dict[str, Any]],
    clusters: list[ClusterSpec],
    cfg: Config,
) -> dict[str, np.ndarray]:
    needed_ids: list[str] = []
    seen: set[str] = set()
    for spec in clusters:
        for rid in spec.data_ids:
            if rid in profile_map and rid not in seen:
                seen.add(rid)
                needed_ids.append(rid)

    if not needed_ids:
        return {}

    logging.info(
        "Building embedding vectors for centroid computation: ids=%d, model=%s, device=%s",
        len(needed_ids),
        cfg.embedding_model,
        resolve_embedding_device(cfg.embedding_device),
    )
    texts = [row_to_embedding_text(profile_map[rid]) for rid in needed_ids]
    vectorizer = DenseVectorizer(cfg.embedding_model, cfg.embedding_device)
    vectors = vectorizer.encode_texts(texts, batch_size=cfg.embedding_batch_size)

    store: dict[str, np.ndarray] = {}
    for rid, vec in zip(needed_ids, vectors):
        store[rid] = vec
    return store


def compute_cluster_center_vector(
    spec: ClusterSpec,
    vector_store: dict[str, np.ndarray],
    precision: int,
) -> list[float]:
    vecs = [vector_store[rid] for rid in spec.data_ids if rid in vector_store]
    if not vecs:
        return []
    center = np.stack(vecs, axis=0).mean(axis=0)
    if precision >= 0:
        center = np.round(center, decimals=precision)
    return [float(x) for x in center.tolist()]


def collect_subtree_ids(node: dict[str, Any]) -> list[str]:
    ids = [str(x) for x in node.get("data_ids", [])]
    for child in node.get("children", []) or []:
        ids.extend(collect_subtree_ids(child))
    return unique_preserve(ids)


def collect_tree_direct_ids(tree: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    stack = [tree]
    while stack:
        node = stack.pop()
        for x in node.get("data_ids", []) or []:
            out.add(str(x))
        stack.extend(node.get("children", []) or [])
    return out


def extract_cluster_specs(
    tree: dict[str, Any],
    scope: str,
    min_cluster_size: int,
    max_clusters: int | None,
) -> list[ClusterSpec]:
    specs: list[ClusterSpec] = []

    def walk(node: dict[str, Any], depth: int) -> None:
        node_id = str(node.get("node_id", f"N_{len(specs)}"))
        children = node.get("children", []) or []
        direct_ids = unique_preserve([str(x) for x in node.get("data_ids", []) or []])
        subtree_ids = collect_subtree_ids(node) if scope == "subtree" else direct_ids
        candidate_ids = subtree_ids

        emit = False
        if scope == "leaf":
            emit = len(children) == 0 and len(candidate_ids) >= min_cluster_size
        elif scope == "data-bearing":
            emit = len(candidate_ids) >= min_cluster_size
        elif scope == "subtree":
            emit = len(candidate_ids) >= min_cluster_size
        else:
            raise ValueError(f"Unsupported scope: {scope}")

        if emit:
            specs.append(
                ClusterSpec(
                    order=len(specs),
                    node_id=node_id,
                    depth=depth,
                    data_ids=candidate_ids,
                    children_count=len(children),
                    subtree_size=int(node.get("subtree_size", len(candidate_ids))),
                    leaf_payload_size=int(node.get("leaf_payload_size", len(direct_ids))),
                )
            )

        for child in children:
            walk(child, depth + 1)

    walk(tree, 0)
    if max_clusters is not None:
        specs = specs[:max_clusters]
    return specs


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in tree file: {path}")
    return data


def load_profile_map(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows: dict[str, dict[str, Any]] = {}
    total_rows = 0
    duplicate_id_rows = 0
    parse_errors = 0

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            raw = line.strip()
            if not raw:
                continue
            total_rows += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            rid = str(row.get("id", idx))
            if rid in rows:
                duplicate_id_rows += 1
                continue
            rows[rid] = row

    stats = {
        "rows_total": total_rows,
        "rows_unique_ids": len(rows),
        "duplicate_id_rows": duplicate_id_rows,
        "parse_error_rows": parse_errors,
    }
    return rows, stats


def sample_representatives(
    spec: ClusterSpec,
    profile_map: dict[str, dict[str, Any]],
    sample_n: int,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    matched = [(rid, profile_map[rid]) for rid in spec.data_ids if rid in profile_map]
    if len(matched) <= sample_n:
        return matched
    rng = random.Random(f"{seed}:{spec.node_id}")
    idxs = sorted(rng.sample(range(len(matched)), sample_n))
    return [matched[i] for i in idxs]


def build_user_prompt(
    spec: ClusterSpec,
    representatives: list[tuple[str, dict[str, Any]]],
    matched_count: int,
    signature: dict[str, Any],
) -> str:
    top_labels = signature.get("top_labels", [])
    top_keywords = signature.get("top_keywords", [])
    dominant_label = signature.get("dominant_label", "")
    dominant_ratio = float(signature.get("dominant_label_ratio", 0.0) or 0.0)
    unique_label_count = int(signature.get("unique_label_count", 0) or 0)

    labels_text = ", ".join(
        f"{humanize_label(str(item.get('label', '')))} ({int(item.get('count', 0))})"
        for item in top_labels
    )
    keywords_text = ", ".join(
        f"{str(item.get('keyword', ''))} ({int(item.get('count', 0))})"
        for item in top_keywords
    )

    lines = [
        "Name this capability cluster from representative instruction-tuning samples.",
        "",
        "Cluster metadata:",
        f"- node_id: {spec.node_id}",
        f"- depth: {spec.depth}",
        f"- cluster_size_tree: {len(spec.data_ids)}",
        f"- matched_profile_count: {matched_count}",
        f"- unique_output_labels: {unique_label_count}",
        f"- dominant_output_label: {humanize_label(str(dominant_label)) if dominant_label else 'N/A'}",
        f"- dominant_output_ratio: {dominant_ratio:.4f}",
        f"- top_output_labels: {labels_text if labels_text else 'N/A'}",
        f"- top_query_keywords: {keywords_text if keywords_text else 'N/A'}",
        "",
        "Representative samples:",
    ]

    for i, (rid, row) in enumerate(representatives, start=1):
        cdt = clean_text(row.get("CDT_description") or row.get("T_description"))
        inst = clean_text(row.get("instruction"))
        inp = clean_text(row.get("input"))
        out = clean_text(row.get("output") or row.get("response"))

        lines.append(f"[Sample {i}] id={rid}")
        if cdt:
            lines.append(f"CDT_description: {clip_text(cdt, 300)}")
        if inst:
            lines.append(f"instruction: {clip_text(inst, 220)}")
        if inp:
            lines.append(f"input: {clip_text(inp, 140)}")
        if out:
            lines.append(f"output: {clip_text(out, 140)}")
        lines.append("")

    lines.extend(
        [
            "Naming constraints:",
            "- Capability name must be specific to this cluster's focus, not generic.",
            "- Avoid generic names like 'Financial Intent Classification' or 'Conversational Intent Classification'.",
            "- If output labels reveal a clear theme, include that theme in the name.",
            "",
            "Return strict JSON only:",
            '{',
            '  "capability_name": "...",',
            '  "capability_name_zh": "...",',
            '  "capability_definition": "..."',
            '}',
        ]
    )
    return "\n".join(lines)


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    def _try_parse(candidate: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    obj = _try_parse(cleaned)
    if obj is not None:
        return obj

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    return _try_parse(match.group(0))


def normalize_name(name: str) -> str:
    x = clean_text(name).strip("\"'` ")
    x = re.sub(r"[^\w\s/&-]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x


def fallback_name_from_representatives(representatives: list[tuple[str, dict[str, Any]]]) -> tuple[str, str]:
    tokens: list[str] = []
    for _, row in representatives:
        text = clean_text(row.get("CDT_description") or row.get("T_description")).lower()
        if not text:
            continue
        for tok in re.findall(r"[a-z]{3,}", text):
            if tok in STOPWORDS:
                continue
            tokens.append(tok)

    if not tokens:
        return "General Instruction Capability", "通用指令能力"

    common = [t for t, _ in Counter(tokens).most_common(2)]
    english = " ".join(x.capitalize() for x in common) + " Capability"
    return english.strip(), "能力簇"


def make_specific_name_from_signature(
    current_name: str,
    signature: dict[str, Any],
) -> str:
    top_labels = signature.get("top_labels", []) or []
    top_keywords = signature.get("top_keywords", []) or []

    if top_labels:
        label = humanize_label(str(top_labels[0].get("label", "")))
        if label:
            return f"{label.title()} Intent Classification"
    if top_keywords:
        keyword = str(top_keywords[0].get("keyword", "")).strip()
        if keyword:
            return f"{keyword.title()} Query Intent Classification"
    return current_name


def label_to_zh_topic(label: str) -> str:
    raw = clean_text(label).lower()
    raw = re.sub(r"[^a-z0-9_]", "", raw)
    if not raw:
        return ""
    if raw in LABEL_ZH_OVERRIDES:
        return LABEL_ZH_OVERRIDES[raw]

    parts = [p for p in raw.split("_") if p]
    mapped: list[str] = []
    mapped_count = 0
    for p in parts:
        if p in TOKEN_ZH_MAP:
            mapped.append(TOKEN_ZH_MAP[p])
            mapped_count += 1
        else:
            mapped.append(p)
    if mapped_count == 0:
        return ""
    return "".join(mapped)


def english_name_to_zh_topic(name: str) -> str:
    raw = clean_text(name).lower()
    raw = re.sub(r"\b(intent|classification|query|queries)\b", " ", raw)
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    parts = [p for p in raw.split() if p]
    mapped: list[str] = []
    mapped_count = 0
    for p in parts:
        if p in TOKEN_ZH_MAP:
            mapped.append(TOKEN_ZH_MAP[p])
            mapped_count += 1
        elif p in {"financial", "banking", "conversation", "conversational", "dialogue", "service"}:
            continue
        else:
            mapped.append(p)
    if mapped_count == 0:
        return ""
    return "".join(mapped)


def derive_synced_zh_name(row: dict[str, Any]) -> str:
    topic = label_to_zh_topic(str(row.get("dominant_label", "")))
    if not topic:
        topic = english_name_to_zh_topic(str(row.get("capability_name", "")))
    if not topic:
        return ""
    if topic.endswith("分类"):
        return topic
    return f"{topic}意图分类"


class DeepSeekClusterNamer:
    def __init__(self, cfg: Config):
        self.api_key = cfg.api_key
        self.model = cfg.model
        self.endpoint = normalize_endpoint(cfg.base_url)
        self.timeout = cfg.timeout
        self.max_retries = cfg.max_retries
        self.retry_base_delay = cfg.retry_base_delay
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self._thread_local = threading.local()

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _call(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._get_session().post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                body = resp.text

                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Retryable HTTP {resp.status_code}: {body[:300]}",
                        response=resp,
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(f"Non-retryable HTTP {resp.status_code}: {body[:500]}")

                data = resp.json()
                content = str(data["choices"][0]["message"]["content"]).strip()
                if not content:
                    raise ValueError("Empty model response content.")
                return content
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise RuntimeError(f"DeepSeek request failed after retries: {exc}") from exc
                backoff = self.retry_base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, 0.25 * backoff)
                time.sleep(backoff + jitter)
        raise RuntimeError("Unexpected retry loop exit.")

    def name_cluster(self, user_prompt: str) -> tuple[dict[str, str], str]:
        raw = self._call(user_prompt)
        obj = parse_json_object(raw)

        if obj is None:
            # Keep content for debug; caller will fallback naming if required.
            raise ValueError(f"Model response is not valid JSON object: {raw[:400]}")

        capability_name = normalize_name(str(obj.get("capability_name", "")))
        capability_name_zh = normalize_name(str(obj.get("capability_name_zh", "")))
        capability_definition = clean_text(obj.get("capability_definition", ""))

        if not capability_name:
            raise ValueError(f"Missing capability_name in model response: {raw[:400]}")

        if not capability_name_zh:
            capability_name_zh = "能力簇"

        return (
            {
                "capability_name": capability_name,
                "capability_name_zh": capability_name_zh,
                "capability_definition": capability_definition,
            },
            raw,
        )


def process_cluster(
    spec: ClusterSpec,
    cfg: Config,
    profile_map: dict[str, dict[str, Any]],
    vector_store: dict[str, np.ndarray],
    namer: DeepSeekClusterNamer,
) -> dict[str, Any]:
    signature = build_cluster_signature(spec=spec, profile_map=profile_map)
    matched_count = sum(1 for rid in spec.data_ids if rid in profile_map)
    missing_count = len(spec.data_ids) - matched_count
    representatives = sample_representatives(
        spec=spec,
        profile_map=profile_map,
        sample_n=cfg.samples_per_cluster,
        seed=cfg.seed,
    )

    result: dict[str, Any] = {
        "order": spec.order,
        "node_id": spec.node_id,
        "depth": spec.depth,
        "cluster_size_tree": len(spec.data_ids),
        "matched_profile_count": matched_count,
        "missing_profile_count": missing_count,
        "children_count": spec.children_count,
        "subtree_size": spec.subtree_size,
        "leaf_payload_size": spec.leaf_payload_size,
        "representative_data_ids": [rid for rid, _ in representatives],
        "label_unique_count": int(signature.get("unique_label_count", 0)),
        "dominant_label": signature.get("dominant_label", ""),
        "dominant_label_ratio": float(signature.get("dominant_label_ratio", 0.0)),
        "label_distribution_topk": signature.get("top_labels", []),
        "query_keywords_topk": signature.get("top_keywords", []),
        "query_examples_topk": signature.get("query_examples", []),
    }

    if cfg.include_center_vector:
        result["center_vector"] = compute_cluster_center_vector(
            spec=spec,
            vector_store=vector_store,
            precision=cfg.center_vector_precision,
        )

    if cfg.include_data_ids:
        result["data_ids"] = list(spec.data_ids)

    if not representatives:
        result["capability_name"], result["capability_name_zh"] = (
            "Unmapped Cluster",
            "未映射簇",
        )
        result["capability_definition"] = (
            "No profile rows matched this cluster's data_ids, so capability name cannot be inferred."
        )
        result["error"] = "No matched profile rows."
        return result

    prompt = build_user_prompt(
        spec=spec,
        representatives=representatives,
        matched_count=matched_count,
        signature=signature,
    )

    try:
        parsed, raw = namer.name_cluster(prompt)
        if is_generic_name(parsed.get("capability_name", "")):
            parsed["capability_name"] = make_specific_name_from_signature(parsed["capability_name"], signature)
            parsed["name_refined_from_signature"] = True
        result.update(parsed)
        if cfg.include_raw_response:
            result["model_raw_response"] = raw
        return result
    except Exception as exc:  # noqa: BLE001
        fallback_en, fallback_zh = fallback_name_from_representatives(representatives)
        result["capability_name"] = fallback_en
        result["capability_name_zh"] = fallback_zh
        result["capability_definition"] = (
            "Fallback name because API call/parsing failed; verify with rerun if needed."
        )
        result["error"] = str(exc)
        return result


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            node_id = str(row.get("node_id", "")).strip()
            if node_id:
                out[node_id] = row
    return out


def refine_generic_names(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        name = normalize_name(str(row.get("capability_name", "")).strip())
        row["capability_name"] = name
        if not is_generic_name(name):
            pass
        else:
            signature = {
                "top_labels": row.get("label_distribution_topk", []),
                "top_keywords": row.get("query_keywords_topk", []),
            }
            refined = make_specific_name_from_signature(name, signature)
            if refined and refined != name:
                row["capability_name"] = refined
                row["name_refined_from_signature"] = True

        current_zh = clean_text(row.get("capability_name_zh", ""))
        needs_sync_zh = (
            (not current_zh)
            or (current_zh in GENERIC_ZH_NAMES)
            or bool(row.get("name_refined_from_signature"))
        )
        if needs_sync_zh:
            synced_zh = derive_synced_zh_name(row)
            if synced_zh:
                row["capability_name_zh"] = synced_zh


def enrich_row_with_signature(
    row: dict[str, Any],
    spec: ClusterSpec,
    profile_map: dict[str, dict[str, Any]],
    cfg: Config,
    vector_store: dict[str, np.ndarray],
) -> None:
    signature = build_cluster_signature(spec=spec, profile_map=profile_map)
    matched_count = int(signature.get("matched_count", 0))
    row["matched_profile_count"] = matched_count
    row["missing_profile_count"] = len(spec.data_ids) - matched_count
    row["label_unique_count"] = int(signature.get("unique_label_count", 0))
    row["dominant_label"] = signature.get("dominant_label", "")
    row["dominant_label_ratio"] = float(signature.get("dominant_label_ratio", 0.0))
    row["label_distribution_topk"] = signature.get("top_labels", [])
    row["query_keywords_topk"] = signature.get("top_keywords", [])
    row["query_examples_topk"] = signature.get("query_examples", [])
    if cfg.include_center_vector:
        has_existing_center = isinstance(row.get("center_vector"), list) and bool(row.get("center_vector"))
        if vector_store:
            computed_center = compute_cluster_center_vector(
                spec=spec,
                vector_store=vector_store,
                precision=cfg.center_vector_precision,
            )
            if computed_center or not has_existing_center:
                row["center_vector"] = computed_center
        elif "center_vector" not in row:
            row["center_vector"] = []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def display_cluster_name(row: dict[str, Any]) -> str:
    zh = clean_text(row.get("capability_name_zh"))
    en = clean_text(row.get("capability_name"))
    if zh and en and zh != en:
        return f"{zh} | {en}"
    if zh:
        return zh
    if en:
        return en
    return "未命名簇"


def build_named_tree_lines(
    tree: dict[str, Any],
    named_rows_by_node: dict[str, dict[str, Any]],
) -> list[str]:
    lines: list[str] = []

    def walk(node: dict[str, Any], prefix: str, is_last: bool, is_root: bool = False) -> None:
        node_id = str(node.get("node_id", "UNKNOWN"))
        named_row = named_rows_by_node.get(node_id)
        if named_row is not None:
            label = f"{display_cluster_name(named_row)} [{node_id}]"
        else:
            label = f"[{node_id}]"

        if is_root:
            lines.append(label)
        else:
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{label}")

        children = node.get("children", []) or []
        next_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            walk(child, next_prefix, i == len(children) - 1, False)

    walk(tree, prefix="", is_last=True, is_root=True)
    return lines


def print_named_tree_to_cli(tree: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    by_node = {
        str(row.get("node_id", "")).strip(): row
        for row in rows
        if str(row.get("node_id", "")).strip()
    }
    lines = build_named_tree_lines(tree=tree, named_rows_by_node=by_node)
    print("\n=== Capability Tree (Named Clusters) ===")
    for ln in lines:
        print(ln)


def build_config(args: argparse.Namespace) -> Config:
    tree_json = args.tree_json
    profile_jsonl = args.profile_jsonl

    output_jsonl = args.output_jsonl
    if output_jsonl is None:
        output_jsonl = tree_json.parent / f"capability_cluster_names_{args.cluster_scope}.jsonl"

    output_summary_json = args.output_summary_json
    if output_summary_json is None:
        output_summary_json = tree_json.parent / f"capability_cluster_names_{args.cluster_scope}_summary.json"

    max_clusters = None if args.max_clusters == -1 else max(1, args.max_clusters)

    return Config(
        tree_json=tree_json,
        profile_jsonl=profile_jsonl,
        output_jsonl=output_jsonl,
        output_summary_json=output_summary_json,
        cluster_scope=args.cluster_scope,
        min_cluster_size=max(1, args.min_cluster_size),
        max_clusters=max_clusters,
        samples_per_cluster=max(1, args.samples_per_cluster),
        include_data_ids=bool(args.include_data_ids),
        include_raw_response=bool(args.include_raw_response),
        seed=args.seed,
        api_key=str(args.api_key or "").strip(),
        model=args.model,
        base_url=args.base_url,
        timeout=max(5, args.timeout),
        concurrency=max(1, args.concurrency),
        max_retries=max(1, args.max_retries),
        retry_base_delay=max(0.1, args.retry_base_delay),
        temperature=max(0.0, float(args.temperature)),
        max_tokens=max(32, args.max_tokens),
        include_center_vector=bool(args.include_center_vector),
        center_vector_precision=args.center_vector_precision,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=max(1, args.embedding_batch_size),
        force_rebuild_center_vector=bool(args.force_rebuild_center_vector),
        resume=bool(args.resume),
        print_named_tree=bool(args.print_named_tree),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cfg = build_config(args)

    if not cfg.tree_json.exists():
        raise FileNotFoundError(f"Tree JSON not found: {cfg.tree_json}")
    if not cfg.profile_jsonl.exists():
        raise FileNotFoundError(f"Profile JSONL not found: {cfg.profile_jsonl}")

    logging.info("Loading tree: %s", cfg.tree_json)
    tree = load_json(cfg.tree_json)
    logging.info("Loading profile: %s", cfg.profile_jsonl)
    profile_map, profile_stats = load_profile_map(cfg.profile_jsonl)

    tree_direct_ids = collect_tree_direct_ids(tree)
    profile_ids = set(profile_map.keys())
    overlap_ids = tree_direct_ids & profile_ids
    only_tree_ids = tree_direct_ids - profile_ids
    only_profile_ids = profile_ids - tree_direct_ids

    logging.info(
        "ID linkage (direct tree data_ids vs profile ids): tree=%d, profile=%d, overlap=%d, tree_only=%d, profile_only=%d",
        len(tree_direct_ids),
        len(profile_ids),
        len(overlap_ids),
        len(only_tree_ids),
        len(only_profile_ids),
    )
    if only_tree_ids:
        logging.warning(
            "There are %d tree IDs not found in profile JSONL. Affected clusters may become 'Unmapped Cluster'.",
            len(only_tree_ids),
        )

    clusters = extract_cluster_specs(
        tree=tree,
        scope=cfg.cluster_scope,
        min_cluster_size=cfg.min_cluster_size,
        max_clusters=cfg.max_clusters,
    )
    if not clusters:
        raise ValueError("No clusters found after filtering. Please adjust --cluster-scope / --min-cluster-size.")

    logging.info(
        "Extracted %d clusters (scope=%s, min_cluster_size=%d).",
        len(clusters),
        cfg.cluster_scope,
        cfg.min_cluster_size,
    )

    existing: dict[str, dict[str, Any]] = {}
    if cfg.resume:
        existing = load_existing_results(cfg.output_jsonl)
        if existing:
            logging.info("Loaded %d existing cluster results from %s", len(existing), cfg.output_jsonl)

    pending = [spec for spec in clusters if spec.node_id not in existing]
    logging.info("Pending clusters to process: %d", len(pending))

    vector_store: dict[str, np.ndarray] = {}
    if cfg.include_center_vector:
        def has_nonempty_center(row: dict[str, Any]) -> bool:
            cv = row.get("center_vector")
            return isinstance(cv, list) and bool(cv)

        need_center_build = bool(
            cfg.force_rebuild_center_vector
            or pending
            or any(not has_nonempty_center(existing.get(spec.node_id, {})) for spec in clusters)
        )
        if need_center_build:
            vector_store = build_profile_vector_store(profile_map=profile_map, clusters=clusters, cfg=cfg)
            if vector_store:
                first_dim = len(next(iter(vector_store.values())))
                logging.info("Centroid vector store ready: ids=%d, dim=%d", len(vector_store), first_dim)
            if cfg.force_rebuild_center_vector:
                logging.info("Force center vector rebuild is enabled.")
        else:
            logging.info("Skip centroid embedding build: all existing rows already have center_vector.")

    if pending and not cfg.api_key:
        raise ValueError(
            "Missing API key. Set --api-key or DEEPSEEK_API_KEY/OPENAI_API_KEY. "
            "No pending clusters can be processed without DeepSeek API."
        )

    namer = DeepSeekClusterNamer(cfg) if pending else None

    results_by_node: dict[str, dict[str, Any]] = dict(existing)
    processed_this_run = 0

    if pending and namer is not None:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures: dict[Future[dict[str, Any]], ClusterSpec] = {
                pool.submit(process_cluster, spec, cfg, profile_map, vector_store, namer): spec for spec in pending
            }

            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="Capability naming")

            for fut in iterator:
                spec = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "order": spec.order,
                        "node_id": spec.node_id,
                        "depth": spec.depth,
                        "cluster_size_tree": len(spec.data_ids),
                        "matched_profile_count": 0,
                        "missing_profile_count": len(spec.data_ids),
                        "children_count": spec.children_count,
                        "subtree_size": spec.subtree_size,
                        "leaf_payload_size": spec.leaf_payload_size,
                        "representative_data_ids": [],
                        "capability_name": "Failed Cluster",
                        "capability_name_zh": "失败簇",
                        "capability_definition": "Unexpected worker failure.",
                        "error": str(exc),
                    }
                    if cfg.include_data_ids:
                        row["data_ids"] = list(spec.data_ids)
                    if cfg.include_center_vector:
                        row["center_vector"] = compute_cluster_center_vector(
                            spec=spec,
                            vector_store=vector_store,
                            precision=cfg.center_vector_precision,
                        )

                results_by_node[spec.node_id] = row
                processed_this_run += 1

    ordered_results: list[dict[str, Any]] = []
    for spec in clusters:
        row = results_by_node.get(spec.node_id)
        if row is None:
            continue
        enrich_row_with_signature(
            row=row,
            spec=spec,
            profile_map=profile_map,
            cfg=cfg,
            vector_store=vector_store,
        )
        ordered_results.append(row)

    if not ordered_results:
        raise ValueError("No output rows generated.")

    refine_generic_names(ordered_results)
    write_jsonl(cfg.output_jsonl, ordered_results)
    logging.info("Saved cluster names: %s (rows=%d)", cfg.output_jsonl, len(ordered_results))

    unresolved = sum(1 for r in ordered_results if r.get("error"))
    mapped = sum(1 for r in ordered_results if str(r.get("capability_name", "")).strip())
    with_match = sum(1 for r in ordered_results if int(r.get("matched_profile_count", 0)) > 0)
    refined_name_count = sum(1 for r in ordered_results if r.get("name_refined_from_signature"))
    center_vector_count = sum(1 for r in ordered_results if isinstance(r.get("center_vector"), list) and r.get("center_vector"))
    center_vector_dim = 0
    for r in ordered_results:
        cv = r.get("center_vector")
        if isinstance(cv, list) and cv:
            center_vector_dim = len(cv)
            break

    summary = {
        "tree_json": str(cfg.tree_json),
        "profile_jsonl": str(cfg.profile_jsonl),
        "cluster_scope": cfg.cluster_scope,
        "min_cluster_size": cfg.min_cluster_size,
        "samples_per_cluster": cfg.samples_per_cluster,
        "cluster_total": len(clusters),
        "cluster_written": len(ordered_results),
        "cluster_processed_this_run": processed_this_run,
        "cluster_with_profile_match": with_match,
        "cluster_with_name": mapped,
        "cluster_with_error": unresolved,
        "cluster_name_refined_count": refined_name_count,
        "cluster_with_center_vector": center_vector_count,
        "center_vector_dim": center_vector_dim,
        "id_linkage": {
            "tree_direct_ids": len(tree_direct_ids),
            "profile_ids": len(profile_ids),
            "overlap_ids": len(overlap_ids),
            "tree_only_ids": len(only_tree_ids),
            "profile_only_ids": len(only_profile_ids),
        },
        "profile_stats": profile_stats,
        "api": {
            "base_url": normalize_endpoint(cfg.base_url),
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "concurrency": cfg.concurrency,
            "max_retries": cfg.max_retries,
            "timeout": cfg.timeout,
        },
        "embedding": {
            "include_center_vector": cfg.include_center_vector,
            "center_vector_precision": cfg.center_vector_precision,
            "embedding_model": cfg.embedding_model,
            "embedding_device": cfg.embedding_device,
            "embedding_batch_size": cfg.embedding_batch_size,
            "force_rebuild_center_vector": cfg.force_rebuild_center_vector,
        },
        "output_jsonl": str(cfg.output_jsonl),
        "generated_at_unix": time.time(),
    }

    cfg.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Saved summary: %s", cfg.output_summary_json)

    if cfg.print_named_tree:
        print_named_tree_to_cli(tree=tree, rows=ordered_results)


if __name__ == "__main__":
    main()
