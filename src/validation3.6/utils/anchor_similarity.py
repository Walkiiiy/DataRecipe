"""Capability-anchor retrieval by cosine similarity.

Given one data sample, return the capability name of the most similar anchor.
This implementation is dependency-light (pure Python stdlib).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Union


DataLike = Union[str, Dict]


def _tokenize(text: str) -> List[str]:
    # Keep Chinese words and alnum spans; lowercase for stability.
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", text)]


def _record_to_text(data: DataLike) -> str:
    if isinstance(data, str):
        return data

    instruction = str(data.get("instruction", "")).strip()
    inp = str(data.get("input", "")).strip()
    response = str(data.get("response", "")).strip()

    chunks: List[str] = []
    if instruction:
        chunks.append(f"Instruction: {instruction}")
    if inp:
        chunks.append(f"Input: {inp}")
    if response:
        chunks.append(f"Response: {response}")

    if chunks:
        return "\n".join(chunks)

    # Fallback: flatten all fields for unknown schema.
    return " ".join(f"{k}: {v}" for k, v in data.items())


class CapabilityAnchorRetriever:
    """Retrieve nearest capability anchor via cosine similarity.

    Args:
        anchor_dir: Directory of anchor jsonl files.
        vector_dim: Hashing vector size for sparse text representation.

    Notes:
        - Uses a lightweight hashed TF-IDF embedding (pure Python).
        - Similarity is cosine between L2-normalized vectors.
    """

    def __init__(self, anchor_dir: str, vector_dim: int = 8192) -> None:
        self.anchor_dir = Path(anchor_dir)
        self.vector_dim = vector_dim

        if not self.anchor_dir.exists():
            raise ValueError(f"anchor_dir does not exist: {anchor_dir}")

        self._anchors_text: List[str] = []
        self._anchors_capability: List[str] = []

        self._load_anchors()
        if not self._anchors_text:
            raise ValueError(f"No anchor samples found under: {anchor_dir}")

        self.idf = self._build_idf(self._anchors_text)
        self.anchor_matrix = self._encode_texts(self._anchors_text)

    def _load_anchors(self) -> None:
        files = sorted(self.anchor_dir.rglob("*.jsonl"))
        for fp in files:
            fallback_cap = fp.stem
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    text = _record_to_text(ex)

                    cap_name = str(
                        ex.get("能力锚点属性")
                        or ex.get("capability_tag")
                        or ex.get("task_name")
                        or fallback_cap
                    )

                    self._anchors_text.append(text)
                    self._anchors_capability.append(cap_name)

    def _hashed_tf(self, text: str) -> List[float]:
        vec = [0.0] * self.vector_dim
        for tok in _tokenize(text):
            idx = hash(tok) % self.vector_dim
            vec[idx] += 1.0
        return vec

    def _build_idf(self, texts: Sequence[str]) -> List[float]:
        n = len(texts)
        df = [0.0] * self.vector_dim

        for text in texts:
            seen = set()
            for tok in _tokenize(text):
                seen.add(hash(tok) % self.vector_dim)
            for idx in seen:
                df[idx] += 1.0

        return [math.log((n + 1.0) / (d + 1.0)) + 1.0 for d in df]

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def _encode_text(self, text: str) -> List[float]:
        tf = self._hashed_tf(text)
        vec = [tf[i] * self.idf[i] for i in range(self.vector_dim)]
        return self._l2_normalize(vec)

    def _encode_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._encode_text(t) for t in texts]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _best_index(self, data: DataLike) -> int:
        query = self._encode_text(_record_to_text(data))

        best_idx = 0
        best_sim = -1e18
        for i, anchor_vec in enumerate(self.anchor_matrix):
            sim = self._cosine(query, anchor_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx

    def retrieve(self, data: DataLike) -> str:
        """Return capability name of the nearest anchor for given data."""
        best_idx = self._best_index(data)
        return self._anchors_capability[best_idx]

    def retrieve_with_anchor(self, data: DataLike) -> Dict[str, str]:
        """Return nearest capability name and corresponding anchor text."""
        best_idx = self._best_index(data)
        return {
            "capability_name": self._anchors_capability[best_idx],
            "anchor_text": self._anchors_text[best_idx],
        }


__all__ = ["CapabilityAnchorRetriever"]
