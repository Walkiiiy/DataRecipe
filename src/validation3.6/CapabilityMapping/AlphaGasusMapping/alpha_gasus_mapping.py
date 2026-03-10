"""AlphaGasus multi-aspect mapping based on shared DeepSeek utility.

Workflow:
1) Find nearest capability anchor by cosine similarity.
2) Use ONE prompt to score four aspects:
   - 相关度 (between current data and nearest anchor)
   - 准确性
   - 能力多样性
   - 难度
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Union

from src.utils import CapabilityAnchorRetriever, DeepSeekConfig, deepseek_single_turn_chat


DataLike = Union[str, Dict]


@dataclass
class LLMConfig:
    """Config for AlphaGasus multi-aspect mapping."""

    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-chat"
    timeout: int = 60
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    anchor_dir: str = (
        "/home/walkiiiy/DataRecipe/data/flan/"
        "niv2_capability_data_ramdom1000_preprocessed/capability_anchors"
    )
    anchor_vector_dim: int = 8192

    normalization_eps: float = 1e-6

    def to_deepseek_config(self) -> DeepSeekConfig:
        return DeepSeekConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class AlphaGasusAccuracyMappingService:
    """AlphaGasus-style multi-aspect scorer with anchor-aware relevance."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config if config is not None else LLMConfig()
        self.retriever = CapabilityAnchorRetriever(
            anchor_dir=self.config.anchor_dir,
            vector_dim=self.config.anchor_vector_dim,
        )

    def score(self, data: DataLike) -> Dict[str, object]:
        """Return four normalized scores in one API call.

        Args:
            data: Current sample (dict or text).

        Returns:
            Dict with keys:
            - ``相关度``
            - ``准确性``
            - ``能力多样性``
            - ``难度``
            - ``能力锚点``
        """
        current_text = self._to_text(data)
        if not current_text.strip():
            raise ValueError("data must not be empty.")

        nearest = self.retriever.retrieve_with_anchor(data)
        anchor_name = nearest["capability_name"]
        anchor_text = nearest["anchor_text"]

        system_prompt, user_prompt = self._build_prompts(
            current_text=current_text,
            anchor_name=anchor_name,
            anchor_text=anchor_text,
        )
        content = deepseek_single_turn_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            config=self.config.to_deepseek_config(),
        )

        scores = self._parse_and_normalize_scores(content)
        scores["能力锚点"] = anchor_name
        return scores

    @staticmethod
    def _to_text(data: DataLike) -> str:
        if isinstance(data, str):
            return data

        instruction = str(data.get("instruction", "")).strip()
        inp = str(data.get("input", "")).strip()
        response = str(data.get("response", "")).strip()

        parts = []
        if instruction:
            parts.append(f"Instruction:\n{instruction}")
        if inp:
            parts.append(f"Input:\n{inp}")
        if response:
            parts.append(f"Response:\n{response}")

        if parts:
            return "\n\n".join(parts)
        return json.dumps(data, ensure_ascii=False)

    def _build_prompts(
        self,
        current_text: str,
        anchor_name: str,
        anchor_text: str,
    ) -> tuple[str, str]:
        system_prompt = (
            "You are a strict AlphaGasus-style evaluator. "
            "You must return only valid JSON with exactly four raw scores."
        )

        user_prompt = (
            "Please evaluate the CURRENT sample in one shot over four aspects.\n\n"
            "Step requirement:\n"
            "1) First use the provided nearest capability anchor as reference for relevance judgment.\n"
            "2) Then score all four aspects on a 0-5 scale (higher is better / stronger).\n"
            "3) Output JSON only, no explanation text.\n\n"
            "CURRENT sample:\n"
            f"{current_text}\n\n"
            "Nearest capability anchor name:\n"
            f"{anchor_name}\n\n"
            "Nearest capability anchor sample:\n"
            f"{anchor_text}\n\n"
            "Aspect definitions:\n"
            "- 相关度: semantic/task relevance between CURRENT sample and nearest anchor sample.\n"
            "- 准确性: response correctness and faithfulness to instruction/input in CURRENT sample.\n"
            "- 能力多样性: diversity of capability signals shown by CURRENT sample.\n"
            "- 难度: task complexity/difficulty level of CURRENT sample.\n\n"
            "Return format (strict JSON):\n"
            '{"raw_scores": {"相关度": r1, "准确性": r2, "能力多样性": r3, "难度": r4}}'
        )
        return system_prompt, user_prompt

    def _parse_and_normalize_scores(self, content: str) -> Dict[str, float]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {content}") from exc

        raw_scores = parsed.get("raw_scores") if isinstance(parsed, dict) else None
        if not isinstance(raw_scores, dict):
            raise ValueError(f"Model output must contain object field 'raw_scores': {parsed}")

        required = ["相关度", "准确性", "能力多样性", "难度"]
        missing = [k for k in required if k not in raw_scores]
        if missing:
            raise ValueError(f"raw_scores missing fields: {missing}")

        eps = self.config.normalization_eps
        if not (0.0 < eps < 0.5):
            raise ValueError("normalization_eps must be in (0, 0.5).")

        out: Dict[str, float] = {}
        for k in required:
            v = raw_scores[k]
            if not isinstance(v, (int, float)):
                raise ValueError(f"raw_scores[{k}] must be numeric, got: {v}")

            raw = min(5.0, max(0.0, float(v)))
            unit = raw / 5.0
            out[k] = unit * (1.0 - 2.0 * eps) + eps

        return out


__all__ = ["LLMConfig", "AlphaGasusAccuracyMappingService"]
