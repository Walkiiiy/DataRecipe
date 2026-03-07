"""DeepSeek single-turn chat utility for experiments."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek Chat Completions API.

    Args:
        api_key: DeepSeek API key. If None, reads from ``DEEPSEEK_API_KEY``.
        base_url: Chat completions endpoint.
        model: DeepSeek model name.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature.
        max_tokens: Optional max output tokens.
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/chat/completions"
    model: str = "deepseek-chat"
    timeout: int = 60
    temperature: float = 0.0
    max_tokens: Optional[int] = None


def deepseek_single_turn_chat(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    config: Optional[DeepSeekConfig] = None,
) -> str:
    """Call DeepSeek API with one user turn and return assistant text.

    Args:
        user_prompt: User message content.
        system_prompt: System message content.
        config: Optional API config.

    Returns:
        Assistant response text.

    Raises:
        ValueError: If input or API key is invalid.
        RuntimeError: If API request fails.
    """
    cfg = config if config is not None else DeepSeekConfig()

    if not user_prompt or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string.")

    api_key = cfg.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set config.api_key or env var DEEPSEEK_API_KEY.")

    payload: Dict = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if cfg.max_tokens is not None:
        payload["max_tokens"] = cfg.max_tokens

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request = urllib.request.Request(
        url=cfg.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTPError: {exc.code}, detail={detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek API URLError: {exc}") from exc

    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected API response format: {raw}") from exc

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Assistant response content is empty.")

    return content.strip()


__all__ = ["DeepSeekConfig", "deepseek_single_turn_chat"]
