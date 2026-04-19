"""Anthropic-compatible LLM provider helpers.

This repo uses the Anthropic SDK against MiniMax's Anthropic-compatible
endpoint. The provider keeps the full assistant content blocks so multi-turn
tool flows can be replayed without losing non-text blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - exercised only when dependency missing
    Anthropic = None  # type: ignore[assignment]


DEFAULT_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    text: str
    content_blocks: list[dict[str, Any]]
    raw_response: Any | None = None


class AnthropicCompatibleProvider:
    """Minimal wrapper around the Anthropic SDK for compatible endpoints."""

    def __init__(
        self,
        *,
        provider: str = "minimax",
        model: str = DEFAULT_MINIMAX_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url or os.getenv("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if Anthropic is None:  # pragma: no cover - dependency issue
            raise RuntimeError("anthropic SDK is not installed")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for MiniMax calls")
        self._client = Anthropic(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        stream: bool | None = None,
        **_: Any,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if metadata is not None:
            request["metadata"] = metadata
        if thinking is not None:
            request["thinking"] = thinking
        if stream is not None:
            request["stream"] = stream

        response = self.client.messages.create(**request)
        content_blocks = normalize_content_blocks(getattr(response, "content", []))
        return LLMResponse(
            provider=self.provider,
            model=self.model,
            text=extract_text(content_blocks),
            content_blocks=content_blocks,
            raw_response=response,
        )


def build_minimax_provider(
    *,
    model: str = DEFAULT_MINIMAX_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    client: Any | None = None,
) -> AnthropicCompatibleProvider:
    return AnthropicCompatibleProvider(
        provider="minimax",
        model=model,
        api_key=api_key,
        base_url=base_url,
        client=client,
    )


def normalize_content_blocks(blocks: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for block in blocks or []:
        if isinstance(block, dict):
            normalized.append({str(key): value for key, value in block.items()})
            continue
        if hasattr(block, "model_dump"):
            try:
                normalized.append(block.model_dump())
                continue
            except Exception:
                pass
        item: dict[str, Any] = {}
        for key in ("type", "text", "id", "name", "input", "thinking", "index"):
            if hasattr(block, key):
                value = getattr(block, key)
                if value is not None:
                    item[key] = value
        if not item:
            item["text"] = str(block)
        normalized.append(item)
    return normalized


def extract_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if str(block.get("type") or "").lower() == "text" and block.get("text") is not None:
            parts.append(str(block.get("text")))
    return "".join(parts).strip()


def assistant_message_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "assistant", "content": blocks}
