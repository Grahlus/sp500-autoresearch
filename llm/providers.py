"""OpenAI-compatible MiniMax LLM provider helpers."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - exercised only when dependency missing
    OpenAI = None  # type: ignore[assignment]


DEFAULT_OPENAI_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    text: str
    content_blocks: list[dict[str, Any]]
    raw_response: Any | None = None


class MiniMaxOpenAICompatibleProvider:
    """Minimal wrapper around MiniMax's OpenAI-compatible chat endpoint."""

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
        ensure_minimax_openai_env()
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        self.api_key = api_key or minimax_openai_api_key()
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if OpenAI is None:  # pragma: no cover - dependency issue
            raise RuntimeError("openai SDK is not installed")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for MiniMax calls")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
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
        request_messages = normalize_openai_messages(messages, system=system)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": request_messages,
        }
        if temperature is not None:
            request["temperature"] = normalize_minimax_temperature(temperature)
        if top_p is not None:
            request["top_p"] = top_p
        if tools is not None:
            request["tools"] = normalize_openai_tools(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if stream is not None:
            request["stream"] = stream

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        content_blocks = normalize_openai_assistant_message(message)
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
) -> MiniMaxOpenAICompatibleProvider:
    return MiniMaxOpenAICompatibleProvider(
        provider="minimax",
        model=model,
        api_key=api_key,
        base_url=base_url,
        client=client,
    )


def minimax_openai_api_key() -> str | None:
    """Return the OpenAI-compatible MiniMax key"""
    return os.getenv("OPENAI_API_KEY")


def ensure_minimax_openai_env() -> None:
    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    minimax_openai_api_key()


def normalize_minimax_temperature(temperature: float) -> float:
    return min(1.0, max(0.01, float(temperature)))


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            block_type = str(block.get("type") or "").lower()
            if block_type in ("text", "") and block.get("text") is not None:
                parts.append(str(block["text"]))
    return "".join(parts)


def normalize_openai_messages(
    messages: list[dict[str, Any]],
    *,
    system: str | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if system is not None:
        normalized.append({"role": "system", "content": _content_text(system)})
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            normalized.append(assistant_message_from_blocks(content))
            continue
        item: dict[str, Any] = {"role": role, "content": _content_text(content)}
        if role == "tool" and message.get("tool_call_id"):
            item["tool_call_id"] = message["tool_call_id"]
        normalized.append(item)
    return normalized


def normalize_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            normalized.append(tool)
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or tool.get("input_schema") or {},
                },
            }
        )
    return normalized


def normalize_openai_assistant_message(message: Any) -> list[dict[str, Any]]:
    content_blocks: list[dict[str, Any]] = []
    content = getattr(message, "content", None)
    if content:
        content_blocks.append({"type": "text", "text": str(content)})
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        arguments = getattr(function, "arguments", "") if function is not None else ""
        parsed_arguments: Any = arguments
        try:
            parsed_arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            pass
        content_blocks.append(
            {
                "type": "tool_call",
                "id": getattr(tool_call, "id", None),
                "name": getattr(function, "name", None) if function is not None else None,
                "input": parsed_arguments,
                "function": {
                    "name": getattr(function, "name", None) if function is not None else None,
                    "arguments": arguments,
                },
            }
        )
    return content_blocks


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
    text = extract_text(blocks)
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") != "tool_call":
            continue
        function = block.get("function") if isinstance(block.get("function"), dict) else {}
        arguments = function.get("arguments")
        if arguments is None:
            arguments = json.dumps(block.get("input") or {})
        tool_calls.append(
            {
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": function.get("name") or block.get("name"),
                    "arguments": arguments,
                },
            }
        )
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message

