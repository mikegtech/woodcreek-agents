"""Minimal vLLM OpenAI-compatible client.

Bypasses ChatOpenAI to send clean, None-free payloads directly to vLLM.
Avoids vLLM 400 errors from langchain_openai sending null fields.

Adapted from TruPryce — uses loguru instead of structlog.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger
from openai import AsyncOpenAI

from dacribagents.infrastructure.llm.factory import normalize_base_url


def clean_none(d: dict) -> dict:
    """Remove keys whose value is None."""
    return {k: v for k, v in d.items() if v is not None}


def lc_messages_to_openai(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI-compatible dicts."""
    result: list[dict] = []
    for msg in messages:
        content = msg.content if msg.content is not None else ""

        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            entry: dict = {"role": "assistant", "content": content}
            if msg.tool_calls:
                import json  # noqa: PLC0415

                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"] if isinstance(tc["args"], str) else json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            entry = {"role": "tool", "content": content, "name": name}
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            result.append(entry)
        else:
            result.append({"role": "user", "content": content})

    return result


async def chat_completion(  # noqa: PLR0913
    *,
    base_url: str,
    model: str,
    messages: list[BaseMessage],
    temperature: float,
    max_tokens: int | None = None,
    timeout_s: float = 120.0,
) -> str:
    """Non-streaming chat completion against a vLLM OpenAI-compatible endpoint.

    Sends a minimal, None-free payload to avoid vLLM 400 errors.
    Returns assistant response content as string.
    """
    normalized_url = normalize_base_url(base_url)

    client = AsyncOpenAI(base_url=normalized_url, api_key="not-needed", timeout=timeout_s)

    payload: dict = {
        "model": model,
        "messages": lc_messages_to_openai(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": [],
        "stream": False,
    }
    payload = clean_none(payload)

    try:
        response = await client.chat.completions.create(**payload)
    except Exception as exc:
        logger.error(f"vllm_client error (model={model}, url={normalized_url}): {exc}")
        raise RuntimeError(f"vllm_client error: {exc}") from exc

    return response.choices[0].message.content or ""
