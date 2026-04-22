# prompting/local_llm.py
"""
Calls a local OpenAI-compatible API server (vLLM serving Qwen3.5-2B)
using plain HTTP requests — no openai SDK required.

Single-server mode (default): set LOCAL_LLM_BASE_URL, leave LARGE_LLM_BASE_URL unset.
Dual-server mode: set both. Prompts >= ROUTING_THRESHOLD_TOKENS go to LARGE.
"""
import json
import os
import requests

MODEL_NAME = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3.5-4B")
LOCAL_API_BASE   = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000")
LARGE_API_BASE   = os.environ.get("LARGE_LLM_BASE_URL", "http://localhost:8001") # empty = single-server mode
ROUTING_THRESHOLD_TOKENS = int(os.environ.get("ROUTING_THRESHOLD_TOKENS", "4000"))

_SESSION       = requests.Session()
_LARGE_SESSION = requests.Session()


def _route(messages: list) -> tuple:
    """Return (base_url, model_name, session) for this request."""
    if LARGE_API_BASE:
        estimated_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        if estimated_tokens >= ROUTING_THRESHOLD_TOKENS:
            return LARGE_API_BASE, MODEL_NAME, _LARGE_SESSION
    return LOCAL_API_BASE, MODEL_NAME, _SESSION


def query_local_llm(
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    stop: list = None,
) -> str:
    """
    POST to /v1/chat/completions and return the assistant reply as a string.
    Compatible with any OpenAI-compatible server (vLLM, SGLang, etc.).
    """
    api_base, model_name, session = _route(messages)

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": True
        },
        "thinking_token_budget": 7000,
    }
    if stop:
        payload["stop"] = stop

    url = f"{api_base}/v1/chat/completions"
    try:
        resp = session.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Local LLM server timed out after 600s. "
            f"Is vLLM running at {api_base}?"
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to local LLM server at {api_base}. "
        )
    except KeyError:
        raise RuntimeError(
            f"Unexpected response format from server: {resp.text[:500]}"
        )