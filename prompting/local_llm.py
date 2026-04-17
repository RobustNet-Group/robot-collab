# prompting/local_llm.py
"""
Calls a local OpenAI-compatible API server (vLLM serving Qwen3.5-2B)
using plain HTTP requests — no openai SDK required.
"""
import json
import os
import requests

LOCAL_API_BASE = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000")
LOCAL_MODEL_NAME = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3.5-4B")

_SESSION = requests.Session()  # reuse TCP connection across calls


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

    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "presence_penalty": 1.5,
        "top_k": 20,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    if stop:
        payload["stop"] = stop

    url = f"{LOCAL_API_BASE}/v1/chat/completions"
    try:
        resp = _SESSION.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300,   # generous timeout; small model is fast but sim step may queue
        )
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Local LLM server timed out after 120s. "
            f"Is vLLM running at {LOCAL_API_BASE}?"
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to local LLM server at {LOCAL_API_BASE}. "
            f"Start it with: vllm serve Qwen/Qwen3.5-2B --port 8000 "
            f"--language-model-only --max-model-len 8192"
        )
    except KeyError:
        raise RuntimeError(
            f"Unexpected response format from server: {resp.text[:500]}"
        )
