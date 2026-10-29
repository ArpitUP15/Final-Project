"""Shared, guarded Ollama access for every AutoDataGen agent.

Ollama can recommend, explain, and propose candidates. It never supplies numeric
model metrics, bypasses schema checks, or chooses the final model without validation.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any



def _enabled() -> bool:
    return os.getenv("OLLAMA_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()


def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _timeout() -> float:
    try:
        return min(90.0, max(2.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "20"))))
    except ValueError:
        return 20.0


def ollama_status() -> dict[str, Any]:
    """Report local Ollama reachability and configured model availability."""
    enabled = _enabled()
    result: dict[str, Any] = {
        "enabled": enabled, "base_url": _base_url(), "configured_model": _model_name(),
        "available": False, "model_available": False, "models": [],
    }
    if not enabled:
        result["message"] = "Ollama integration is disabled by OLLAMA_ENABLED."
        return result
    request = urllib.request.Request(f"{_base_url()}/api/tags", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=min(4.0, _timeout())) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [str(item.get("name", "")) for item in payload.get("models", []) if item.get("name")]
        result.update(available=True, models=models,
                      model_available=any(name == _model_name() or name.split(":")[0] == _model_name().split(":")[0] for name in models))
        result["message"] = "Ollama is reachable and the configured model is installed." if result["model_available"] else "Ollama is reachable, but the configured model is not installed. Run `ollama pull <model>`."
    except Exception as exc:
        result["message"] = f"Cannot reach Ollama: {type(exc).__name__}. Start the Ollama app/service and verify OLLAMA_BASE_URL."
    return result


def _chat(system: str, user: str, *, json_mode: bool = False, num_predict: int = 384) -> str | None:
    if not _enabled():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama
        kwargs: dict[str, Any] = {
            "model": _model_name(), "base_url": _base_url(), "temperature": 0,
            "num_predict": num_predict, "client_kwargs": {"timeout": _timeout()},
        }
        if json_mode:
            kwargs["format"] = "json"
        model = ChatOllama(**kwargs)
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = response.content
        if isinstance(content, list):
            content = "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        return str(content).strip()
    except Exception:
        # All agents remain functional if the local model is unavailable or times out.
        return None


def ask_json(system: str, payload: dict[str, Any], *, num_predict: int = 384) -> dict[str, Any] | None:
    """Ask Ollama for JSON. Untrusted output is parsed, then callers must validate it."""
    text = _chat(system + " Return only valid JSON, with no markdown fences.",
                 json.dumps(payload, ensure_ascii=False, default=str), json_mode=True, num_predict=num_predict)
    if not text:
        return None
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def ask_text(system: str, payload: dict[str, Any], *, num_predict: int = 256) -> str | None:
    return _chat(system, json.dumps(payload, ensure_ascii=False, default=str), num_predict=num_predict)


def llm_enabled() -> bool:
    return _enabled()
