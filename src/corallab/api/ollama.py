"""Read a local Ollama server's metadata for a model. No generation, no cost.

Calls ``POST /api/show`` (Modelfile parameters and model_info) and
``GET /api/ps`` (models loaded now). Both are metadata endpoints.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE = "http://localhost:11434"

#: ``fetch(url, body, timeout)`` returns the decoded JSON object, or None on any failure.
Fetch = Callable[[str, dict[str, Any] | None, float], dict[str, Any] | None]


def http_json(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any] | None:
    """GET (no body) or POST JSON to ``url``; the decoded object, or None on any failure."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("ollama: %s failed: %s", url, exc)
        return None
    return decoded if isinstance(decoded, dict) else None


@dataclass(frozen=True)
class OllamaModelInfo:
    reachable: bool
    modelfile_num_ctx: int | None = None
    loaded_context_length: int | None = None
    max_context_length: int | None = None


def ollama_model_name(litellm_model: str) -> str:
    """``ollama_chat/llama3.2`` → ``llama3.2``."""
    _prefix, sep, name = litellm_model.partition("/")
    return name if sep else litellm_model


def _modelfile_num_ctx(parameters: Any) -> int | None:
    if not isinstance(parameters, str):
        return None
    for line in parameters.splitlines():
        key, _, value = line.strip().partition(" ")
        if key == "num_ctx":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def _max_context(model_info: Any) -> int | None:
    if not isinstance(model_info, dict):
        return None
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def _same_model(listed: str, name: str) -> bool:
    return listed == name or listed == f"{name}:latest" or name == f"{listed}:latest"


def _loaded_context(ps: dict[str, Any] | None, name: str) -> int | None:
    if not ps:
        return None
    for entry in ps.get("models") or []:
        if not isinstance(entry, dict):
            continue
        listed = entry.get("name") or entry.get("model") or ""
        if _same_model(str(listed), name) and isinstance(entry.get("context_length"), int):
            return int(entry["context_length"])
    return None


def query_ollama(
    api_base: str | None,
    model: str,
    *,
    timeout: float = 2.0,
    fetch: Fetch = http_json,
) -> OllamaModelInfo:
    """Metadata for ``model`` (an Ollama name or a LiteLLM ``ollama_chat/...`` string)."""
    base = (api_base or DEFAULT_OLLAMA_BASE).rstrip("/")
    name = ollama_model_name(model)
    show = fetch(f"{base}/api/show", {"model": name}, timeout)
    if show is None:
        return OllamaModelInfo(reachable=False)
    ps = fetch(f"{base}/api/ps", None, timeout)
    return OllamaModelInfo(
        reachable=True,
        modelfile_num_ctx=_modelfile_num_ctx(show.get("parameters")),
        loaded_context_length=_loaded_context(ps, name),
        max_context_length=_max_context(show.get("model_info")),
    )
