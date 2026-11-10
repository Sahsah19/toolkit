"""Load + ${ENV_VAR}-interpolate an LLM endpoint registry from YAML.

Keeps the registry YAML safe to commit by deferring all secret lookups
until the file is read. Missing env vars resolve to ``None`` so the
``ModelRouter`` can still construct (and surface a clear error only when a
model on that endpoint is actually used).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from corallab.llm.registry_schema import EndpointConfig, LLMRegistry

logger = logging.getLogger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# Foundry's portal copies .../openai/v1. LiteLLM's azure/ client then appends
# /openai/deployments/{model}/..., which 404s. The resource root is what
# azure/ expects. openai/ and azure_ai/ bases keep the suffix.
_AZURE_FOUNDRY_V1_SUFFIX = "/openai/v1"


def _interpolate(value: Any) -> Any:
    """Replace any ``${VAR}`` token in a string with ``os.environ[VAR]``.

    Returns ``None`` when the entire string is a single ``${VAR}`` token and
    the env var is unset — distinguishes "user wanted a value but it's
    missing" from "literal empty string." Mixed strings ('${HOST}/api') with
    a missing var raise ``KeyError`` because the result would be malformed.
    Non-string values pass through unchanged.
    """
    if not isinstance(value, str):
        return value
    match = _ENV_PATTERN.fullmatch(value.strip())
    if match is not None:
        var = match.group(1)
        return os.environ.get(var)
    if "${" not in value:
        return value

    def _sub(m: re.Match[str]) -> str:
        var = m.group(1)
        env = os.environ.get(var)
        if env is None:
            raise KeyError(f"Environment variable {var!r} referenced in endpoints.yaml is not set")
        return env

    return _ENV_PATTERN.sub(_sub, value)


def _strip_azure_foundry_v1_suffix(api_base: Any, *, provider: Any) -> Any:
    """Drop a trailing /openai/v1 from Azure OpenAI resource roots.

    Returns ``api_base`` unchanged for non-azure providers and for values
    that are not a string ending in that suffix.
    """
    if provider != "azure" or not isinstance(api_base, str):
        return api_base
    trimmed = api_base.rstrip("/")
    if not trimmed.endswith(_AZURE_FOUNDRY_V1_SUFFIX):
        return api_base
    resource_root = trimmed[: -len(_AZURE_FOUNDRY_V1_SUFFIX)]
    logger.warning(
        "Stripped /openai/v1 from Azure api_base %s → %s. "
        "LiteLLM's azure/ client appends /openai/deployments/... itself; "
        "the Foundry portal URL 404s if left in place.",
        api_base,
        resource_root,
    )
    return resource_root


def _resolve_endpoint(raw: dict[str, Any]) -> EndpointConfig:
    """Build an EndpointConfig with ${VAR} interpolation applied to api_base."""
    resolved = dict(raw)
    if "api_base" in resolved:
        resolved["api_base"] = _strip_azure_foundry_v1_suffix(
            _interpolate(resolved["api_base"]),
            provider=resolved.get("provider"),
        )
    return EndpointConfig.model_validate(resolved)


def load_registry(path: Path, *, interpolate: bool = True) -> LLMRegistry:
    """Load and validate an endpoints.yaml file.

    With ``interpolate=False`` each ``api_base`` keeps its ``${VAR}`` tokens, so
    nothing is read from the environment: use it to inspect the registry's
    shape and the env-var names it needs.

    Raises ``FileNotFoundError`` when the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"LLM registry not found at {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"endpoints.yaml at {path} must be a mapping at the top level")

    raw_endpoints = raw.get("endpoints", {}) or {}
    raw_models = raw.get("models", {}) or {}
    extra = set(raw) - {"endpoints", "models"}
    if extra:
        raise ValueError(f"endpoints.yaml at {path} has unknown top-level keys: {sorted(extra)}")

    endpoints = {
        name: _resolve_endpoint(cfg) if interpolate else EndpointConfig.model_validate(cfg)
        for name, cfg in raw_endpoints.items()
    }
    registry = LLMRegistry.model_validate({"endpoints": endpoints, "models": raw_models})

    unknown_endpoint_refs = [
        f"{model_id}->{model.endpoint}"
        for model_id, model in registry.models.items()
        if model.endpoint not in registry.endpoints
    ]
    if unknown_endpoint_refs:
        raise ValueError(
            "endpoints.yaml: model entries reference unknown endpoints: "
            + ", ".join(unknown_endpoint_refs)
        )

    return registry


def resolve_api_key(endpoint: EndpointConfig) -> str | None:
    """Look up the API key from the env var named in ``api_key_env``."""
    if endpoint.api_key_env is None:
        return None
    return os.environ.get(endpoint.api_key_env)


def endpoint_env_vars(endpoint: EndpointConfig) -> list[str]:
    """Env-var names an endpoint needs: ``api_key_env`` and ``${VAR}`` tokens in ``api_base``.

    Read from an endpoint loaded with ``interpolate=False``; an interpolated
    ``api_base`` no longer holds its tokens.
    """
    names: list[str] = []
    if endpoint.api_key_env:
        names.append(endpoint.api_key_env)
    for var in _ENV_PATTERN.findall(endpoint.api_base or ""):
        if var not in names:
            names.append(var)
    return names
