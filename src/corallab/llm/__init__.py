"""LLM endpoint registry + ModelRouter wrapping LiteLLM's ``Router``.

Public surface:

- ``EndpointConfig``, ``ModelRoute``, ``LLMRegistry`` — Pydantic schemas
  for ``data-repository/endpoints/endpoints.yaml``.
- ``load_registry(path)`` — parse + ``${ENV_VAR}`` interpolation.
- ``ResolvedModel`` — resolution result a single ``LLMClient.complete()``
  call needs to stamp the trace + forward to LiteLLM.
- ``ModelRouter`` — wraps ``litellm.Router`` so the same ``model_id``
  string used in ``RunManifest`` can route to a specific endpoint with
  per-endpoint credentials.

``LLMClient`` (`corallab.llm_client`) resolves every call through the
active router. A model id that is not registered raises
``UnknownModelError``.
"""

from __future__ import annotations

from corallab.llm.registry_loader import endpoint_env_vars, load_registry
from corallab.llm.registry_schema import EndpointConfig, LLMRegistry, ModelRoute
from corallab.llm.router import (
    ModelRouter,
    ResolvedModel,
    UnknownModelError,
    get_router,
    set_router,
)

__all__ = [
    "EndpointConfig",
    "LLMRegistry",
    "ModelRoute",
    "ModelRouter",
    "ResolvedModel",
    "UnknownModelError",
    "endpoint_env_vars",
    "get_router",
    "load_registry",
    "set_router",
]
