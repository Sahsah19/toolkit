"""Which env vars each model needs, and whether each is set. Values are never read out."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from corallab import paths
from corallab.api._manifests import model_roles, read_run_manifest, resolve_manifest_path
from corallab.api.models import EnvReport, EnvVarStatus, ModelEnvStatus
from corallab.llm.registry_loader import endpoint_env_vars, load_registry
from corallab.llm.registry_schema import LLMRegistry
from corallab.schemas.eval.run_manifest import RunManifest


def load_env_file() -> bool:
    """Load ``.env`` the way the CLI callback does. True when a file was found.

    Existing environment variables win; nothing is printed or returned.
    """
    found = bool(find_dotenv())
    load_dotenv()
    return found


def is_set(name: str) -> bool:
    """Whether ``name`` is set to a non-empty value."""
    return bool(os.environ.get(name))


def model_env_status(registry: LLMRegistry, model_id: str, roles: list[str]) -> ModelEnvStatus:
    """The env vars ``model_id``'s endpoint needs and whether each is set."""
    route = registry.models.get(model_id)
    if route is None:
        return ModelEnvStatus(
            model_id=model_id,
            roles=roles,
            ready=False,
            error=f"{model_id!r} is not in the endpoint registry; declare it under models",
        )
    endpoint = registry.endpoints[route.endpoint]
    statuses = [EnvVarStatus(name=v, is_set=is_set(v)) for v in endpoint_env_vars(endpoint)]
    return ModelEnvStatus(
        model_id=model_id,
        roles=roles,
        endpoint=route.endpoint,
        env_vars=statuses,
        ready=all(s.is_set for s in statuses),
    )


def load_manifest(manifest: RunManifest | str | Path) -> RunManifest:
    """A ``RunManifest`` as given, or read from a path or manifest id.

    Raises ``FileNotFoundError``, ``ValueError`` or ``pydantic.ValidationError``.
    """
    if isinstance(manifest, RunManifest):
        return manifest
    path = resolve_manifest_path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest}")
    data = read_run_manifest(path)
    data.setdefault("run_id", "run-generated")
    return RunManifest.model_validate(data)


def check_env(
    manifest: RunManifest | str | Path | None = None,
    *,
    registry: str | Path | None = None,
    dotenv: bool = True,
) -> EnvReport:
    """Report the env vars each model needs and whether each is set.

    With ``manifest`` (a ``RunManifest``, a path or a manifest id), only the
    models it calls, each with its roles; without, every model in the registry.
    ``dotenv`` loads ``.env`` first, as the CLI does.
    """
    dotenv_loaded = load_env_file() if dotenv else False
    registry_path = Path(registry) if registry is not None else paths.endpoints_registry_path()
    reg = load_registry(registry_path, interpolate=False)
    if manifest is not None:
        wanted = model_roles(load_manifest(manifest))
    else:
        wanted = {model_id: [] for model_id in reg.models}
    models = [model_env_status(reg, model_id, roles) for model_id, roles in wanted.items()]
    missing = sorted({s.name for m in models for s in m.env_vars if not s.is_set})
    return EnvReport(
        dotenv_loaded=dotenv_loaded,
        registry_path=str(registry_path),
        models=models,
        missing=missing,
    )
