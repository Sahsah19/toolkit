"""Finding, reading and walking run manifests. Light imports only: the CLI imports this."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from corallab import paths
from corallab.schemas.eval.run_manifest import ModelCallConfig, RunManifest

ROLE_SUT = "system_under_test"
ROLE_PERSON = "person_response"
ROLE_FIDELITY = "elicitation_fidelity"
ROLE_JUDGE = "judge"


def read_run_manifest(path: str | Path) -> dict[str, Any]:
    """Read a run manifest YAML file into a mapping.

    Comments, including the ``yaml-language-server`` schema directive, are ignored.
    """
    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{source} is not a run manifest mapping")
    return loaded


def resolve_manifest_path(value: str | Path) -> Path:
    """Find a manifest from a path or a bare id.

    Tries, in order: the path as given, the path under the data repository
    (``paths.resolve_data_path``), the name under ``run_manifests/``, and the
    name plus ``.yaml`` there. Returns the path as given when none exists.
    """
    given = Path(value)
    candidates = [given, paths.resolve_data_path(given), paths.eval_run_configs_dir() / given]
    if given.suffix not in {".yaml", ".yml"}:
        candidates.append(paths.eval_run_configs_dir() / f"{given}.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return given


def relative_to_data(path: Path) -> str:
    """``path`` relative to the data repository when it lies under it, else as given."""
    try:
        return str(path.resolve().relative_to(paths.data_dir().resolve()))
    except ValueError:
        return str(path)


def fidelity_model(manifest: RunManifest) -> ModelCallConfig | None:
    """The model the elicitation fidelity check calls, or None when it does not run.

    The check runs only at L3 with ``enabled`` true; an unset model reuses
    ``person_response_model``.
    """
    sim = manifest.simulation
    if manifest.evaluation_level < 3 or not sim.elicitation_fidelity.enabled:
        return None
    return sim.elicitation_fidelity.model or sim.person_response_model


def manifest_roles(manifest: RunManifest) -> list[tuple[str, ModelCallConfig]]:
    """Every (role, model) pair the manifest's pipeline calls, in pipeline order."""
    roles: list[tuple[str, ModelCallConfig]] = [
        (ROLE_SUT, cfg) for cfg in manifest.system_under_test
    ]
    person = manifest.simulation.person_response_model
    if manifest.evaluation_level >= 3 and person is not None:
        roles.append((ROLE_PERSON, person))
    fidelity = fidelity_model(manifest)
    if fidelity is not None:
        roles.append((ROLE_FIDELITY, fidelity))
    roles.extend((ROLE_JUDGE, judge.model) for judge in manifest.judges if judge.enabled)
    return roles


def model_roles(manifest: RunManifest) -> dict[str, list[str]]:
    """Model id → the roles that call it, each role listed once."""
    out: dict[str, list[str]] = {}
    for role, cfg in manifest_roles(manifest):
        roles = out.setdefault(cfg.model_id, [])
        if role not in roles:
            roles.append(role)
    return out
