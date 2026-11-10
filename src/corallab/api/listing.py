"""List what a run can be built from: manifests, test cases, personas and models.

Each function reads the files through the toolkit's own schemas and returns
summaries. A file or line that does not validate is listed with ``error`` set
rather than dropped, so an agent sees it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from corallab import paths
from corallab.api._manifests import fidelity_model, read_run_manifest, relative_to_data
from corallab.api.models import ManifestSummary, ModelSummary, PersonaSummary, TestCaseSummary
from corallab.llm.registry_loader import endpoint_env_vars, load_registry
from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.test_case_record import TestCaseRecord


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    parts = [f"{'.'.join(map(str, e['loc'])) or '(root)'}: {e['msg']}" for e in errors[:3]]
    more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
    return "; ".join(parts) + more


def _manifest_summary(path: Path) -> ManifestSummary:
    base = {"id": path.stem, "path": relative_to_data(path)}
    try:
        data = read_run_manifest(path)
    except (ValueError, OSError) as exc:
        return ManifestSummary(**base, valid=False, error=str(exc))
    # The CLI generates a run_id when the file has none.
    data.setdefault("run_id", "run-generated")
    try:
        manifest = RunManifest.model_validate(data)
    except ValidationError as exc:
        return ManifestSummary(
            **base,
            valid=False,
            error=_first_error(exc),
            run_id=data.get("run_id") if isinstance(data.get("run_id"), str) else None,
        )
    fidelity = fidelity_model(manifest)
    person = manifest.simulation.person_response_model
    return ManifestSummary(
        **base,
        valid=True,
        run_id=manifest.run_id,
        evaluation_level=manifest.evaluation_level,
        test_case_records_path=manifest.test_case_records_path,
        persona_records_path=manifest.persona_records_path,
        run_dir=manifest.run_dir,
        system_under_test=[cfg.model_id for cfg in manifest.system_under_test],
        person_response_model=person.model_id if person else None,
        fidelity_model=fidelity.model_id if fidelity else None,
        judge_models=[j.model.model_id for j in manifest.judges if j.enabled],
        max_turns=manifest.simulation.max_turns,
        batch_size=manifest.simulation.batch_size,
    )


def list_manifests(directory: str | Path | None = None) -> list[ManifestSummary]:
    """Every ``*.yaml`` run manifest in ``directory`` (default ``run_manifests/``), by name."""
    root = Path(directory) if directory is not None else paths.eval_run_configs_dir()
    return [_manifest_summary(p) for p in sorted(root.glob("*.y*ml"))]


def _test_case_files(path: str | Path | None) -> list[Path]:
    if path is not None:
        resolved = paths.resolve_data_path(path)
        if resolved.is_dir():
            return sorted(p for p in resolved.rglob("*.jsonl") if "schema" not in p.parts)
        return [resolved]
    root = paths.eval_test_cases_dir()
    return sorted(p for p in root.rglob("*.jsonl") if "schema" not in p.relative_to(root).parts)


def read_test_case_summaries(file: Path) -> list[TestCaseSummary]:
    """One summary per non-blank line of a test case records file."""
    rel = relative_to_data(file)
    out: list[TestCaseSummary] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = TestCaseRecord.model_validate_json(line)
        except ValidationError as exc:
            out.append(
                TestCaseSummary(
                    id=f"{rel}:{number}", path=rel, line=number, error=_first_error(exc)
                )
            )
            continue
        out.append(
            TestCaseSummary(
                id=record.test_case_id,
                path=rel,
                line=number,
                display_name=record.display_name,
                evaluation_level=record.evaluation_level,
                pathway_id=record.pathway_id,
                persona_ref=record.persona_ref,
                kl_version=record.kl.version,
                kl_digest=record.kl.digest,
                behaviors=[a.behavior for a in record.behavior_assertions],
                n_scripted_turns=len(record.turns),
                n_strategies=len(record.scenario_strategies),
            )
        )
    return out


def list_test_cases(path: str | Path | None = None) -> list[TestCaseSummary]:
    """Test cases in one records file, a directory of them, or all of ``test_case_records/``.

    ``path`` resolves like a manifest's ``test_case_records_path``. Raises
    ``FileNotFoundError`` when a named file does not exist.
    """
    files = _test_case_files(path)
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Test case records not found: {file}")
    return [summary for file in files for summary in read_test_case_summaries(file)]


def _persona_summary(path: Path) -> PersonaSummary:
    rel = relative_to_data(path)
    try:
        persona = PersonaRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        return PersonaSummary(id=path.stem, path=rel, error=_first_error(exc))
    traits = persona.rendering.traits
    return PersonaSummary(
        id=persona.ref,
        path=rel,
        persona_id=persona.persona_id,
        version=persona.version,
        variant_of=persona.variant_of.persona if persona.variant_of else None,
        age=traits.age,
        gender=traits.gender,
        elevated=[e.value for e in persona.elevated],
    )


def list_personas(directory: str | Path = "persona_records") -> list[PersonaSummary]:
    """Every ``*.json`` persona record directly in ``directory``, resolved like a manifest's."""
    root = paths.resolve_data_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Persona records directory not found: {root}")
    return [_persona_summary(p) for p in sorted(root.glob("*.json"))]


def list_models(registry: str | Path | None = None) -> list[ModelSummary]:
    """Every model id in the endpoint registry, with the env-var names its endpoint needs.

    The registry is read without ``${VAR}`` interpolation, so no secret or
    configured address is resolved.
    """
    path = Path(registry) if registry is not None else paths.endpoints_registry_path()
    reg = load_registry(path, interpolate=False)
    out: list[ModelSummary] = []
    for model_id, route in reg.models.items():
        endpoint = reg.endpoints[route.endpoint]
        literal_base = (
            endpoint.api_base if endpoint.api_base and "${" not in endpoint.api_base else None
        )
        out.append(
            ModelSummary(
                id=model_id,
                endpoint=route.endpoint,
                provider=endpoint.provider,
                litellm_model=route.litellm_model,
                env_vars=endpoint_env_vars(endpoint),
                local=endpoint.provider == "ollama",
                api_base=literal_base,
            )
        )
    return out
