"""Check a run manifest before spending anything on it.

``preflight`` reads the manifest and everything it names with the toolkit's own
loaders and validators, and returns a ``PreflightReport``. It makes no model
calls. The one network call it may make is a metadata query to a local Ollama
server (``query_ollama=False`` turns it off). Problems are reported, never
raised: an error means the run would fail or produce results that do not
count; a warning means it may.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from corallab import paths
from corallab.api import _prompts
from corallab.api._manifests import (
    ROLE_FIDELITY,
    ROLE_PERSON,
    ROLE_SUT,
    fidelity_model,
    model_roles,
    read_run_manifest,
    resolve_manifest_path,
)
from corallab.api._prompts import TranscriptAssumption
from corallab.api.env import check_env
from corallab.api.listing import read_test_case_summaries
from corallab.api.models import (
    CatalogInfo,
    EnvReport,
    Issue,
    OllamaContext,
    PreflightReport,
    PromptEstimate,
    RoleCalls,
    Severity,
    TestCaseSummary,
)
from corallab.api.ollama import Fetch, OllamaModelInfo, http_json, query_ollama
from corallab.llm.registry_loader import load_registry
from corallab.llm.registry_schema import LLMRegistry
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.run_manifest import ModelCallConfig, RunManifest

logger = logging.getLogger(__name__)

#: Context assumed for an Ollama model whose num_ctx is set nowhere preflight can see.
#: Measured on Ollama 0.40.16: with no num_ctx a model loads at 4,096, a 3,891-token
#: prompt is read whole, and a 6,771-token prompt is cut to its last 2,050 tokens.
ASSUMED_OLLAMA_CONTEXT = 4096

_LOADER_KIND_TO_CHECK = {"pin": "catalog_pin", "persona": "personas"}


class _Issues:
    def __init__(self) -> None:
        self.errors: list[Issue] = []
        self.warnings: list[Issue] = []

    def error(self, check: str, message: str, subject: str | None = None) -> None:
        self.errors.append(
            Issue(severity=Severity.ERROR, check=check, message=message, subject=subject)
        )

    def warn(self, check: str, message: str, subject: str | None = None) -> None:
        self.warnings.append(
            Issue(severity=Severity.WARNING, check=check, message=message, subject=subject)
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _load_manifest(path: Path, issues: _Issues) -> RunManifest | None:
    if not path.is_file():
        issues.error("manifest", f"run manifest not found: {path}")
        return None
    try:
        data = read_run_manifest(path)
    except (ValueError, OSError) as exc:
        issues.error("manifest", str(exc))
        return None
    for key in ("test_case_records_path", "run_dir"):
        if key in data and not data[key]:
            issues.error("manifest", f"{key} is empty; the CLI refuses to run")
    # The CLI fills a missing run_id with run-<random>; validate as it would.
    data.setdefault("run_id", "run-generated")
    try:
        return RunManifest.model_validate(data)
    except ValidationError as exc:
        for e in exc.errors():
            where = ".".join(map(str, e["loc"])) or "(manifest)"
            issues.error("manifest", f"{where}: {e['msg']}")
        return None


def _check_prompt_versions(manifest: RunManifest, issues: _Issues) -> None:
    from corallab.fidelity.elicitation import prompt_problem
    from corallab.judging.judges.kl_two_stage import KlTwoStageConfig, prompt_dir

    if fidelity_model(manifest) is not None:
        problem = prompt_problem(manifest)
        if problem:
            issues.error("fidelity_prompt", problem)
    for judge in manifest.judges:
        if not judge.enabled:
            continue
        unknown = sorted(set(judge.options) - set(KlTwoStageConfig.__dataclass_fields__))
        if unknown:
            issues.warn("judge_options", f"the judge ignores unknown options {unknown}")
        try:
            cfg = KlTwoStageConfig.from_options(judge.options)
        except (ValueError, TypeError) as exc:
            issues.error("judge_options", str(exc), judge.model.model_id)
            continue
        if not prompt_dir(cfg.prompt_version).is_dir():
            issues.error(
                "judge_options",
                f"judge prompt_version {cfg.prompt_version!r} is not a prompt directory",
                judge.model.model_id,
            )


# ---------------------------------------------------------------------------
# Models and environment
# ---------------------------------------------------------------------------


def _check_models(manifest: RunManifest, issues: _Issues) -> LLMRegistry | None:
    try:
        registry = load_registry(paths.endpoints_registry_path(), interpolate=False)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        issues.error("models", f"endpoint registry cannot be read: {exc}")
        return None
    for model_id, roles in model_roles(manifest).items():
        if model_id not in registry.models:
            issues.error(
                "models",
                f"{model_id!r} ({', '.join(roles)}) is not in endpoints.yaml; "
                "the run would raise UnknownModelError",
                model_id,
            )
    return registry


def _check_env(manifest: RunManifest, issues: _Issues, *, dotenv: bool) -> EnvReport | None:
    try:
        report = check_env(manifest, dotenv=dotenv)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        issues.error("env", f"cannot check env vars: {exc}")
        return None
    for var in report.missing:
        users = [
            f"{m.model_id} ({', '.join(m.roles)})"
            for m in report.models
            if any(s.name == var and not s.is_set for s in m.env_vars)
        ]
        issues.error("env", f"{var} is not set; needed by {'; '.join(users)}", var)
    return report


# ---------------------------------------------------------------------------
# Catalog, test cases and personas
# ---------------------------------------------------------------------------


def _load_catalog(issues: _Issues) -> Any:
    from corallab.kl import load_library

    kl_path = paths.knowledge_library_path()
    if kl_path is not None and not kl_path.exists():
        issues.error("catalog", f"CORALMIND_KL_PATH names {kl_path}, which does not exist")
        return None
    try:
        return load_library()
    except Exception as exc:  # LibraryParseError and anything coralmap raises on a bad tree
        issues.error("catalog", f"the Knowledge Library did not load: {exc}")
        return None


def _catalog_info(library: Any, summaries: list[TestCaseSummary]) -> CatalogInfo:
    kl_path = paths.knowledge_library_path()
    pins = sorted(
        {f"{s.kl_version} {s.kl_digest}" for s in summaries if s.error is None and s.kl_digest}
    )
    loaded = f"{library.version} {library.digest}" if library is not None else None
    return CatalogInfo(
        kl_path=str(kl_path) if kl_path is not None else None,
        version=library.version if library is not None else None,
        digest=library.digest if library is not None else None,
        test_case_pins=pins,
        matches=loaded is not None and all(p == loaded for p in pins),
    )


def _check_personas(
    refs: dict[str, str], personas_root: Path, issues: _Issues, already: set[str]
) -> None:
    """Load each persona ``refs`` names (test case id → ref) that no check has reported yet."""
    from corallab.simulation.data.test_case_records import load_persona

    seen: dict[str, str | None] = {}
    for tc_id, ref in refs.items():
        if ref not in seen:
            loaded = load_persona(personas_root, ref)
            seen[ref] = loaded if isinstance(loaded, str) else None
        problem = seen[ref]
        if problem and tc_id not in already:
            issues.error("personas", f"{tc_id}: {problem}", ref)


def _load_cases(
    manifest: RunManifest,
    library: Any,
    summaries: list[TestCaseSummary],
    issues: _Issues,
) -> list[RunCase]:
    from corallab.simulation.data.test_case_records import (
        TestCaseRecordsError,
        check_test_case_records,
        load_test_case_records,
    )

    personas_root = paths.resolve_data_path(manifest.persona_records_path)
    if not personas_root.is_dir():
        issues.error("personas", f"persona records directory not found: {personas_root}")
        return []
    valid = {s.id: s.persona_ref for s in summaries if s.error is None and s.persona_ref}
    if library is None:
        _check_personas(valid, personas_root, issues, set())
        return []
    try:
        cases = load_test_case_records(
            manifest.test_case_records_path,
            manifest.persona_records_path,
            library,
            evaluation_level=manifest.evaluation_level,
        )
    except TestCaseRecordsError as exc:
        hint = (
            ""
            if paths.knowledge_library_path()
            else " (CORALMIND_KL_PATH is unset, so the library bundled with coralmap was loaded)"
        )
        reported: set[str] = set()
        for problem in exc.details:
            check = _LOADER_KIND_TO_CHECK.get(problem.kind, "test_cases")
            message = problem.message + (hint if problem.kind == "pin" else "")
            issues.error(check, message, problem.test_case_id)
            if problem.kind == "persona" and problem.test_case_id:
                reported.add(problem.test_case_id)
        # The loader stops at the first problem per test case, so a pin or level
        # mismatch hides a missing persona. Check those personas too.
        _check_personas(valid, personas_root, issues, reported)
        return []
    for message in check_test_case_records(cases, library):
        issues.error("pathways", message, message.partition(":")[0])
    return cases


# ---------------------------------------------------------------------------
# Call estimates
# ---------------------------------------------------------------------------


def estimate_calls(
    manifest: RunManifest, summaries: list[TestCaseSummary], kg: Any | None
) -> list[RoleCalls]:
    """Model calls per role over the whole run, before retries.

    Each ``system_under_test`` entry is its own transcript set, so the person,
    fidelity and judge calls repeat for each. Judge counts need the library
    index ``kg``; without it they are left out.
    """
    valid = [s for s in summaries if s.error is None]
    n = len(valid)
    n_sut = len(manifest.system_under_test)
    level = manifest.evaluation_level
    max_turns = manifest.simulation.max_turns
    rows: list[RoleCalls] = []

    for cfg in manifest.system_under_test:
        if level == 1:
            rows.append(
                RoleCalls(
                    role=ROLE_SUT,
                    model_id=cfg.model_id,
                    calls=n,
                    upper_bound=False,
                    basis=f"{n} test cases × 1 scripted turn",
                )
            )
        elif level == 2:
            scripted = sum(s.n_scripted_turns or 0 for s in valid)
            rows.append(
                RoleCalls(
                    role=ROLE_SUT,
                    model_id=cfg.model_id,
                    calls=scripted,
                    upper_bound=False,
                    basis=f"one per scripted user turn across {n} test cases",
                )
            )
        else:
            rows.append(
                RoleCalls(
                    role=ROLE_SUT,
                    model_id=cfg.model_id,
                    calls=n * max_turns,
                    upper_bound=True,
                    basis=f"{n} test cases × max_turns {max_turns}; a conversation the person "
                    "closes early makes fewer",
                )
            )

    sets = f" × {n_sut} SUT entries" if n_sut > 1 else ""
    person = manifest.simulation.person_response_model
    if level >= 3 and person is not None:
        per_case = max(max_turns - 1, 0)
        rows.append(
            RoleCalls(
                role=ROLE_PERSON,
                model_id=person.model_id,
                calls=n * per_case * n_sut,
                upper_bound=True,
                basis=f"{n} test cases × {per_case} (max_turns − 1; the opening is scripted){sets}",
            )
        )
    fidelity = fidelity_model(manifest)
    if fidelity is not None:
        rows.append(
            RoleCalls(
                role=ROLE_FIDELITY,
                model_id=fidelity.model_id,
                calls=n * n_sut,
                upper_bound=False,
                basis=f"one per transcript{sets}; run by `corallab run` and `corallab check`, "
                "not `corallab sim`",
            )
        )

    if kg is None:
        return rows
    from corallab.judging.judges.kl_two_stage import KlTwoStageConfig, build_definition_groups

    n_pathways = len(kg.pathways())
    for judge in manifest.judges:
        if not judge.enabled:
            continue
        try:
            judge_cfg = KlTwoStageConfig.from_options(judge.options)
        except (ValueError, TypeError):
            continue
        groups = len(build_definition_groups(kg, "ai_behavior", judge_cfg.group_size)) + len(
            build_definition_groups(kg, "user_behavior", judge_cfg.group_size)
        )
        per_call = judge_cfg.pathways_per_call or max(n_pathways, 1)
        stage2 = math.ceil(n_pathways / per_call) if n_pathways else 0
        rows.append(
            RoleCalls(
                role=_prompts.ROLE_JUDGE_STAGE1,
                model_id=judge.model.model_id,
                calls=n * groups * n_sut,
                upper_bound=False,
                basis=f"{n} transcripts × {groups} definition groups at group_size "
                f"{judge_cfg.group_size}{sets}; a transcript with no turns is not judged",
            )
        )
        rows.append(
            RoleCalls(
                role=_prompts.ROLE_JUDGE_STAGE2,
                model_id=judge.model.model_id,
                calls=n * stage2 * n_sut,
                upper_bound=True,
                basis=f"{n} transcripts × {stage2} calls if all {n_pathways} pathways became "
                f"candidates (pathways_per_call {judge_cfg.pathways_per_call}){sets}; only "
                "pathways with a behavior seen in Stage 1 are sent",
            )
        )
    return rows


def _calls_by_model(rows: list[RoleCalls]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row.model_id] = out.get(row.model_id, 0) + row.calls
    return out


# ---------------------------------------------------------------------------
# Ollama context
# ---------------------------------------------------------------------------


def _effective_context(cfg: ModelCallConfig, info: OllamaModelInfo) -> tuple[int, str]:
    num_ctx = cfg.extra_params.get("num_ctx")
    if isinstance(num_ctx, int):
        return num_ctx, "extra_params"
    if info.modelfile_num_ctx:
        return info.modelfile_num_ctx, "modelfile"
    if info.loaded_context_length:
        return info.loaded_context_length, "loaded"
    return ASSUMED_OLLAMA_CONTEXT, "assumed"


def suggested_num_ctx(tokens: int, max_context: int | None) -> int:
    """A power of two with a quarter's headroom over ``tokens``, capped at ``max_context``."""
    target = 1 << math.ceil(math.log2(max(tokens * 1.25, 2048)))
    return min(target, max_context) if max_context else target


def _role_configs(manifest: RunManifest, model_id: str) -> list[tuple[str, ModelCallConfig, Any]]:
    """(role, config, judge or None) for every role that calls ``model_id``."""
    out: list[tuple[str, ModelCallConfig, Any]] = [
        (ROLE_SUT, cfg, None) for cfg in manifest.system_under_test if cfg.model_id == model_id
    ]
    person = manifest.simulation.person_response_model
    if manifest.evaluation_level >= 3 and person is not None and person.model_id == model_id:
        out.append((ROLE_PERSON, person, None))
    fidelity = fidelity_model(manifest)
    if fidelity is not None and fidelity.model_id == model_id:
        out.append((ROLE_FIDELITY, fidelity, None))
    out.extend(
        ("judge", judge.model, judge)
        for judge in manifest.judges
        if judge.enabled and judge.model.model_id == model_id
    )
    return out


def _prompt_sizes(
    manifest: RunManifest,
    role: str,
    judge: Any,
    cases: list[RunCase],
    kg: Any,
    litellm_model: str,
    assumption: TranscriptAssumption,
) -> list[_prompts.PromptSize]:
    if role == ROLE_SUT:
        return [_prompts.sut_prompt(manifest, cases, litellm_model, assumption)]
    if role == ROLE_PERSON:
        return [_prompts.person_prompt(manifest, cases, litellm_model, assumption)]
    if role == ROLE_FIDELITY:
        return [_prompts.fidelity_prompt(manifest, cases, litellm_model, assumption)]
    if kg is None:
        return []
    return _prompts.judge_prompts(manifest, judge, cases, kg, litellm_model, assumption)


def _check_ollama(
    manifest: RunManifest,
    registry: LLMRegistry,
    cases: list[RunCase],
    kg: Any,
    issues: _Issues,
    *,
    query: bool,
    timeout: float,
    fetch: Fetch,
    assumption: TranscriptAssumption,
) -> list[OllamaContext]:
    out: list[OllamaContext] = []
    for model_id, roles in model_roles(manifest).items():
        route = registry.models.get(model_id)
        if route is None or registry.endpoints[route.endpoint].provider != "ollama":
            continue
        api_base = registry.endpoints[route.endpoint].api_base
        if api_base and "${" in api_base:
            api_base = None  # never resolve an env-var address here; use the default
        if query:
            info = query_ollama(api_base, route.litellm_model, timeout=timeout, fetch=fetch)
            if not info.reachable:
                issues.warn(
                    "ollama",
                    f"Ollama did not answer for {model_id} ({route.litellm_model}); start the "
                    "server and pull the model before running",
                    model_id,
                )
        else:
            info = OllamaModelInfo(reachable=False)
        estimates: list[PromptEstimate] = []
        for role, cfg, judge in _role_configs(manifest, model_id):
            if not cases:
                break
            try:
                sizes = _prompt_sizes(
                    manifest, role, judge, cases, kg, route.litellm_model, assumption
                )
            except Exception as exc:  # an estimate must never stop the report
                issues.warn("ollama", f"could not size {role} prompts: {exc}", model_id)
                continue
            context, source = _effective_context(cfg, info)
            for size in sizes:
                fits = size.tokens <= context
                estimates.append(
                    PromptEstimate(
                        role=size.role,
                        tokens=size.tokens,
                        context_length=context,
                        context_source=source,
                        fits=fits,
                        note=size.note,
                    )
                )
                if not fits:
                    suggestion = suggested_num_ctx(size.tokens, info.max_context_length)
                    issues.warn(
                        "ollama",
                        f"{size.role} prompts reach ~{size.tokens:,} tokens but {model_id} gets "
                        f"a context of {context:,} ({source}); Ollama cuts a prompt that does "
                        "not fit to its last half-context of tokens, dropping the start. Set "
                        f"extra_params: {{num_ctx: {suggestion}}} on this model in the manifest",
                        model_id,
                    )
                if info.max_context_length and size.tokens > info.max_context_length:
                    issues.warn(
                        "ollama",
                        f"{size.role} prompts reach ~{size.tokens:,} tokens, over {model_id}'s "
                        f"trained maximum of {info.max_context_length:,}",
                        model_id,
                    )
        out.append(
            OllamaContext(
                model_id=model_id,
                roles=roles,
                reachable=info.reachable if query else None,
                modelfile_num_ctx=info.modelfile_num_ctx,
                loaded_context_length=info.loaded_context_length,
                max_context_length=info.max_context_length,
                prompts=estimates,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def preflight(
    manifest_path: str | Path,
    *,
    query_ollama: bool = True,
    ollama_timeout: float = 2.0,
    transcript: TranscriptAssumption | None = None,
    dotenv: bool = True,
    fetch: Fetch = http_json,
) -> PreflightReport:
    """Check everything a run manifest needs and estimate its model calls. Never raises
    for a problem in the inputs; each becomes an error or a warning in the report.

    ``manifest_path`` is a path, a path under the data repository, or a manifest id
    (``l3_local``). ``dotenv`` loads ``.env`` as the CLI does.
    ``transcript`` sets the tokens per synthetic message used to size prompts.
    """
    issues = _Issues()
    path = resolve_manifest_path(manifest_path)
    report = PreflightReport(manifest_path=str(path), ok=False)
    manifest = _load_manifest(path, issues)
    if manifest is None:
        return report.model_copy(update={"errors": issues.errors, "warnings": issues.warnings})

    _check_prompt_versions(manifest, issues)
    registry = _check_models(manifest, issues)
    env = _check_env(manifest, issues, dotenv=dotenv)

    library = _load_catalog(issues)
    records_file = paths.resolve_data_path(manifest.test_case_records_path)
    summaries: list[TestCaseSummary] = []
    cases: list[RunCase] = []
    if not records_file.is_file():
        issues.error("test_cases", f"test case records not found: {records_file}")
    else:
        summaries = read_test_case_summaries(records_file)
        if not any(s.error is None for s in summaries):
            issues.error("test_cases", f"{records_file} holds no valid test case")
        cases = _load_cases(manifest, library, summaries, issues)

    kg = None
    if library is not None:
        from corallab.judging.kl_index import LibraryIndex

        kg = LibraryIndex(library)
    calls = estimate_calls(manifest, summaries, kg)
    if kg is None:
        issues.warn("catalog", "judge calls are not estimated: the library did not load")

    ollama: list[OllamaContext] = []
    if registry is not None:
        ollama = _check_ollama(
            manifest,
            registry,
            cases,
            kg,
            issues,
            query=query_ollama,
            timeout=ollama_timeout,
            fetch=fetch,
            assumption=transcript or TranscriptAssumption(),
        )
        if ollama and not cases:
            issues.warn("ollama", "prompt sizes are not estimated: the test cases did not load")

    return PreflightReport(
        manifest_path=str(path),
        ok=not issues.errors,
        errors=issues.errors,
        warnings=issues.warnings,
        run_id=manifest.run_id,
        evaluation_level=manifest.evaluation_level,
        test_case_count=sum(1 for s in summaries if s.error is None),
        catalog=_catalog_info(library, summaries),
        env=env,
        calls=calls,
        calls_by_model=_calls_by_model(calls),
        ollama=ollama,
    )
