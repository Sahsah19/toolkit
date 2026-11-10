"""Typed results the API layer returns. Every tool input and output is one of these."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.run_manifest import ModelCallConfig, RunManifest
from corallab.schemas.eval.test_case_record import TestCaseRecord


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------


class ManifestSummary(_Model):
    """One run manifest file, read and validated as ``RunManifest``."""

    id: str = Field(description="File name without .yaml; preflight accepts it as manifest_path")
    path: str = Field(description="Path relative to the data repository")
    valid: bool = Field(description="Whether the file validates as RunManifest")
    error: str | None = Field(default=None, description="Why it does not validate")
    run_id: str | None = None
    evaluation_level: int | None = None
    test_case_records_path: str | None = None
    persona_records_path: str | None = None
    run_dir: str | None = None
    system_under_test: list[str] = Field(default_factory=list, description="SUT model ids")
    person_response_model: str | None = None
    fidelity_model: str | None = Field(
        default=None, description="Elicitation fidelity model, after the person-model fallback"
    )
    judge_models: list[str] = Field(default_factory=list, description="Enabled judges' models")
    max_turns: int | None = None
    batch_size: int | None = None


class TestCaseSummary(_Model):
    """One line of a test case records file."""

    __test__ = False

    id: str = Field(description="test_case_id, or '<file>:<line>' for a line that does not parse")
    path: str = Field(description="Records file, relative to the data repository")
    line: int
    error: str | None = Field(default=None, description="Why the line is not a TestCaseRecord")
    display_name: str | None = None
    evaluation_level: int | None = None
    pathway_id: str | None = None
    persona_ref: str | None = None
    kl_version: str | None = None
    kl_digest: str | None = None
    behaviors: list[str] = Field(default_factory=list, description="Asserted behavior ids")
    n_scripted_turns: int | None = None
    n_strategies: int | None = None


class PersonaSummary(_Model):
    """One persona record file."""

    id: str = Field(description="Pinned reference test cases use, 'p-ocd-07@1', or the file stem")
    path: str
    error: str | None = None
    persona_id: str | None = None
    version: int | None = None
    variant_of: str | None = None
    age: int | None = None
    gender: str | None = None
    elevated: list[str] = Field(default_factory=list, description="Elevated context value ids")


class ModelSummary(_Model):
    """One model id in the endpoint registry. Never carries a key or a resolved address."""

    id: str = Field(description="Logical model id, as written in a run manifest")
    endpoint: str
    provider: str
    litellm_model: str
    env_vars: list[str] = Field(description="Env-var names the endpoint needs")
    local: bool = Field(description="Served by a local Ollama server")
    api_base: str | None = Field(
        default=None, description="Only when the registry writes it literally (no ${VAR})"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class EnvVarStatus(_Model):
    name: str
    is_set: bool = Field(description="Set and non-empty. The value is never read out")


class ModelEnvStatus(_Model):
    model_id: str
    roles: list[str] = Field(
        default_factory=list, description="Manifest roles using this model; empty without one"
    )
    endpoint: str | None = Field(default=None, description="None when the id is not registered")
    env_vars: list[EnvVarStatus] = Field(default_factory=list)
    ready: bool = Field(description="Registered and every env var it needs is set")
    error: str | None = None


class EnvReport(_Model):
    dotenv_loaded: bool = Field(description="A .env file was found and loaded (values unread)")
    registry_path: str
    models: list[ModelEnvStatus]
    missing: list[str] = Field(description="Every needed env var that is unset, sorted")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Issue(_Model):
    severity: Severity
    check: str = Field(
        description=(
            "manifest, models, env, catalog, catalog_pin, test_cases, personas, "
            "pathways, fidelity_prompt, judge_options, ollama"
        )
    )
    message: str
    subject: str | None = Field(
        default=None, description="Model id, test case id or persona ref the issue is about"
    )


class CatalogInfo(_Model):
    kl_path: str | None = Field(description="CORALMIND_KL_PATH, or None for the bundled library")
    version: str | None = Field(description="Loaded library version; None when it did not load")
    digest: str | None
    test_case_pins: list[str] = Field(
        default_factory=list, description="Distinct 'version digest' pins the test cases carry"
    )
    matches: bool = Field(description="Every test case pins the loaded library")


class RoleCalls(_Model):
    """Model calls one role makes over the whole run, before retries."""

    role: str = Field(
        description=(
            "system_under_test, person_response, elicitation_fidelity, judge_stage1, judge_stage2"
        )
    )
    model_id: str
    calls: int
    upper_bound: bool = Field(description="True when the real count can be lower")
    basis: str = Field(description="How the number was worked out")


class PromptEstimate(_Model):
    """The largest prompt one role sends a local model, against the context it gets."""

    role: str
    tokens: int = Field(description="Approximate prompt tokens of the largest call in this role")
    context_length: int | None = Field(
        description="Context window (num_ctx) this role's calls get, when known or assumed"
    )
    context_source: str = Field(
        description=(
            "extra_params (num_ctx in the manifest), modelfile, "
            "or assumed (the server default, 4,096 on Ollama 0.40.16)"
        )
    )
    fits: bool | None = Field(description="tokens <= context_length; None when not comparable")
    note: str = Field(description="What the estimate includes and assumes")


class OllamaContext(_Model):
    """A local Ollama model: what the server reports and the prompts each role will send it."""

    model_id: str
    roles: list[str]
    reachable: bool | None = Field(
        description="The server answered /api/show for this model; None when not queried"
    )
    modelfile_num_ctx: int | None = Field(
        default=None, description="num_ctx set in the model's Modelfile parameters"
    )
    loaded_context_length: int | None = Field(
        default=None,
        description=(
            "Context of the model if it is loaded now (/api/ps). For information only: "
            "a call without num_ctx reloads it at the server default"
        ),
    )
    max_context_length: int | None = Field(
        default=None, description="The model's trained maximum, from /api/show model_info"
    )
    prompts: list[PromptEstimate] = Field(default_factory=list)


class PreflightReport(_Model):
    manifest_path: str
    ok: bool = Field(description="No errors. Warnings do not affect it")
    errors: list[Issue] = Field(default_factory=list)
    warnings: list[Issue] = Field(default_factory=list)
    run_id: str | None = None
    evaluation_level: int | None = None
    test_case_count: int | None = None
    catalog: CatalogInfo | None = None
    env: EnvReport | None = None
    calls: list[RoleCalls] = Field(default_factory=list)
    calls_by_model: dict[str, int] = Field(
        default_factory=dict, description="Sum of calls per model id (upper bounds included)"
    )
    ollama: list[OllamaContext] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


class WriteAction(StrEnum):
    CREATED = "created"
    OVERWRITTEN = "overwritten"
    APPENDED = "appended"
    UPDATED = "updated"


class WriteResult(_Model):
    """What a writer did. Nothing is written when ``problems`` is not empty."""

    written: bool
    path: str = Field(description="The file written, or that would have been written")
    action: WriteAction | None = Field(default=None, description="None when nothing was written")
    diff: str = Field(default="", description="Unified diff of the file, before → after")
    problems: list[str] = Field(
        default_factory=list, description="Why nothing was written; empty when it was"
    )
    notes: list[str] = Field(
        default_factory=list, description="Worth a look, but did not stop the write"
    )


ModelChoice = str | ModelCallConfig


class ManifestOverrides(_Model):
    """Changes ``create_manifest`` makes to its template. Unset fields keep the template's value.

    A model given as a bare id becomes ``{model_id: <id>}``: the template's sampling params for
    that role are dropped and the role's default temperature applies. Give a ``ModelCallConfig``
    to set params.
    """

    run_id: str | None = None
    evaluation_level: int | None = Field(default=None, ge=1, le=3)
    system_under_test: list[ModelChoice] | None = Field(
        default=None, min_length=1, description="Replaces the whole list"
    )
    person_response_model: ModelChoice | None = Field(
        default=None, description="simulation.person_response_model"
    )
    fidelity_model: ModelChoice | None = Field(
        default=None, description="simulation.elicitation_fidelity.model"
    )
    judge_model: ModelChoice | None = Field(default=None, description="Every judge's model")
    max_turns: int | None = Field(default=None, ge=1, description="simulation.max_turns")
    batch_size: int | None = Field(default=None, ge=1, description="simulation.batch_size")
    test_case_records_path: str | None = None
    persona_records_path: str | None = None
    run_dir: str | None = None
    set: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Any other field, by dotted path with list indexes, applied last: "
            "{'judges.0.options.max_concurrency': 1}. A value of null removes the key"
        ),
    )


class ManifestWriteResult(WriteResult):
    template_path: str | None = None
    template_diff: str = Field(default="", description="Unified diff, template → new manifest")
    manifest: RunManifest | None = Field(default=None, description="The manifest as validated")
    preflight: PreflightReport | None = Field(
        default=None, description="Preflight on the written file; None when nothing was written"
    )


class ModelAddResult(WriteResult):
    model: ModelSummary | None = None
    env: ModelEnvStatus | None = Field(
        default=None, description="The env vars the model's endpoint needs, and whether each is set"
    )


class BehaviorInfo(_Model):
    """A user behavior a test case must assert, as the catalog defines it."""

    id: str
    label: str
    definition: str | None = None
    source: str = Field(description="'pathway' or 'setup_behaviors'")


class TestCaseWriteResult(WriteResult):
    __test__ = False

    test_case: TestCaseRecord | None = None
    expected_behaviors: list[BehaviorInfo] = Field(
        default_factory=list,
        description="One assertion is needed for each; returned even when nothing was written",
    )


class PersonaWriteResult(WriteResult):
    persona: PersonaRecord | None = None
    changed_paths: list[str] = Field(
        default_factory=list,
        description="For a variant: rendering paths that differ from the base persona",
    )
    base_diff: str = Field(default="", description="For a variant: unified diff, base → this")
