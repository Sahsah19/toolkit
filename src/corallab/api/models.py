"""Typed results the API layer returns. Every tool input and output is one of these."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.run_manifest import ModelCallConfig, RunManifest
from corallab.schemas.eval.test_case_record import TestCaseRecord
from corallab.schemas.eval.transcript import FidelityVerdict, Transcript
from corallab.schemas.judge.eval_result import EvalResult


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


# ---------------------------------------------------------------------------
# Runs: starting, watching, stopping and deleting
# ---------------------------------------------------------------------------

RunStage = Literal["starting", "sim", "check", "judge", "finished", "failed", "cancelled"]
RunLocation = Literal["results", "sample", "elsewhere"]


class CallCount(_Model):
    """Model calls one role made on one model, counted from a run's trace files."""

    role: str = Field(
        description="system_under_test, person_response, elicitation_fidelity, judge_stage1, "
        "judge_stage2, or the trace's own role name for any other caller"
    )
    model: str = Field(description="The provider model the trace names (resolved LiteLLM model)")
    calls: int = Field(description="Successful responses; a call that raised is not traced")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class StartRunResult(_Model):
    """What ``start_run`` did. Nothing is started when ``problems`` is not empty."""

    started: bool
    run_id: str | None = Field(
        default=None, description="The results folder's name; every other run function takes it"
    )
    results_dir: str | None = Field(
        default=None, description="The folder the CLI created, relative to the data repository"
    )
    pid: int | None = None
    manifest_path: str = Field(description="The manifest asked for")
    run_manifest: str | None = Field(
        default=None,
        description="The manifest the run was started with: the one asked for, or a copy "
        "narrowed to test_case_ids",
    )
    test_case_ids: list[str] | None = None
    command: list[str] = Field(default_factory=list, description="The command started")
    problems: list[str] = Field(default_factory=list, description="Why nothing was started")
    warnings: list[str] = Field(
        default_factory=list, description="Worth a look, but did not stop the start"
    )
    preflight: PreflightReport | None = None


class RunStatus(_Model):
    """How far a run has got, read from its files and its process."""

    run_id: str
    results_dir: str | None = None
    found: bool
    started_by_api: bool = Field(
        description="Started by start_run, so its process and exit code are known"
    )
    pid: int | None = None
    alive: bool | None = Field(description="None when the run was not started by start_run")
    exit_code: int | None = Field(
        default=None, description="None while running, after a cancel, or when unknown"
    )
    cancelled: bool = False
    stage: RunStage | None = Field(
        default=None,
        description="While alive, the stage last written to (from the newest trace file); "
        "after exit, finished, failed or cancelled. None when neither is known",
    )
    batches_total: int | None = None
    batches_simulated: int = Field(description="Batch folders holding a transcript file")
    transcripts_total: int | None = Field(
        default=None, description="Valid test cases in the run's inputs × system_under_test entries"
    )
    transcripts_done: int = Field(
        description="Transcript lines written. A batch's file is written when the whole batch ends"
    )
    checks_done: int = 0
    judged_done: int = 0
    judge_calls_done: int = Field(default=0, description="Judge model calls traced so far")
    calls: list[CallCount] = Field(default_factory=list, description="Every traced call so far")
    errors: list[str] = Field(
        default_factory=list,
        description="ERROR lines of the run log (addresses removed), failed simulations, checks "
        "and judge results, and the end of the process output when it failed",
    )
    started_at: str | None = None
    elapsed_seconds: float | None = None
    problems: list[str] = Field(default_factory=list)


class CancelResult(_Model):
    run_id: str
    cancelled: bool
    was_alive: bool
    signal: str | None = Field(default=None, description="SIGTERM, or SIGKILL when it was needed")
    problems: list[str] = Field(default_factory=list)


class BurrApp(_Model):
    """One application folder of the Burr tracker."""

    app_id: str
    path: str
    reason: str = Field(description="Why it belongs to the run, or why that is not known")


class DeleteRunResult(_Model):
    """What ``delete_run`` removed, or would remove when ``dry_run``."""

    run_id: str
    dry_run: bool
    deleted: bool
    results_dir: str | None = None
    files: list[str] = Field(default_factory=list, description="Files in the results folder")
    bytes: int = 0
    agent_files: list[str] = Field(
        default_factory=list,
        description="start_run's own files for this run: process record, output, narrowed inputs",
    )
    burr_apps: list[BurrApp] = Field(
        default_factory=list, description="Tracker folders attributed to this run"
    )
    burr_unattributed: list[BurrApp] = Field(
        default_factory=list,
        description="Tracker folders whose run could not be read from their own metadata; "
        "never deleted",
    )
    db_rows: dict[str, int] = Field(
        default_factory=dict, description="Rows of the run's apps in each SQLite persister file"
    )
    problems: list[str] = Field(default_factory=list, description="Why nothing was deleted")


class RunListing(_Model):
    """One results folder."""

    run_id: str
    path: str
    location: RunLocation
    created_at: str | None = Field(
        default=None, description="From the -YYYYMMDD-HHMMSS the CLI appends to the folder name"
    )
    manifest_run_id: str | None = None
    evaluation_level: int | None = None
    records_file: str | None = Field(default=None, description="The copied test case file's name")
    system_under_test: list[str] = Field(default_factory=list)
    batches: int = 0
    transcripts: int = 0
    checked: int = 0
    judged: int = 0
    started_by_api: bool = False
    alive: bool | None = None


# ---------------------------------------------------------------------------
# Runs: reading results
# ---------------------------------------------------------------------------


class TokenCount(_Model):
    prompt: int = 0
    completion: int = 0
    cached: int = 0


class StrategyResult(_Model):
    index: int
    adherence: str = Field(description="followed, not_followed or trigger_absent")


class ExpectedBehaviorResult(_Model):
    """The fidelity check's verdict on one expected user behavior."""

    behavior: str
    catalog_occurred: bool
    assertion_occurred: bool
    miss: str | None = None


class GateResult(_Model):
    gate: str
    part_id: str
    label: str
    verdict: str = Field(description="holds, does_not_hold or unknown")
    basis: str = Field(description="test_case, conversation or none")
    conflict: bool


class PathwayResult(_Model):
    """The judge's Stage 2 verdict on one pathway."""

    pathway_id: str
    pathway_did: str
    label: str
    target: bool = Field(description="The pathway the test case was written for")
    support: str = Field(
        description="supported, partially_supported, not_supported or insufficient_evidence"
    )
    gate_status: str = Field(description="met, not_met, unknown or no_gates")
    applies: bool = Field(
        description="A supported instance and every gate holds (met or no_gates): what the "
        "viewer counts as the pathway firing"
    )
    instances: int
    gates: list[GateResult] = Field(default_factory=list)


class BehaviorFound(_Model):
    """A behavior the judge's Stage 1 observed."""

    behavior_id: str
    behavior_did: str
    label: str
    kind: str
    turns: list[int] = Field(default_factory=list)


class TranscriptResult(_Model):
    """One conversation of a test case, with its fidelity check and judge verdicts."""

    transcript_id: str
    system_under_test: str
    sample: int
    batch: str
    exchanges: int
    exit_reason: str | None = None
    sim_failure: str | None = None
    fidelity_status: str | None = Field(
        default=None, description="valid, invalid or not_tested; None when not checked"
    )
    fidelity_source: Literal["check_file", "transcript"] | None = None
    fidelity_error: str | None = None
    strategies: list[StrategyResult] = Field(default_factory=list)
    expected_behaviors: list[ExpectedBehaviorResult] = Field(default_factory=list)
    judge_status: str | None = Field(default=None, description="None when not judged")
    judge_model: str | None = None
    prompt_version: str | None = None
    kl_version: str | None = None
    kl_digest: str | None = None
    pathways: list[PathwayResult] = Field(default_factory=list)
    pathways_skipped: int = 0
    pathways_failed: list[str] = Field(default_factory=list)
    behaviors_found: list[BehaviorFound] = Field(default_factory=list)
    judge_calls: int | None = Field(
        default=None, description="Stage 1 + Stage 2 calls the judge result records"
    )
    judge_retries: int | None = None
    judge_call_failures: int | None = None
    judge_tokens: TokenCount | None = None
    calls: list[CallCount] = Field(
        default_factory=list,
        description="Traced calls tied to this transcript's test case. Sim calls count only "
        "when the batch held this one transcript: their trace rows do not name a test case",
    )


class CaseResult(_Model):
    test_case_id: str
    pathway_id: str | None = Field(default=None, description="From the run's copy of the records")
    transcripts: list[TranscriptResult]


class CatalogUse(_Model):
    """Which Knowledge Library the run's parts name."""

    test_case_pins: list[str] = Field(
        default_factory=list, description="'version digest' the run's test cases pin"
    )
    judge: list[str] = Field(
        default_factory=list, description="'version digest' the judge results record"
    )
    produced_by: list[str] = Field(
        default_factory=list,
        description="'name version' of the library in the records' produced_by",
    )
    loaded: str | None = Field(
        default=None, description="'version digest' loaded now, used for the call estimate"
    )


class CallComparison(_Model):
    """Preflight's call estimate for one role against the calls the run traced."""

    role: str
    estimated: int
    upper_bound: bool
    actual: int
    difference: int = Field(description="actual − estimated")
    basis: str
    note: str = ""


class RunSummary(_Model):
    run_id: str
    path: str
    location: RunLocation
    manifest_run_id: str | None = None
    evaluation_level: int | None = None
    system_under_test: list[str] = Field(default_factory=list)
    judge_models: list[str] = Field(default_factory=list)
    transcripts: int = 0
    checked: int = 0
    judged: int = 0
    fidelity: dict[str, int] = Field(
        default_factory=dict, description="Transcripts per fidelity status ('unchecked' for none)"
    )
    test_cases: list[CaseResult] = Field(default_factory=list)
    calls: list[CallCount] = Field(default_factory=list)
    tokens: TokenCount = Field(default_factory=TokenCount)
    catalog: CatalogUse = Field(default_factory=CatalogUse)
    estimate: list[CallComparison] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    problems: list[str] = Field(
        default_factory=list, description="Files or lines that could not be read"
    )


class TranscriptView(_Model):
    batch: str
    transcript: Transcript
    fidelity: FidelityVerdict | None = Field(
        default=None, description="From the check file, else the transcript's own field"
    )
    fidelity_source: Literal["check_file", "transcript"] | None = None


class TranscriptsResult(_Model):
    run_id: str
    test_case_id: str
    transcripts: list[TranscriptView] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class TranscriptVerdicts(_Model):
    transcript_id: str
    test_case_id: str
    system_under_test: str
    fidelity: FidelityVerdict | None = None
    fidelity_error: str | None = None
    judge: EvalResult | None = Field(
        default=None,
        description="The judge result without its copy of the conversation; raw model "
        "responses only with include_raw",
    )


class VerdictsResult(_Model):
    run_id: str
    verdicts: list[TranscriptVerdicts] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)


class PathwayRate(_Model):
    judged: int = Field(description="Transcripts judged successfully")
    assessed: int = Field(description="Of those, how many had this pathway assessed in Stage 2")
    supported: int = 0
    partially_supported: int = 0
    not_supported: int = 0
    insufficient_evidence: int = 0
    applies: int = 0
    rate: float | None = Field(default=None, description="applies / judged")


class PathwayComparison(_Model):
    pathway_id: str
    label: str
    a: PathwayRate
    b: PathwayRate
    difference: float | None = Field(default=None, description="b.rate − a.rate")


class BehaviorRate(_Model):
    found: int
    judged: int
    rate: float | None = Field(default=None, description="found / judged")


class BehaviorComparison(_Model):
    behavior_id: str
    label: str
    kind: str
    a: BehaviorRate
    b: BehaviorRate
    difference: float | None = Field(default=None, description="b.rate − a.rate")


class RunHeader(_Model):
    run_id: str
    manifest_run_id: str | None = None
    system_under_test: list[str] = Field(default_factory=list)
    judge_models: list[str] = Field(default_factory=list)
    prompt_versions: list[str] = Field(default_factory=list)
    catalog: list[str] = Field(
        default_factory=list, description="'version digest' the judge results record"
    )
    transcripts: int = 0
    judged: int = 0
    counted: int = Field(description="Judged transcripts the rates count")
    fidelity: dict[str, int] = Field(default_factory=dict)


class RunComparison(_Model):
    a: RunHeader
    b: RunHeader
    pathways: list[PathwayComparison] = Field(default_factory=list)
    behaviors: list[BehaviorComparison] = Field(default_factory=list)
    test_cases_only_in_a: list[str] = Field(default_factory=list)
    test_cases_only_in_b: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
