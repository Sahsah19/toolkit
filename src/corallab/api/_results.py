"""Finding a run's results folder and reading its files with the toolkit's own schemas.

The layout is what ``corallab sim``, ``check``, ``judge`` and ``run`` write: transcripts in
``batch-NNN/<stem>-sim.jsonl``, fidelity checks in ``<stem>-check.jsonl``, judge results in
``<stem>-judge.jsonl``, model calls in ``*.trace.jsonl``, and the manifest and test case
records copied into ``inputs/``. A run written before the check had its own file keeps the
verdict on the transcript (``Transcript.fidelity``); both are read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from corallab import paths
from corallab.api import _prompts
from corallab.api._manifests import ROLE_FIDELITY, ROLE_PERSON, ROLE_SUT
from corallab.api.models import CallCount, RunLocation
from corallab.schemas.eval.run_manifest import RunManifest

STATE_DIRNAME = ".agent-runs"
"""``start_run``'s own records, under the results folder. The viewer lists only folders that
hold transcript or judge files, so it skips this one."""

_STAMP = re.compile(r"-(\d{8}-\d{6})(?:-\d+)?$")


def state_dir() -> Path:
    return paths.results_dir() / STATE_DIRNAME


def samples_dir() -> Path:
    return paths.data_dir() / "sample_results"


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def location_of(directory: Path) -> RunLocation:
    if is_within(directory, samples_dir()):
        return "sample"
    if is_within(directory, paths.results_dir()):
        return "results"
    return "elsewhere"


def created_at(run_id: str) -> str | None:
    """The time in the ``-YYYYMMDD-HHMMSS`` the CLI appends to a run folder, ISO 8601."""
    stamp = _STAMP.search(run_id)
    if stamp is None:
        return None
    return datetime.strptime(stamp.group(1), "%Y%m%d-%H%M%S").isoformat()


def looks_like_run(directory: Path) -> bool:
    """A folder the CLI wrote: copied inputs, a batch folder, or a results file at its root."""
    if not directory.is_dir():
        return False
    if (directory / "inputs").is_dir() or any(directory.glob("batch-*")):
        return True
    return any(directory.glob("*-sim.jsonl")) or any(directory.glob("*-judge.jsonl"))


def recorded_dir(run_id: str) -> Path | None:
    """The results folder ``start_run`` recorded for ``run_id``, if it did."""
    record = state_dir() / f"{run_id}.json"
    if not record.is_file():
        return None
    try:
        value = json.loads(record.read_text(encoding="utf-8")).get("results_dir")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return Path(value) if isinstance(value, str) else None


def find_run(run: str | Path) -> Path | None:
    """A run's folder from its id (the folder name) or a path to it.

    An id is looked up in ``start_run``'s records, then ``results/``, then
    ``sample_results/``.
    """
    given = Path(run)
    if given.is_absolute() or len(given.parts) > 1:
        return given if looks_like_run(given) else None
    name = str(run)
    if not name or name.startswith("."):
        return None
    for candidate in (recorded_dir(name), paths.results_dir() / name, samples_dir() / name):
        if candidate is not None and looks_like_run(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@dataclass
class RunFiles:
    """The files of one results folder, each list in batch order."""

    directory: Path
    batches: list[Path] = field(default_factory=list)
    sim: list[Path] = field(default_factory=list)
    check: list[Path] = field(default_factory=list)
    judge: list[Path] = field(default_factory=list)
    traces: list[Path] = field(default_factory=list)
    log: Path | None = None
    inputs_manifest: Path | None = None
    inputs_records: Path | None = None
    manifest_data: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return self.directory.name


def batch_name(path: Path, directory: Path) -> str:
    """``batch-NNN`` for a file in a batch folder; empty for one at the run's root."""
    return "" if path.parent == directory else path.parent.name


def run_files(directory: Path) -> RunFiles:
    files = RunFiles(directory=directory)
    folders = [directory, *sorted(p for p in directory.glob("batch-*") if p.is_dir())]
    files.batches = folders[1:]
    for folder in folders:
        files.sim += sorted(folder.glob("*-sim.jsonl"))
        files.check += sorted(folder.glob("*-check.jsonl"))
        files.judge += sorted(folder.glob("*-judge.jsonl"))
        files.traces += sorted(folder.glob("*.trace.jsonl"))
    log = directory / f"{directory.name}.log"
    files.log = log if log.is_file() else None
    inputs = directory / "inputs"
    if inputs.is_dir():
        files.inputs_manifest, files.manifest_data = _inputs_manifest(inputs)
        records = sorted(p for p in inputs.glob("*.jsonl") if p.is_file())
        wanted = Path(str(files.manifest_data.get("test_case_records_path") or "")).name
        named = [p for p in records if p.name == wanted]
        found = named or records
        files.inputs_records = found[0] if found else None
    return files


def _inputs_manifest(inputs: Path) -> tuple[Path | None, dict[str, Any]]:
    """The manifest copied into ``inputs/``: a YAML or JSON mapping naming a system under test."""
    for path in sorted(inputs.glob("*")):
        if path.suffix not in (".yaml", ".yml", ".json") or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and "system_under_test" in data:
            return path, data
    return None, {}


def run_manifest(files: RunFiles) -> tuple[RunManifest | None, str | None]:
    """The copied manifest, validated, or why it is missing or invalid."""
    if files.inputs_manifest is None:
        return None, "no manifest in inputs/"
    data = dict(files.manifest_data)
    data.setdefault("run_id", "run-generated")
    try:
        return RunManifest.model_validate(data), None
    except ValidationError as exc:
        return None, f"{files.inputs_manifest.name} does not validate: {exc.error_count()} error(s)"


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for raw in handle if raw.strip())
    except OSError:
        return 0


def read_records[M: BaseModel](
    paths_: Iterable[Path], model: type[M], directory: Path
) -> tuple[list[tuple[Path, M]], list[str]]:
    """Every line of each file as ``model``, with the file; a line that does not validate is a
    problem naming its file and line."""
    records: list[tuple[Path, M]] = []
    problems: list[str] = []
    for path in paths_:
        name = path.relative_to(directory).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            problems.append(f"{name}: {exc}")
            continue
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append((path, model.model_validate_json(line)))
            except ValidationError as exc:
                problems.append(
                    f"{name}:{number}: not a {model.__name__} ({exc.error_count()} error(s))"
                )
    return records, problems


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

TRACE_ROLES = {
    "sut": ROLE_SUT,
    "person_response_generator": ROLE_PERSON,
    "elicitation_fidelity": ROLE_FIDELITY,
}


def call_role(trace_role: str) -> str:
    """The preflight role a trace row's role belongs to."""
    if trace_role.startswith("coralmind_s1_"):
        return _prompts.ROLE_JUDGE_STAGE1
    if trace_role.startswith("coralmind_s2_"):
        return _prompts.ROLE_JUDGE_STAGE2
    return TRACE_ROLES.get(trace_role, trace_role)


def is_judge_role(role: str) -> bool:
    return role in (_prompts.ROLE_JUDGE_STAGE1, _prompts.ROLE_JUDGE_STAGE2)


@dataclass(frozen=True)
class TraceCall:
    """One ``llm_call`` row of a trace file."""

    trace: Path
    role: str
    model: str
    test_case_id: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def trace_calls(traces: Iterable[Path]) -> list[TraceCall]:
    """Every model call in ``traces``. Unreadable lines are skipped: a trace still being written
    can end in half a line."""
    calls: list[TraceCall] = []
    for trace in traces:
        try:
            handle = trace.open("rb")
        except OSError:
            continue
        with handle:
            for raw in handle:
                if b'"llm_call"' not in raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("type") != "llm_call":
                    continue
                calls.append(
                    TraceCall(
                        trace=trace,
                        role=call_role(str(row.get("role") or "")),
                        model=str(row.get("resolved_litellm_model") or row.get("model") or ""),
                        test_case_id=row.get("test_case_id") or None,
                        prompt_tokens=_int(row.get("prompt_tokens")),
                        completion_tokens=_int(row.get("completion_tokens")),
                        cached_tokens=_int(row.get("cached_tokens")),
                    )
                )
    return calls


def count_calls(calls: Iterable[TraceCall]) -> list[CallCount]:
    """Calls and tokens per (role, model), in first-seen order."""
    out: dict[tuple[str, str], CallCount] = {}
    for call in calls:
        key = (call.role, call.model)
        row = out.setdefault(key, CallCount(role=call.role, model=call.model, calls=0))
        row.calls += 1
        row.prompt_tokens += call.prompt_tokens
        row.completion_tokens += call.completion_tokens
        row.cached_tokens += call.cached_tokens
    return list(out.values())


def trace_stage(trace: Path) -> str:
    """Which stage writes a trace file, from its name."""
    if trace.name.endswith("-judge.trace.jsonl"):
        return "judge"
    if trace.name.endswith("-check.trace.jsonl"):
        return "check"
    return "sim"
