"""Start, watch, stop, list and delete runs of ``corallab run``.

A run's id is the name of the results folder the CLI creates: the manifest's ``run_dir`` with
``-YYYYMMDD-HHMMSS`` (and ``-2``, ``-3`` on a collision) appended. ``start_run`` finds the
folder the CLI really created and records the process in ``results/.agent-runs/<run_id>.json``,
so a later process can read its status, cancel it or delete it. Runs started some other way are
read from their files alone.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from corallab import paths
from corallab.api._burr import (
    attribute,
    default_burr_dir,
    delete_persister_rows,
    named_app_pattern,
    persister_app_ids,
    persister_rows,
)
from corallab.api._manifests import read_run_manifest, relative_to_data, resolve_manifest_path
from corallab.api._results import (
    count_calls,
    count_lines,
    created_at,
    find_run,
    is_judge_role,
    location_of,
    looks_like_run,
    read_records,
    run_files,
    run_manifest,
    samples_dir,
    state_dir,
    trace_calls,
    trace_stage,
)
from corallab.api.listing import list_manifests, read_test_case_summaries
from corallab.api.manifests import dump_manifest
from corallab.api.models import (
    BurrApp,
    CancelResult,
    DeleteRunResult,
    RunListing,
    RunStage,
    RunStatus,
    StartRunResult,
)
from corallab.api.preflight import preflight
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import FidelityCheckRecord, Transcript
from corallab.schemas.judge.eval_result import EvalResult

_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}
"""Processes this process started, so they can be reaped and their exit seen at once."""

_ADDRESS = re.compile(r"(?:https?://|api_base=)\S+")
_LOG_ERROR = re.compile(r"^\S+ \S+ ERROR (.*)$")
_MAX_ERRORS = 20
_POLL_S = 0.1


class _RunRecord(BaseModel):
    """What ``start_run`` keeps about a run's process."""

    run_id: str
    results_dir: str
    pid: int
    command: list[str]
    manifest_path: str
    run_manifest: str
    test_case_ids: list[str] | None = None
    started_at: str
    output: str
    exit_file: str
    scratch: str | None = None
    cancelled_at: str | None = None


def run_command(manifest: Path) -> list[str]:
    """The command ``start_run`` starts: ``corallab run`` in this Python environment."""
    return [sys.executable, "-m", "corallab.cli", "run", "--run-manifest", str(manifest)]


# ---------------------------------------------------------------------------
# Process records
# ---------------------------------------------------------------------------


def _record_path(run_id: str) -> Path:
    return state_dir() / f"{run_id}.json"


def _read_record(run_id: str) -> _RunRecord | None:
    path = _record_path(run_id)
    if not path.is_file():
        return None
    try:
        return _RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_record(record: _RunRecord) -> None:
    path = _record_path(record.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _records() -> list[_RunRecord]:
    directory = state_dir()
    if not directory.is_dir():
        return []
    found = (_read_record(p.stem) for p in sorted(directory.glob("*.json")))
    return [r for r in found if r is not None]


def _alive(pid: int) -> bool:
    """Whether ``start_run``'s process ``pid`` is still running.

    A process this one started is polled, which also reaps it. Any other is probed with signal
    0 and then ``ps``, so a zombie or a reused pid running something else counts as ended.
    """
    child = _CHILDREN.get(pid)
    if child is not None:
        return child.poll() is None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        out = subprocess.run(
            ["ps", "-o", "stat=,command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(out) and not out.startswith("Z") and "corallab.api._child" in out


def _exit_code(record: _RunRecord) -> int | None:
    try:
        return int(Path(record.exit_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _redact(text: str) -> str:
    """``text`` without network addresses, which the run log prints for each call."""
    return _ADDRESS.sub("<address>", text)


def _tail(path: Path, lines: int = 15) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    kept = [line.rstrip() for line in text.splitlines() if line.strip()]
    return [_redact(line)[:300] for line in kept[-lines:]]


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


def _line_id(line: str) -> str | None:
    try:
        value = json.loads(line).get("test_case_id")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _narrow(
    manifest_file: Path, data: dict, ids: list[str]
) -> tuple[Path | None, Path | None, list[str]]:
    """Copy the manifest and its test case records, keeping only ``ids``.

    Returns (the copied manifest, the folder holding the copies, problems). The records copy
    keeps each kept line byte for byte, under a ``test_case_records/`` folder of its own so the
    viewer names its suite as it names the original's.
    """
    records = paths.resolve_data_path(str(data.get("test_case_records_path") or ""))
    if not records.is_file():
        return None, None, [f"test case records not found: {records}"]
    known = {s.id for s in read_test_case_summaries(records) if s.error is None}
    wanted = list(dict.fromkeys(ids))
    unknown = [i for i in wanted if i not in known]
    if unknown:
        return None, None, [f"not in {relative_to_data(records)}: {', '.join(unknown)}"]
    lines = records.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if line.strip() and _line_id(line) in wanted]

    scratch = state_dir() / "inputs" / uuid.uuid4().hex[:12]
    records_root = paths.resolve_data_path("test_case_records")
    try:
        relative = records.resolve().relative_to(records_root.resolve())
    except ValueError:
        relative = Path(records.name)
    narrowed_records = scratch / "test_case_records" / relative
    narrowed_records.parent.mkdir(parents=True, exist_ok=True)
    narrowed_records.write_text("".join(kept), encoding="utf-8")
    narrowed = dict(data)
    narrowed["test_case_records_path"] = relative_to_data(narrowed_records)
    narrowed_manifest = scratch / manifest_file.name
    narrowed_manifest.write_text(dump_manifest(narrowed), encoding="utf-8")
    return narrowed_manifest, scratch, []


def _shared_notes(manifest_file: Path, manifest: RunManifest) -> list[str]:
    """Other manifests whose runs share this one's results folder name or run_id."""
    mine = paths.resolve_run_dir(manifest.run_dir).resolve()
    own = manifest_file.resolve()
    notes: list[str] = []
    for other in list_manifests():
        if not other.valid or (paths.data_dir() / other.path).resolve() == own:
            continue
        if other.run_dir and paths.resolve_run_dir(other.run_dir).resolve() == mine:
            notes.append(
                f"run_dir {manifest.run_dir} is shared with {other.path}; both manifests' "
                "runs get folders with the same name"
            )
        if other.run_id == manifest.run_id:
            notes.append(f"run_id {manifest.run_id} is shared with {other.path}")
    notes += [
        f"run {record.run_id} of this manifest is still running"
        for record in _records()
        if Path(record.manifest_path).resolve() == own and _alive(record.pid)
    ]
    return notes


def _folder_pattern(base: Path) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(base.name)}-\d{{8}}-\d{{6}}(?:-\d+)?$")


def _run_folders(base: Path) -> set[str]:
    """Folders beside ``base`` named as the CLI names this manifest's runs."""
    if not base.parent.is_dir():
        return set()
    pattern = _folder_pattern(base)
    return {p.name for p in base.parent.iterdir() if p.is_dir() and pattern.match(p.name)}


@contextlib.contextmanager
def _start_lock() -> Iterator[None]:
    """Serialize ``start_run`` calls, so each sees only the folder its own process created."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".start.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _terminate(pid: int, grace: float) -> str | None:
    """SIGTERM the process group, then SIGKILL after ``grace`` seconds. Returns the last signal."""
    sent = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return sent
        except PermissionError:
            return sent
        sent = sig.name
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not _alive(pid):
                return sent
            time.sleep(_POLL_S)
    return sent


def start_run(
    manifest_path: str | Path,
    test_case_ids: list[str] | None = None,
    *,
    wait_seconds: float = 120.0,
    query_ollama: bool = True,
    dotenv: bool = True,
) -> StartRunResult:
    """Preflight a manifest and, when it has no errors, start ``corallab run`` in the background.

    ``manifest_path`` is a path or a manifest id. ``test_case_ids`` narrows the run to those
    test cases: the manifest and records are copied with only those lines, and the copies run.
    Waits up to ``wait_seconds`` for the CLI to create its results folder and returns that
    folder's name as ``run_id``. Never raises for a problem in the inputs; nothing is started
    when ``problems`` is not empty. Preflight warnings and manifests sharing this one's
    ``run_dir`` or ``run_id`` are in ``warnings``.
    """
    path = resolve_manifest_path(manifest_path)
    result = StartRunResult(
        started=False, manifest_path=str(manifest_path), test_case_ids=test_case_ids
    )
    if not path.is_file():
        return result.model_copy(update={"problems": [f"manifest not found: {manifest_path}"]})
    try:
        data = read_run_manifest(path)
    except ValueError as exc:
        return result.model_copy(update={"problems": [str(exc)]})

    to_run, scratch = path, None
    if test_case_ids is not None:
        if not test_case_ids:
            return result.model_copy(update={"problems": ["test_case_ids is empty"]})
        narrowed, scratch, problems = _narrow(path, data, test_case_ids)
        if narrowed is None:
            return result.model_copy(update={"problems": problems})
        to_run = narrowed
    result.run_manifest = relative_to_data(to_run)

    report = preflight(to_run, query_ollama=query_ollama, dotenv=dotenv)
    result.preflight = report
    if not report.ok:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        return result.model_copy(
            update={"problems": [f"preflight {e.check}: {e.message}" for e in report.errors]}
        )
    manifest = RunManifest.model_validate({"run_id": "run-generated", **data})
    warnings = [f"preflight {w.check}: {w.message}" for w in report.warnings]
    warnings += _shared_notes(path, manifest)

    base = paths.resolve_run_dir(manifest.run_dir)
    command = run_command(to_run)
    result.command = command
    token = uuid.uuid4().hex[:12]
    output = state_dir() / f"{token}.out"
    exit_file = state_dir() / f"{token}.exit"
    with _start_lock():
        before = _run_folders(base)
        with output.open("wb") as out:
            child = subprocess.Popen(
                [sys.executable, "-m", "corallab.api._child", str(exit_file), "--", *command],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _CHILDREN[child.pid] = child
        created, problem = _detect(base, before, child, wait_seconds)

    if created is None:
        if child.poll() is None:
            _terminate(child.pid, 5.0)
        tail = _tail(output)
        output.unlink(missing_ok=True)
        exit_file.unlink(missing_ok=True)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        return result.model_copy(
            update={"warnings": warnings, "problems": [problem, *tail], "command": command}
        )

    record = _RunRecord(
        run_id=created.name,
        results_dir=str(created),
        pid=child.pid,
        command=command,
        manifest_path=str(path),
        run_manifest=str(to_run),
        test_case_ids=test_case_ids,
        started_at=datetime.now(UTC).isoformat(),
        output=str(output),
        exit_file=str(exit_file),
        scratch=str(scratch) if scratch else None,
    )
    _write_record(record)
    return result.model_copy(
        update={
            "started": True,
            "run_id": created.name,
            "results_dir": relative_to_data(created),
            "pid": child.pid,
            "warnings": warnings,
        }
    )


def _detect(
    base: Path, before: set[str], child: subprocess.Popen[bytes], wait_seconds: float
) -> tuple[Path | None, str]:
    """The results folder ``child`` created, or (None, why not)."""
    deadline = time.monotonic() + wait_seconds
    while True:
        code = child.poll()  # before looking, so a folder made just before exiting is seen
        new = sorted(_run_folders(base) - before)
        if len(new) == 1:
            return base.parent / new[0], ""
        if len(new) > 1:
            return None, f"several new results folders appeared at once: {', '.join(new)}"
        if code is not None:
            return None, f"corallab run exited with code {code} before creating its results folder"
        if time.monotonic() >= deadline:
            return (
                None,
                f"no results folder appeared within {wait_seconds:g} s; the run was stopped",
            )
        time.sleep(_POLL_S)


# ---------------------------------------------------------------------------
# run_status, cancel_run
# ---------------------------------------------------------------------------


def _log_errors(log: Path | None) -> list[str]:
    if log is None:
        return []
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    found = [m.group(1) for line in lines if (m := _LOG_ERROR.match(line))]
    return [f"log: {_redact(message)[:300]}" for message in found]


def _record_errors(
    files_dir: Path, sims: list[Path], checks: list[Path], judges: list[Path]
) -> list[str]:
    errors: list[str] = []
    transcripts, problems = read_records(sims, Transcript, files_dir)
    for _, t in transcripts:
        if t.sim_failure is not None:
            errors.append(f"sim {t.transcript_id}: {t.sim_failure.reason}")
    records, more = read_records(checks, FidelityCheckRecord, files_dir)
    problems += more
    for _, c in records:
        if c.error:
            errors.append(f"check {c.transcript_id}: {c.error}")
    results, more = read_records(judges, EvalResult, files_dir)
    problems += more
    for _, r in results:
        run = r.coralmind_run
        if r.status != "success" or (run is not None and run.run_status != "success"):
            detail = run.run_error_detail if run is not None else None
            errors.append(
                f"judge {r.transcript_id}: {r.status}" + (f" ({detail})" if detail else "")
            )
    return errors + problems


def _transcripts_total(files_dir: Path) -> tuple[int | None, int | None]:
    """(transcripts the run will write, batches) from its copied manifest and records."""
    files = run_files(files_dir)
    manifest, _ = run_manifest(files)
    if manifest is None or files.inputs_records is None:
        return None, None
    cases = sum(1 for s in read_test_case_summaries(files.inputs_records) if s.error is None)
    return cases * len(manifest.system_under_test), math.ceil(
        cases / manifest.simulation.batch_size
    )


def _newest_mtime(directory: Path) -> float | None:
    times = [p.stat().st_mtime for p in directory.rglob("*") if p.is_file()]
    return max(times, default=None)


def run_status(run_id: str) -> RunStatus:
    """How far a run has got: its stage, transcripts, checks and judge results written, model
    calls traced, errors so far, elapsed time, and whether its process is alive.

    ``alive``, ``exit_code`` and ``cancelled`` are known only for runs ``start_run`` started.
    A batch's transcripts are written when the whole batch ends; until then its progress
    shows in ``calls``.
    """
    record = _read_record(run_id)
    directory = find_run(run_id)
    if directory is None and record is not None:
        directory = Path(record.results_dir)
    if directory is None or not directory.is_dir():
        return RunStatus(
            run_id=run_id,
            found=False,
            started_by_api=record is not None,
            alive=None,
            batches_simulated=0,
            transcripts_done=0,
            problems=[f"no run named {run_id}"],
        )
    files = run_files(directory)
    calls = trace_calls(files.traces)
    total, batches = _transcripts_total(directory)

    alive: bool | None = None
    exit_code: int | None = None
    cancelled = False
    errors = _log_errors(files.log)
    errors += _record_errors(directory, files.sim, files.check, files.judge)
    if record is not None:
        alive = _alive(record.pid)
        exit_code = None if alive else _exit_code(record)
        cancelled = record.cancelled_at is not None
        if exit_code not in (None, 0) and not cancelled:
            errors += [f"output: {line}" for line in _tail(Path(record.output))]

    stage: RunStage | None = None
    if alive is False:
        stage = "cancelled" if cancelled else "finished" if exit_code == 0 else "failed"
    elif files.traces:
        newest = max(files.traces, key=lambda p: p.stat().st_mtime)
        stage = trace_stage(newest)  # type: ignore[assignment]
    elif alive:
        stage = "starting"

    started = record.started_at if record is not None else created_at(run_id)
    elapsed = None
    if started is not None:
        start = datetime.fromisoformat(started)
        if start.tzinfo is None:
            start = start.astimezone()
        if alive:
            end = time.time()
        elif record is not None and Path(record.exit_file).is_file():
            end = Path(record.exit_file).stat().st_mtime
        else:
            end = _newest_mtime(directory) or time.time()
        elapsed = round(max(end - start.timestamp(), 0.0), 1)

    return RunStatus(
        run_id=run_id,
        results_dir=relative_to_data(directory),
        found=True,
        started_by_api=record is not None,
        pid=record.pid if record is not None else None,
        alive=alive,
        exit_code=exit_code,
        cancelled=cancelled,
        stage=stage,
        batches_total=batches,
        batches_simulated=sum(1 for b in files.batches if any(b.glob("*-sim.jsonl"))),
        transcripts_total=total,
        transcripts_done=sum(count_lines(p) for p in files.sim),
        checks_done=sum(count_lines(p) for p in files.check),
        judged_done=sum(count_lines(p) for p in files.judge),
        judge_calls_done=sum(1 for c in calls if is_judge_role(c.role)),
        calls=count_calls(calls),
        errors=errors[-_MAX_ERRORS:],
        started_at=started,
        elapsed_seconds=elapsed,
    )


def cancel_run(run_id: str, *, grace_seconds: float = 10.0) -> CancelResult:
    """Stop a run ``start_run`` started: SIGTERM to its process group, SIGKILL after
    ``grace_seconds``. Files it already wrote are kept; ``delete_run`` removes them."""
    record = _read_record(run_id)
    if record is None:
        return CancelResult(
            run_id=run_id,
            cancelled=False,
            was_alive=False,
            problems=[f"{run_id} was not started by start_run, so its process is not known"],
        )
    if not _alive(record.pid):
        return CancelResult(run_id=run_id, cancelled=False, was_alive=False)
    record.cancelled_at = datetime.now(UTC).isoformat()
    _write_record(record)
    sent = _terminate(record.pid, grace_seconds)
    still = _alive(record.pid)
    return CancelResult(
        run_id=run_id,
        cancelled=not still,
        was_alive=True,
        signal=sent,
        problems=[f"process {record.pid} is still running"] if still else [],
    )


# ---------------------------------------------------------------------------
# delete_run, list_runs
# ---------------------------------------------------------------------------


def _agent_files(record: _RunRecord | None) -> list[Path]:
    if record is None:
        return []
    found = [_record_path(record.run_id), Path(record.output), Path(record.exit_file)]
    if record.scratch:
        scratch = Path(record.scratch)
        found += sorted(p for p in scratch.rglob("*") if p.is_file())
    return [p for p in found if p.is_file()]


def delete_run(
    run_id: str,
    include_burr: bool = True,
    *,
    dry_run: bool = True,
    burr_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> DeleteRunResult:
    """Delete a run's results folder, ``start_run``'s files for it, and (with ``include_burr``)
    its Burr tracker folders. A dry run by default: pass ``dry_run=False`` to delete.

    ``burr_dir`` is the tracker's project folder (default ``~/.burr/corallab``). Tracker folders
    whose run cannot be read from their own metadata are listed in ``burr_unattributed`` and
    kept. ``db_path`` names a SQLite persister file given to ``--db-path``; the rows of this
    run's applications are deleted from it, and the file is kept. Runs under
    ``sample_results/`` and runs still alive are refused.
    """
    result = DeleteRunResult(run_id=run_id, dry_run=dry_run, deleted=False)
    if Path(run_id).name != run_id or run_id.startswith("."):
        return result.model_copy(update={"problems": [f"{run_id!r} is not a run id"]})
    record = _read_record(run_id)
    directory = find_run(run_id)
    if directory is None and record is not None and Path(record.results_dir).is_dir():
        directory = Path(record.results_dir)
    if directory is None:
        return result.model_copy(update={"problems": [f"no run named {run_id}"]})
    result.results_dir = relative_to_data(directory)
    location = location_of(directory)
    if location == "sample" or directory.resolve().is_relative_to(samples_dir().resolve()):
        return result.model_copy(update={"problems": ["sample runs are never deleted"]})
    if location == "elsewhere" and record is None:
        return result.model_copy(
            update={"problems": ["only runs under results/ or started by start_run are deleted"]}
        )
    if directory.resolve() in (paths.results_dir().resolve(), state_dir().resolve()):
        return result.model_copy(update={"problems": [f"{directory} is not a run folder"]})
    if record is not None and _alive(record.pid):
        return result.model_copy(update={"problems": ["the run is still alive; cancel_run first"]})

    files = sorted(p for p in directory.rglob("*") if p.is_file())
    result.files = [p.relative_to(directory).as_posix() for p in files]
    result.bytes = sum(p.stat().st_size for p in files)
    agent = _agent_files(record)
    result.agent_files = [relative_to_data(p) for p in agent]

    app_ids: list[str] = []
    if include_burr:
        tracker = Path(burr_dir) if burr_dir is not None else default_burr_dir()
        found = attribute(tracker, run_id)
        result.burr_apps = [
            BurrApp(app_id=f.app_id, path=str(f.path), reason=r) for f, r in found.apps
        ]
        result.burr_unattributed = [
            BurrApp(app_id=f.app_id, path=str(f.path), reason=r) for f, r in found.unattributed
        ]
        app_ids = [f.app_id for f, _ in found.apps]
    db = Path(db_path) if db_path is not None else None
    db_ids: list[str] = []
    if db is not None:
        named = named_app_pattern(run_id)
        db_ids = [i for i in persister_app_ids(db) if i in app_ids or named.match(i)]
        result.db_rows = {str(db): persister_rows(db, db_ids)}

    if dry_run:
        return result

    shutil.rmtree(directory)
    for path in agent:
        path.unlink(missing_ok=True)
    if record is not None and record.scratch:
        shutil.rmtree(record.scratch, ignore_errors=True)
    for app in result.burr_apps:
        shutil.rmtree(app.path, ignore_errors=True)
    if db is not None:
        result.db_rows = {str(db): delete_persister_rows(db, db_ids)}
    result.deleted = True
    return result


def _listing(directory: Path, record: _RunRecord | None) -> RunListing:
    files = run_files(directory)
    data = files.manifest_data
    level = data.get("evaluation_level")
    suts = [
        str(s.get("model_id")) for s in data.get("system_under_test") or [] if isinstance(s, dict)
    ]
    return RunListing(
        run_id=directory.name,
        path=relative_to_data(directory),
        location=location_of(directory),
        created_at=created_at(directory.name),
        manifest_run_id=data.get("run_id") if isinstance(data.get("run_id"), str) else None,
        evaluation_level=level if isinstance(level, int) else None,
        records_file=files.inputs_records.name if files.inputs_records else None,
        system_under_test=suts,
        batches=len(files.batches),
        transcripts=sum(count_lines(p) for p in files.sim),
        checked=sum(count_lines(p) for p in files.check),
        judged=sum(count_lines(p) for p in files.judge),
        started_by_api=record is not None,
        alive=_alive(record.pid) if record is not None else None,
    )


def list_runs(*, include_samples: bool = False) -> list[RunListing]:
    """Every results folder under ``results/``, and runs ``start_run`` recorded elsewhere,
    oldest first. ``include_samples`` adds the published runs in ``sample_results/``."""
    records = {r.run_id: r for r in _records()}
    directories: dict[str, Path] = {}
    roots = [paths.results_dir()] + ([samples_dir()] if include_samples else [])
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.name.startswith(".") and looks_like_run(child):
                directories.setdefault(str(child.resolve()), child)
    for record in records.values():
        path = Path(record.results_dir)
        if path.is_dir():
            directories.setdefault(str(path.resolve()), path)
    listings = [_listing(d, records.get(d.name)) for d in directories.values()]
    return sorted(listings, key=lambda r: (r.created_at or "", r.run_id))
