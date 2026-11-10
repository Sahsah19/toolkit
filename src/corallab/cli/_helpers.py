"""Shared CLI helpers — logging, run-manifest resolution, and the results directory."""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.logging import RichHandler

from corallab import paths
from corallab.api._manifests import read_run_manifest
from corallab.paths import resolve_run_dir
from corallab.paths import results_dir as data_results_dir
from corallab.schemas.eval.run_manifest import RunManifest

if TYPE_CHECKING:
    from corallab.llm_client import TokenAccumulator

logger = logging.getLogger("corallab")

console = Console()
err_console = Console(stderr=True)


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
_BATCH_DIR = re.compile(r"batch-\d+")


def safe_name(value: object) -> str:
    """Sanitize a record id into a single safe file/folder name component.

    Rejects empty and all-dots results (``.``, ``..``) since those would
    resolve to the current/parent directory and escape the output folder.
    """
    name = _SAFE_NAME.sub("_", str(value))
    if not name or set(name) <= {"."}:
        return "unknown"
    return name


def unique_name(base: str, seen: set[str]) -> str:
    """Return a name not in *seen* (suffixing ``-2``, ``-3``, … on collision) and record it."""
    name = base
    i = 2
    while name in seen:
        name = f"{base}-{i}"
        i += 1
    seen.add(name)
    return name


def setup_logging(level: LogLevel, run_stem: str, out_dir: Path) -> None:
    """Configure rich console + file logging for a run."""
    log_level = getattr(logging, level.value)
    log_format = "%(message)s"
    file_log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    root = logging.getLogger()
    root.setLevel(log_level)

    rich_handler = RichHandler(
        level=log_level,
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=level == LogLevel.DEBUG,
    )
    rich_handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(rich_handler)

    log_file = out_dir / f"{run_stem}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(file_log_format))
    root.addHandler(fh)

    logger.info("Log file: %s", log_file)


def make_run_stem(command: str) -> str:
    """Derive a unique stem for a command that has no ``run_dir``.

    Commands that load a run manifest write into a timestamped copy of
    ``manifest.run_dir`` and name files after that directory. This stem is only
    for commands that do not.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"corallab-{command}-{timestamp}"


def timestamped_run_dir(run_dir: Path) -> Path:
    """Return ``run_dir`` with ``-YYYYMMDD-HHMMSS`` appended to its name.

    A second call in the same second gets ``-2``, ``-3``, and so on, so a new
    run never reuses a directory that already exists.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{run_dir.name}-{stamp}"
    candidate = run_dir.with_name(base)
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        numbered = run_dir.with_name(f"{base}-{suffix}")
        if not numbered.exists():
            return numbered
        suffix += 1


def transcript_run_dir(transcripts: str | Path) -> Path | None:
    """Return the results directory that already holds this transcript file.

    Batch files live at ``<run>/batch-NNN/<stem>-sim.jsonl``, so the run
    directory is the batch directory's parent. A file written at the run root
    uses its own parent. A path that is not an existing file returns None.
    """
    path = Path(transcripts)
    if not path.is_file():
        return None
    if _BATCH_DIR.fullmatch(path.parent.name):
        return path.parent.parent
    return path.parent


def resolve_run_manifest(run_manifest_path: str) -> RunManifest:
    """Load a run manifest YAML file and validate it as ``RunManifest``."""
    try:
        run_manifest_data = read_run_manifest(run_manifest_path)
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(1) from exc
    if "run_id" not in run_manifest_data:
        run_manifest_data["run_id"] = f"run-{uuid.uuid4().hex[:8]}"

    if not run_manifest_data.get("test_case_records_path"):
        err_console.print("[bold red]Error:[/] no test_case_records_path in run manifest")
        raise typer.Exit(1)
    if not run_manifest_data.get("run_dir"):
        err_console.print("[bold red]Error:[/] no run_dir in run manifest")
        raise typer.Exit(1)

    return RunManifest.model_validate(run_manifest_data)


def check_fidelity_prompt(manifest: RunManifest) -> None:
    """Exit when the manifest names a fidelity check prompt the checker cannot read."""
    from corallab.fidelity.elicitation import prompt_problem

    if manifest.evaluation_level < 3 or not manifest.simulation.elicitation_fidelity.enabled:
        return
    problem = prompt_problem(manifest)
    if problem:
        err_console.print(f"[bold red]Error:[/] {problem}")
        raise typer.Exit(1)


def results_dir() -> Path:
    """Return (and create) ``<data-repository>/results``."""
    d = data_results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def init_logging(
    ctx: typer.Context,
    manifest: RunManifest | None = None,
    *,
    out_dir: Path | None = None,
) -> tuple[Path, str]:
    """Create the run directory and wire up logging. Returns (out_dir, run_stem).

    A manifest's ``run_dir`` is the base directory. Each invocation appends
    ``-YYYYMMDD-HHMMSS`` (and ``-2``, ``-3``, … if that folder already exists)
    so a later run does not overwrite an earlier one. A relative ``run_dir``
    is placed under the data repository before the timestamp is added.
    Transcripts, judge results, the log, the Burr database, and copied inputs
    go in that timestamped directory. The file stem is the directory's name.

    Pass ``out_dir`` to write into a directory that already exists, such as
    the folder that holds the transcripts ``judge`` was given. Commands with
    no manifest still get a timestamped folder under ``<data-repository>/results/``.
    """
    command = ctx.info_name or "corallab"
    if out_dir is not None:
        resolved = out_dir
        run_stem = resolved.name
    elif manifest is not None:
        resolved = timestamped_run_dir(resolve_run_dir(manifest.run_dir))
        run_stem = resolved.name
    else:
        run_stem = make_run_stem(command)
        resolved = results_dir() / run_stem
    resolved.mkdir(parents=True, exist_ok=True)
    setup_logging(ctx.obj["log_level"], run_stem, resolved)
    return resolved, run_stem


def copy_run_inputs(
    manifest_path: str | Path,
    manifest: RunManifest,
    out_dir: Path,
) -> None:
    """Freeze the run's source inputs (manifest + test case records) into ``out_dir/inputs/``.

    Provenance stopgap until a database backend lands (issue #207): every run
    keeps an immutable copy of the exact run manifest and the test-case JSONL it
    was driven by, so a results folder is self-describing even if the originals
    are later edited or moved. Copy failures are logged but never abort the run —
    preserving inputs is a convenience, not part of the core pipeline. Persona
    records are not copied; the test case records name them by pinned reference.
    """
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for src in (Path(manifest_path), paths.resolve_data_path(manifest.test_case_records_path)):
        dest = inputs_dir / src.name
        try:
            shutil.copy2(src, dest)
        except OSError as exc:
            logger.warning("Could not copy run input %s into %s: %s", src, inputs_dir, exc)
        else:
            logger.info("Copied run input %s → %s", src, dest)


def print_token_summary(accumulator: TokenAccumulator, label: str) -> None:
    """Log and print a run's token-usage summary to the CLI console."""
    lines = accumulator.summary_lines(label)
    logger.info("%s", "\n".join(lines))
    console.print()
    for line in lines:
        console.print(line)
    console.print()
