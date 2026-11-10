"""start_run, run_status, cancel_run, list_runs and delete_run.

The process started is ``tests/api/_fake_run.py``, which names its folder as the CLI does, or
the real ``corallab sim --dry-run``, which makes no model calls. HOME points at a temporary
folder, so Burr's tracker writes there and never to the real ``~/.burr``.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from _records import make_test_case

from corallab.api import runs
from corallab.api.runs import cancel_run, delete_run, list_runs, run_status, start_run
from tests.api._repo import DataRepo, manifest_data
from tests.api._runs import transcript, write_run

FAKE = Path(__file__).with_name("_fake_run.py")
TC1, TC2 = "aimhef-0000f001", "aimhef-0000f002"


@pytest.fixture
def repo(
    data_repo: DataRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DataRepo]:
    """The API test repository with its env vars set and HOME in a temporary folder."""
    monkeypatch.setenv("TEST_API_KEY", "not-a-real-key")  # pragma: allowlist secret
    monkeypatch.setenv("TEST_API_BASE", "http://127.0.0.1:9")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    data_repo.write_cases([make_test_case(tc).model_dump_json() for tc in (TC1, TC2)])
    yield data_repo
    for child in list(runs._CHILDREN.values()):
        if child.poll() is None:
            child.kill()
        child.wait()
    runs._CHILDREN.clear()


def fake(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(runs, "run_command", lambda m: [sys.executable, str(FAKE), str(m), mode])


def start(manifest: str = "smoke", ids: list[str] | None = None):  # type: ignore[no-untyped-def]
    return start_run(manifest, ids, wait_seconds=60, query_ollama=False, dotenv=False)


def wait_until_ended(run_id: str, timeout: float = 90.0):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = run_status(run_id)
        if status.alive is False:
            return status
        time.sleep(0.2)
    raise AssertionError(f"{run_id} did not end within {timeout} s")


# ---------------------------------------------------------------------------
# start_run
# ---------------------------------------------------------------------------


def test_start_run_refuses_a_manifest_preflight_rejects(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_API_KEY")
    fake(monkeypatch, "ok")
    result = start()
    assert not result.started
    assert result.preflight is not None
    assert not result.preflight.ok
    assert any("TEST_API_KEY is not set" in p for p in result.problems)
    assert not (repo.root / "results").exists() or not list((repo.root / "results").glob("api-*"))


def test_start_run_refuses_unknown_inputs(repo: DataRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, "ok")
    assert start("nope").problems == ["manifest not found: nope"]
    assert start(ids=[]).problems == ["test_case_ids is empty"]
    assert start(ids=["aimhef-0000ffff"]).problems == [
        "not in test_case_records/cases.jsonl: aimhef-0000ffff"
    ]
    assert runs._CHILDREN == {}


def test_start_run_returns_the_folder_the_cli_really_created(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake(monkeypatch, "ok")
    # Take this second and the next ones, so the CLI's name collides and gets "-2".
    now = datetime.now()
    taken = [
        repo.root / "results" / f"api-test-{(now + timedelta(seconds=s)):%Y%m%d-%H%M%S}"
        for s in range(30)
    ]
    for folder in taken:
        folder.mkdir(parents=True)
    result = start()
    assert result.started, result.problems
    assert result.run_id is not None
    assert result.run_id.endswith("-2")
    assert result.results_dir == f"results/{result.run_id}"
    assert (repo.root / "results" / result.run_id / "inputs" / "smoke.yaml").is_file()
    assert result.run_manifest == "run_manifests/smoke.yaml"
    assert result.command[-1] == "ok"
    record = json.loads(
        (repo.root / "results" / ".agent-runs" / f"{result.run_id}.json").read_text()
    )
    assert record["pid"] == result.pid


def test_run_status_follows_a_run_to_its_end(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake(monkeypatch, "ok")
    result = start()
    assert result.run_id is not None
    status = wait_until_ended(result.run_id)
    assert status.found
    assert status.started_by_api
    assert (status.stage, status.exit_code, status.cancelled) == ("finished", 0, False)
    assert (status.transcripts_done, status.transcripts_total) == (1, 2)
    assert (status.batches_simulated, status.batches_total) == (1, 2)
    assert [(c.role, c.calls) for c in status.calls] == [("system_under_test", 1)]
    assert status.errors == ["log: corallab.cli.run: Sim failed: <address>"]
    assert status.elapsed_seconds is not None
    assert status.elapsed_seconds >= 0


def test_run_status_reports_a_failed_run(repo: DataRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, "crash")
    result = start()
    assert result.run_id is not None
    status = wait_until_ended(result.run_id)
    assert (status.stage, status.exit_code) == ("failed", 1)


def test_start_run_reports_a_process_that_ends_before_its_folder(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake(monkeypatch, "fail")
    result = start()
    assert not result.started
    assert result.run_id is None
    assert (
        result.problems[0] == "corallab run exited with code 3 before creating its results folder"
    )
    assert any("Error: could not reach" in p for p in result.problems)
    assert not any("invalid" in p for p in result.problems)  # addresses removed
    assert list((repo.root / "results" / ".agent-runs").glob("*.out")) == []


def test_start_run_narrows_the_run_to_test_case_ids(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake(monkeypatch, "ok")
    result = start(ids=[TC2])
    assert result.started
    assert result.run_id is not None
    assert result.run_manifest is not None
    assert result.run_manifest.startswith("results/.agent-runs/inputs/")
    run = repo.root / "results" / result.run_id
    copied = run / "inputs" / "cases.jsonl"
    original = (repo.root / "test_case_records" / "cases.jsonl").read_text().splitlines(True)
    assert copied.read_text() == original[1]  # the kept line, byte for byte
    manifest = yaml.safe_load((run / "inputs" / "smoke.yaml").read_text())
    assert manifest["test_case_records_path"].endswith("/test_case_records/cases.jsonl")
    assert manifest["run_dir"] == "results/api-test"
    assert wait_until_ended(result.run_id).transcripts_total == 1


def test_start_run_warns_about_a_shared_run_dir(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake(monkeypatch, "ok")
    repo.write_manifest("other", manifest_data(run_id="other"))
    result = start()
    assert result.started
    assert (
        "run_dir results/api-test is shared with run_manifests/other.yaml; both manifests' "
        "runs get folders with the same name"
    ) in result.warnings


# ---------------------------------------------------------------------------
# cancel_run
# ---------------------------------------------------------------------------


def test_cancel_run_stops_the_process(repo: DataRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, "sleep")
    result = start()
    assert result.run_id is not None
    status = run_status(result.run_id)
    assert status.alive
    assert status.stage == "sim"
    assert delete_run(result.run_id).problems == ["the run is still alive; cancel_run first"]

    cancelled = cancel_run(result.run_id, grace_seconds=10)
    assert cancelled.cancelled
    assert cancelled.was_alive
    assert cancelled.signal == "SIGTERM"
    status = run_status(result.run_id)
    assert (status.alive, status.stage, status.cancelled) == (False, "cancelled", True)
    assert not cancel_run(result.run_id).was_alive
    assert cancel_run("api-test-19990101-000000").problems == [
        "api-test-19990101-000000 was not started by start_run, so its process is not known"
    ]


def test_status_and_cancel_work_from_another_process(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the Popen handle, as a later process sees it: signal 0 and ps."""
    fake(monkeypatch, "sleep")
    result = start()
    assert result.run_id is not None
    assert result.pid is not None
    child = runs._CHILDREN.pop(result.pid)
    try:
        assert run_status(result.run_id).alive
        assert cancel_run(result.run_id, grace_seconds=10).cancelled
        status = run_status(result.run_id)  # a zombie until reaped: still counts as ended
        assert (status.alive, status.stage) == (False, "cancelled")
    finally:
        child.kill()
        child.wait()


# ---------------------------------------------------------------------------
# A real `corallab sim --dry-run`
# ---------------------------------------------------------------------------


def test_a_real_dry_run_is_found_followed_and_deleted(
    repo: DataRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runs,
        "run_command",
        lambda m: [
            sys.executable,
            "-m",
            "corallab.cli",
            "sim",
            "--dry-run",
            "--run-manifest",
            str(m),
        ],
    )
    result = start()
    assert result.started, result.problems
    assert result.run_id is not None
    status = wait_until_ended(result.run_id)
    assert (status.stage, status.exit_code) == ("finished", 0)
    assert status.calls == []  # dry-run traces are not counted as calls

    burr = tmp_path / "home" / ".burr" / "corallab"
    plan = delete_run(result.run_id, burr_dir=burr)
    assert plan.dry_run
    assert not plan.deleted
    named = {a.app_id for a in plan.burr_apps if a.reason == "named for this run"}
    assert named == {f"{result.run_id}-batch-001", f"{result.run_id}-batch-002"}
    spawned = [a for a in plan.burr_apps if a.reason.startswith("spawned under")]
    assert spawned, "the nested per-case apps are attributed through their metadata"
    assert {a.app_id for a in plan.burr_apps} == {p.name for p in burr.iterdir()}
    assert plan.burr_unattributed == []
    assert "inputs/smoke.yaml" in plan.files
    assert (repo.root / "results" / result.run_id).is_dir()

    done = delete_run(result.run_id, dry_run=False, burr_dir=burr)
    assert done.deleted
    assert not (repo.root / "results" / result.run_id).exists()
    assert list(burr.iterdir()) == []
    assert not (repo.root / "results" / ".agent-runs" / f"{result.run_id}.json").exists()


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_lists_results_and_samples(
    repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake(monkeypatch, "ok")
    started = start()
    assert started.run_id is not None
    wait_until_ended(started.run_id)
    write_run(
        repo.root / "results",
        "api-test-20261110-090000",
        manifest=manifest_data(),
        cases=[make_test_case(TC1).model_dump_json()],
        batches=[{"transcripts": [transcript(TC1)]}],
    )
    write_run(
        repo.root / "sample_results",
        "sample-20261001-000000",
        manifest=manifest_data(),
        cases=[make_test_case(TC1).model_dump_json()],
        batches=[{"transcripts": [transcript(TC1)]}],
    )
    listed = {r.run_id: r for r in list_runs()}
    assert set(listed) == {"api-test-20261110-090000", started.run_id}
    plain = listed["api-test-20261110-090000"]
    assert (plain.location, plain.transcripts, plain.started_by_api, plain.alive) == (
        "results",
        1,
        False,
        None,
    )
    assert plain.created_at == "2026-11-10T09:00:00"
    assert plain.system_under_test == ["remote/model"]
    assert listed[started.run_id].started_by_api
    assert listed[started.run_id].alive is False
    with_samples = {r.run_id: r for r in list_runs(include_samples=True)}
    assert with_samples["sample-20261001-000000"].location == "sample"


# ---------------------------------------------------------------------------
# delete_run
# ---------------------------------------------------------------------------


def _burr_app(root: Path, app_id: str, spawned_by: str | None = None) -> Path:
    folder = root / app_id
    folder.mkdir(parents=True)
    pointer = (
        {
            "type": "pointer_data",
            "app_id": spawned_by,
            "sequence_id": 1,
            "partition_key": "corallab",
        }
        if spawned_by
        else None
    )
    meta = {
        "type": "application_metadata",
        "partition_key": "corallab",
        "parent_pointer": None,
        "spawning_parent_pointer": pointer,
    }
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def test_delete_run_is_a_dry_run_by_default(repo: DataRepo, tmp_path: Path) -> None:
    run = write_run(
        repo.root / "results",
        "api-test-20261110-090000",
        manifest=manifest_data(),
        cases=[make_test_case(TC1).model_dump_json()],
        batches=[{"transcripts": [transcript(TC1)]}],
    )
    burr = tmp_path / "burr"
    _burr_app(burr, "api-test-20261110-090000-sim-batch-001")
    _burr_app(burr, "aaaa", "api-test-20261110-090000-sim-batch-001")
    _burr_app(burr, "bbbb", "aaaa")
    _burr_app(burr, "api-test-20261110-090000-2-judge")  # another run of the same manifest
    _burr_app(burr, "cccc", "api-test-20261110-090000-2-judge")
    _burr_app(burr, "dddd", "gone")  # its spawning app has no folder
    (burr / "eeee").mkdir()  # no metadata at all

    plan = delete_run("api-test-20261110-090000", burr_dir=burr)
    assert plan.problems == []
    assert not plan.deleted
    assert plan.results_dir == "results/api-test-20261110-090000"
    assert "batch-001/api-test-20261110-090000-sim.jsonl" in plan.files
    assert plan.bytes > 0
    assert {a.app_id: a.reason for a in plan.burr_apps} == {
        "api-test-20261110-090000-sim-batch-001": "named for this run",
        "aaaa": "spawned under api-test-20261110-090000-sim-batch-001 (metadata.json)",
        "bbbb": "spawned under api-test-20261110-090000-sim-batch-001 (metadata.json)",
    }
    assert {a.app_id: a.reason for a in plan.burr_unattributed} == {
        "dddd": "names gone, which has no folder",
        "eeee": "eeee: no metadata.json",
    }
    assert run.is_dir()
    assert (burr / "aaaa").is_dir()

    done = delete_run("api-test-20261110-090000", dry_run=False, burr_dir=burr)
    assert done.deleted
    assert not run.exists()
    assert sorted(p.name for p in burr.iterdir()) == [
        "api-test-20261110-090000-2-judge",
        "cccc",
        "dddd",
        "eeee",
    ]


def test_delete_run_can_leave_burr_alone(repo: DataRepo, tmp_path: Path) -> None:
    write_run(
        repo.root / "results",
        "api-test-20261110-090000",
        manifest=manifest_data(),
        cases=[],
        batches=[{"transcripts": [transcript(TC1)]}],
    )
    burr = tmp_path / "burr"
    _burr_app(burr, "api-test-20261110-090000-judge")
    done = delete_run("api-test-20261110-090000", include_burr=False, dry_run=False, burr_dir=burr)
    assert done.deleted
    assert done.burr_apps == []
    assert (burr / "api-test-20261110-090000-judge").is_dir()


def test_delete_run_removes_the_runs_rows_from_a_persister(repo: DataRepo, tmp_path: Path) -> None:
    write_run(
        repo.root / "results",
        "api-test-20261110-090000",
        manifest=manifest_data(),
        cases=[],
        batches=[{"transcripts": [transcript(TC1)]}],
    )
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE corallab_state (partition_key TEXT, app_id TEXT, sequence_id INTEGER, "
            "position TEXT, status TEXT, state TEXT)"
        )
        for app in (
            "api-test-20261110-090000-judge",
            "api-test-20261110-090000-judge",
            "other-judge",
        ):
            conn.execute(
                "INSERT INTO corallab_state VALUES ('corallab', ?, 0, 'x', 'ok', '{}')", (app,)
            )
    plan = delete_run("api-test-20261110-090000", burr_dir=tmp_path / "none", db_path=db)
    assert plan.db_rows == {str(db): 2}
    done = delete_run(
        "api-test-20261110-090000", dry_run=False, burr_dir=tmp_path / "none", db_path=db
    )
    assert done.db_rows == {str(db): 2}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT app_id FROM corallab_state").fetchall() == [("other-judge",)]


def test_delete_run_never_touches_sample_runs(repo: DataRepo, tmp_path: Path) -> None:
    sample = write_run(
        repo.root / "sample_results",
        "sample-20261001-000000",
        manifest=manifest_data(),
        cases=[],
        batches=[{"transcripts": [transcript(TC1)]}],
    )
    result = delete_run("sample-20261001-000000", dry_run=False, burr_dir=tmp_path / "burr")
    assert result.problems == ["sample runs are never deleted"]
    assert not result.deleted
    assert sample.is_dir()
    assert delete_run(str(sample), dry_run=False).problems == [f"{str(sample)!r} is not a run id"]
    assert delete_run("../sample_results").problems == ["'../sample_results' is not a run id"]
    assert delete_run(".agent-runs").problems == ["'.agent-runs' is not a run id"]
    assert delete_run("nope").problems == ["no run named nope"]
