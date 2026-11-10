"""The helpers behind the run functions: Burr attribution, run lookup, trace roles, the
exit-code wrapper."""

from __future__ import annotations

import json
import signal
import sqlite3
import sys
from pathlib import Path

import pytest

from corallab.api import _burr, _child, _results
from tests.api._repo import DataRepo, manifest_data
from tests.api._runs import transcript, write_run

STEM = "run-20261110-100000"


@pytest.mark.parametrize(
    ("app_id", "matches"),
    [
        (f"{STEM}-sim-batch-001", True),  # corallab run
        (f"{STEM}-batch-012", True),  # corallab sim
        (f"{STEM}-judge", True),
        (f"{STEM}__azure-gpt-5-4-mini-sim-batch-001", True),  # several systems under test
        (f"{STEM}__local-judge", True),
        (f"{STEM}-2-judge", False),  # another run of the same manifest, same second
        (f"{STEM}-2-sim-batch-001", False),
        (f"{STEM}-check", False),
        (STEM, False),
        ("other-judge", False),
    ],
)
def test_named_app_pattern(app_id: str, matches: bool) -> None:
    assert bool(_burr.named_app_pattern(STEM).match(app_id)) is matches


def _app(
    root: Path, app_id: str, *, spawned_by: str | None = None, parent: str | None = None
) -> None:
    folder = root / app_id
    folder.mkdir(parents=True)

    def pointer(value: str | None) -> dict[str, object] | None:
        return {"type": "pointer_data", "app_id": value} if value else None

    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "type": "application_metadata",
                "partition_key": "corallab",
                "parent_pointer": pointer(parent),
                "spawning_parent_pointer": pointer(spawned_by),
            }
        )
    )


def test_attribute_follows_spawning_and_parent_pointers(tmp_path: Path) -> None:
    _app(tmp_path, f"{STEM}-judge")
    _app(tmp_path, "forked", parent=f"{STEM}-judge")
    _app(tmp_path, "nested", spawned_by="forked")
    _app(tmp_path, "loop-a", spawned_by="loop-b")
    _app(tmp_path, "loop-b", spawned_by="loop-a")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "metadata.json").write_text("{not json")
    _app(tmp_path, "under-broken", spawned_by="broken")
    _app(tmp_path, "unrelated-root")
    _app(tmp_path, "unrelated-child", spawned_by="unrelated-root")

    found = _burr.attribute(tmp_path, STEM)
    assert sorted(f.app_id for f, _ in found.apps) == ["forked", "nested", f"{STEM}-judge"]
    unplaced = {f.app_id: why for f, why in found.unattributed}
    assert set(unplaced) == {"loop-a", "loop-b", "broken", "under-broken"}
    assert unplaced["loop-a"].startswith("cycle at")
    assert unplaced["under-broken"].startswith("broken: metadata.json unreadable")


def test_attribute_on_a_missing_tracker(tmp_path: Path) -> None:
    found = _burr.attribute(tmp_path / "none", STEM)
    assert (found.apps, found.unattributed) == ([], [])


def test_persister_helpers_on_a_file_without_the_table(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    assert _burr.persister_app_ids(db) == []
    assert _burr.persister_rows(db, ["x"]) == 0
    assert _burr.delete_persister_rows(db, ["x"]) == 0
    assert _burr.persister_rows(tmp_path / "missing.db", ["x"]) == 0


def test_default_burr_dir_follows_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _burr.default_burr_dir() == tmp_path / ".burr" / "corallab"


@pytest.mark.parametrize(
    ("trace_role", "role"),
    [
        ("sut", "system_under_test"),
        ("person_response_generator", "person_response"),
        ("elicitation_fidelity", "elicitation_fidelity"),
        ("coralmind_s1_ai_behavior-g03", "judge_stage1"),
        ("coralmind_s2_kl-000053,kl-000054", "judge_stage2"),
        ("persona_consistency_checker", "persona_consistency_checker"),
    ],
)
def test_call_role(trace_role: str, role: str) -> None:
    assert _results.call_role(trace_role) == role


def test_trace_calls_skip_events_and_half_written_lines(tmp_path: Path) -> None:
    trace = tmp_path / "x-judge.trace.jsonl"
    trace.write_text(
        json.dumps({"type": "event", "role": "action_load"})
        + "\n"
        + json.dumps({"type": "llm_call", "role": "coralmind_s1_g", "prompt_tokens": 7})
        + "\n"
        + '{"type": "llm_call", "role": "su'
    )
    (only,) = _results.trace_calls([trace])
    assert (only.role, only.prompt_tokens, only.test_case_id) == ("judge_stage1", 7, None)
    assert _results.trace_stage(trace) == "judge"
    assert _results.trace_stage(tmp_path / "x-check.trace.jsonl") == "check"
    assert _results.trace_stage(tmp_path / "x-sim-batch-001.trace.jsonl") == "sim"


def test_find_run_looks_in_results_then_samples(data_repo: DataRepo) -> None:
    def make(root: str, name: str) -> Path:
        return write_run(
            data_repo.root / root,
            name,
            manifest=manifest_data(),
            cases=[],
            batches=[{"transcripts": [transcript("aimhef-0000f001")]}],
        )

    in_results = make("results", "same-20261110-100000")
    make("sample_results", "same-20261110-100000")
    in_samples = make("sample_results", "only-sample-20261110-100000")
    assert _results.find_run("same-20261110-100000") == in_results
    assert _results.find_run("only-sample-20261110-100000") == in_samples
    assert _results.find_run(in_samples) == in_samples
    assert _results.location_of(in_samples) == "sample"
    assert _results.find_run(".agent-runs") is None
    assert _results.find_run("nope") is None
    assert _results.created_at("same-20261110-100000-2") == "2026-11-10T10:00:00"
    assert _results.created_at("no-stamp") is None


def test_child_writes_the_exit_code(tmp_path: Path) -> None:
    exit_file = tmp_path / "exit"
    previous = signal.getsignal(signal.SIGTERM)
    try:
        code = _child.main([str(exit_file), "--", sys.executable, "-c", "raise SystemExit(4)"])
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert code == 4
    assert exit_file.read_text() == "4\n"
    assert _child.main(["only-one"]) == 2
