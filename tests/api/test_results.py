"""summarize_run, get_transcript, get_verdicts and compare_runs on synthetic runs and the
published sample run (read only)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from _records import make_test_case

from corallab.api.results import compare_runs, get_transcript, get_verdicts, summarize_run
from corallab.paths import default_repo_root
from tests.api._repo import DataRepo, manifest_data
from tests.api._runs import (
    BEHAVIOR,
    OTHER,
    TARGET,
    call,
    check,
    judge,
    transcript,
    write_jsonl,
    write_run,
)

SAMPLE = (
    default_repo_root()
    / "data-repository"
    / "sample_results"
    / "l3-launch-entrenchment-smoke-20261019-133536"
)
TC1, TC2 = "aimhef-0000f001", "aimhef-0000f002"


def _cases() -> list[str]:
    return [make_test_case(tc).model_dump_json() for tc in (TC1, TC2)]


def _run(data_repo: DataRepo, name: str, *, supported: bool = True, valid2: bool = True) -> Path:
    """Two test cases, one batch each, checked and judged."""
    t1, t2 = transcript(TC1), transcript(TC2)
    support = "supported" if supported else "not_supported"
    return write_run(
        data_repo.root / "results",
        name,
        manifest=manifest_data(),
        cases=_cases(),
        batches=[
            {
                "transcripts": [t1],
                "checks": [check(t1, adherence=("followed", "trigger_absent"))],
                "judges": [
                    judge(t1, pathways=[(TARGET, support, "met"), (OTHER, "supported", "not_met")])
                ],
                "sim_calls": [call("sut"), call("sut"), call("person_response_generator")],
                "check_calls": [call("elicitation_fidelity", TC1)],
                "judge_calls": [
                    call("coralmind_s1_user_behavior-g01", TC1),
                    call("coralmind_s2_kl-900102", TC1),
                ],
            },
            {
                "transcripts": [t2],
                "checks": [check(t2, "valid" if valid2 else "invalid")],
                "judges": [
                    judge(t2, pathways=[(TARGET, "partially_supported", "met")], observed=())
                ],
                "sim_calls": [call("sut")],
            },
        ],
    )


def test_summarize_run_reports_each_test_case(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    summary = summarize_run("api-test-20261110-100000")

    assert summary.problems == []
    assert summary.location == "results"
    assert summary.path == "results/api-test-20261110-100000"
    assert summary.manifest_run_id == "api-test"
    assert (summary.transcripts, summary.checked, summary.judged) == (2, 2, 2)
    assert summary.fidelity == {"valid": 2}
    first = {c.test_case_id: c for c in summary.test_cases}[TC1]
    assert first.pathway_id == TARGET
    (result,) = first.transcripts
    assert result.batch == "batch-001"
    assert result.exchanges == 2
    assert result.fidelity_status == "valid"
    assert result.fidelity_source == "check_file"
    assert [(s.index, s.adherence) for s in result.strategies] == [
        (1, "followed"),
        (2, "trigger_absent"),
    ]
    assert result.expected_behaviors[0].behavior == BEHAVIOR
    by_id = {p.pathway_id: p for p in result.pathways}
    assert by_id[TARGET].target
    assert by_id[TARGET].applies
    assert by_id[TARGET].gates[0].verdict == "holds"
    assert not by_id[OTHER].applies  # supported, but its gate does not hold
    assert by_id[OTHER].gate_status == "not_met"
    assert [b.behavior_id for b in result.behaviors_found] == [BEHAVIOR]
    assert result.judge_calls == 4
    assert result.judge_tokens is not None
    assert result.judge_tokens.prompt == 100
    assert result.kl_digest is not None
    assert {(c.role, c.calls) for c in result.calls} == {
        ("system_under_test", 2),
        ("person_response", 1),
        ("elicitation_fidelity", 1),
        ("judge_stage1", 1),
        ("judge_stage2", 1),
    }
    assert summary.catalog.judge == [f"{result.kl_version} {result.kl_digest}"]
    assert summary.catalog.test_case_pins == summary.catalog.judge


def test_summarize_run_compares_the_estimate_with_traced_calls(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    estimate = {row.role: row for row in summarize_run("api-test-20261110-100000").estimate}

    sut = estimate["system_under_test"]
    assert (sut.estimated, sut.actual, sut.difference) == (8, 3, -5)  # 2 cases × max_turns 4
    assert sut.upper_bound
    assert sut.note == "within the upper bound"
    person = estimate["person_response"]
    assert (person.estimated, person.actual) == (6, 1)
    fidelity = estimate["elicitation_fidelity"]
    assert (fidelity.estimated, fidelity.actual, fidelity.difference) == (2, 1, -1)
    assert fidelity.note == "fewer than estimated: failed calls are not traced"
    assert estimate["judge_stage1"].actual == 1
    assert not estimate["judge_stage1"].upper_bound


def test_summarize_run_reads_an_unchecked_unjudged_run(data_repo: DataRepo) -> None:
    t = transcript(TC1)
    write_run(
        data_repo.root / "results",
        "api-test-20261110-110000",
        manifest=manifest_data(),
        cases=_cases()[:1],
        batches=[{"transcripts": [t], "sim_calls": [call("sut")]}],
    )
    summary = summarize_run("api-test-20261110-110000")
    assert summary.fidelity == {"unchecked": 1}
    assert summary.judged == 0
    (case,) = summary.test_cases
    assert case.transcripts[0].judge_status is None
    rows = {r.role: r for r in summary.estimate}
    assert rows["judge_stage1"].note == "no judge output in this run"
    assert rows["elicitation_fidelity"].note == "no check output in this run"


def test_summarize_run_lists_lines_that_do_not_validate(data_repo: DataRepo) -> None:
    run = _run(data_repo, "api-test-20261110-100000")
    sim = run / "batch-002" / "api-test-20261110-100000-sim.jsonl"
    sim.write_text(sim.read_text() + '{"transcript_id": "broken"}\n')
    summary = summarize_run("api-test-20261110-100000")
    assert summary.transcripts == 2
    assert summary.problems == [
        "batch-002/api-test-20261110-100000-sim.jsonl:2: not a Transcript (5 error(s))"
    ]


def test_summarize_run_takes_the_latest_judge_result(data_repo: DataRepo) -> None:
    run = _run(data_repo, "api-test-20261110-100000")
    t1 = transcript(TC1)
    write_jsonl(
        run / "batch-001" / "api-test-20261110-100000-judge.jsonl",
        [
            judge(
                t1,
                evaluated_at="2026-11-10T12:00:00+00:00",
                pathways=[(TARGET, "not_supported", "met")],
            ),
            judge(t1, evaluated_at="2026-11-10T09:00:00+00:00"),
        ],
    )
    (result,) = {c.test_case_id: c for c in summarize_run(run).test_cases}[TC1].transcripts
    assert [p.support for p in result.pathways] == ["not_supported"]


def test_summarize_run_reports_a_missing_run(data_repo: DataRepo) -> None:
    assert summarize_run("nope").problems == ["no run named nope"]


def test_get_transcript_returns_turns_and_fidelity(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    found = get_transcript("api-test-20261110-100000", TC2)
    (view,) = found.transcripts
    assert view.batch == "batch-002"
    assert view.transcript.turns is not None
    assert len(view.transcript.turns) == 4
    assert view.fidelity is not None
    assert view.fidelity.status == "valid"
    assert view.fidelity_source == "check_file"
    assert get_transcript("api-test-20261110-100000", TC2, system_under_test="x").transcripts == []
    assert get_transcript("api-test-20261110-100000", "aimhef-0000ffff").problems == [
        "no transcript of aimhef-0000ffff in api-test-20261110-100000"
    ]


def test_get_verdicts_drops_the_conversation_and_raw_output(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    verdicts = get_verdicts("api-test-20261110-100000", TC1)
    (item,) = verdicts.verdicts
    assert item.fidelity is not None
    assert item.fidelity.status == "valid"
    assert item.judge is not None
    assert item.judge.conversation_transcript is None
    assert item.judge.coralmind_run is not None
    assert item.judge.coralmind_run.stage1.calls[0].raw_response is None
    raw = get_verdicts("api-test-20261110-100000", TC1, include_raw=True).verdicts[0].judge
    assert raw is not None
    assert raw.coralmind_run is not None
    assert raw.coralmind_run.stage1.calls[0].raw_response == "RAW"
    assert len(get_verdicts("api-test-20261110-100000").verdicts) == 2


def test_compare_runs_puts_rates_side_by_side(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    _run(data_repo, "api-test-20261110-120000", supported=False, valid2=False)
    comparison = compare_runs("api-test-20261110-100000", "api-test-20261110-120000")

    assert comparison.problems == []
    assert (comparison.a.counted, comparison.b.counted) == (2, 2)
    assert comparison.b.fidelity == {"valid": 1, "invalid": 1}
    target = {p.pathway_id: p for p in comparison.pathways}[TARGET]
    assert (target.a.applies, target.a.rate) == (1, 0.5)
    assert (target.b.applies, target.b.rate) == (0, 0.0)
    assert target.difference == -0.5
    assert (target.a.supported, target.a.partially_supported) == (1, 1)
    assert (target.b.not_supported, target.b.partially_supported) == (1, 1)
    other = {p.pathway_id: p for p in comparison.pathways}[OTHER]
    assert other.a.applies == 0
    assert other.a.supported == 1
    (behavior,) = comparison.behaviors
    assert (behavior.behavior_id, behavior.a.rate, behavior.b.rate) == (BEHAVIOR, 0.5, 0.5)
    assert comparison.notes == []


def test_compare_runs_can_count_only_valid_transcripts(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    _run(data_repo, "api-test-20261110-120000", valid2=False)
    comparison = compare_runs(
        "api-test-20261110-100000", "api-test-20261110-120000", fidelity_valid_only=True
    )
    assert (comparison.a.counted, comparison.b.counted) == (2, 1)
    target = {p.pathway_id: p for p in comparison.pathways}[TARGET]
    assert target.b.rate == 1.0


def test_compare_runs_notes_what_makes_rates_less_comparable(data_repo: DataRepo) -> None:
    _run(data_repo, "api-test-20261110-100000")
    t = transcript(TC1)
    write_run(
        data_repo.root / "results",
        "api-test-20261110-130000",
        manifest=manifest_data(),
        cases=_cases()[:1],
        batches=[{"transcripts": [t], "judges": [judge(t, model="local/model")]}],
    )
    comparison = compare_runs("api-test-20261110-100000", "api-test-20261110-130000")
    assert comparison.test_cases_only_in_a == [TC2]
    assert "judge_models differ: remote/model vs local/model" in comparison.notes
    assert "the runs hold different test cases; rates are over different sets" in comparison.notes
    assert compare_runs("api-test-20261110-100000", "nope").problems == ["no run named nope"]


# ---------------------------------------------------------------------------
# The published sample run, read only
# ---------------------------------------------------------------------------


def _digests(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="the sample run is not in this checkout")
def test_sample_run_reads_without_writing(data_repo: DataRepo) -> None:
    before = _digests(SAMPLE)
    summary = summarize_run(SAMPLE)
    transcripts = get_transcript(SAMPLE, "aimhef-32aaeebe")
    verdicts = get_verdicts(SAMPLE)
    comparison = compare_runs(SAMPLE, SAMPLE)
    assert _digests(SAMPLE) == before

    assert summary.problems == []
    assert summary.manifest_run_id == "l3-launch-entrenchment-smoke"
    assert (summary.transcripts, summary.checked, summary.judged) == (1, 1, 0)
    (case,) = summary.test_cases
    (result,) = case.transcripts
    assert result.fidelity_source == "transcript"  # written before the check had its own file
    assert result.fidelity_status == "valid"
    assert result.exchanges == 10
    estimate = {r.role: r for r in summary.estimate}
    assert (estimate["system_under_test"].estimated, estimate["system_under_test"].actual) == (
        10,
        10,
    )
    assert (estimate["person_response"].estimated, estimate["person_response"].actual) == (9, 9)
    assert estimate["elicitation_fidelity"].note == "exact"
    assert estimate["judge_stage1"].note == "no judge output in this run"
    assert summary.catalog.produced_by == ["coralmap 0.2.0"]
    assert len(transcripts.transcripts) == 1
    assert verdicts.verdicts[0].judge is None
    assert comparison.a.counted == 0
