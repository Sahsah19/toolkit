"""Reading a finished (or running) run: summaries, transcripts, verdicts and comparisons.

Every record is read with the toolkit's own schema (``Transcript``, ``FidelityCheckRecord``,
``EvalResult``); a line that does not validate is listed in ``problems``, not dropped silently.
Nothing here writes. A run is named by its id (the results folder's name) or a path to it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from corallab.api._catalog import load_catalog
from corallab.api._manifests import relative_to_data
from corallab.api._results import (
    RunFiles,
    TraceCall,
    batch_name,
    count_calls,
    find_run,
    location_of,
    read_records,
    run_files,
    run_manifest,
    trace_calls,
    trace_stage,
)
from corallab.api.listing import read_test_case_summaries
from corallab.api.models import (
    BehaviorComparison,
    BehaviorFound,
    BehaviorRate,
    CallComparison,
    CaseResult,
    CatalogUse,
    ExpectedBehaviorResult,
    GateResult,
    PathwayComparison,
    PathwayRate,
    PathwayResult,
    RunComparison,
    RunHeader,
    RunSummary,
    StrategyResult,
    TestCaseSummary,
    TokenCount,
    TranscriptResult,
    TranscriptsResult,
    TranscriptVerdicts,
    TranscriptView,
    VerdictsResult,
)
from corallab.api.preflight import estimate_calls
from corallab.schemas.eval.run_manifest import RunManifest
from corallab.schemas.eval.transcript import FidelityCheckRecord, FidelityVerdict, Transcript
from corallab.schemas.judge.coralmind_result import PathwayAssessment
from corallab.schemas.judge.eval_result import EvalResult

GATE_APPLIES = ("met", "no_gates")
"""Gate statuses under which a pathway applies; the viewer's rule."""


def pathway_applies(assessment: PathwayAssessment) -> bool:
    """A supported instance (or, with none, a supported verdict) and every gate holding."""
    supported = any(i.support == "supported" for i in assessment.instances) or (
        not assessment.instances and assessment.support == "supported"
    )
    return supported and assessment.gate_status in GATE_APPLIES


def judged_ok(result: EvalResult | None) -> bool:
    return (
        result is not None
        and result.status == "success"
        and result.coralmind_run is not None
        and result.coralmind_run.run_status == "success"
    )


# ---------------------------------------------------------------------------
# Loading one run
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    directory: Path
    files: RunFiles
    manifest: RunManifest | None
    transcripts: list[tuple[str, Transcript]] = field(default_factory=list)
    """(batch, transcript) in file order."""
    checks: dict[str, FidelityCheckRecord] = field(default_factory=dict)
    judges: dict[str, EvalResult] = field(default_factory=dict)
    """The latest judge result per transcript."""
    cases: list[TestCaseSummary] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.directory.name

    def fidelity(self, t: Transcript) -> tuple[FidelityVerdict | None, str | None, str | None]:
        """(verdict, where it came from, why the check gave none)."""
        check = self.checks.get(t.transcript_id)
        if check is not None:
            source = "check_file" if check.fidelity is not None else None
            return check.fidelity, source, check.error
        if t.fidelity is not None:
            return t.fidelity, "transcript", None
        return None, None, None


def _load(run: str | Path) -> _Run | None:
    directory = find_run(run)
    if directory is None:
        return None
    files = run_files(directory)
    manifest, problem = run_manifest(files)
    data = _Run(directory=directory, files=files, manifest=manifest)
    if problem:
        data.problems.append(problem)
    transcripts, problems = read_records(files.sim, Transcript, directory)
    data.transcripts = [(batch_name(p, directory), t) for p, t in transcripts]
    checks, more = read_records(files.check, FidelityCheckRecord, directory)
    data.checks = {c.transcript_id: c for _, c in checks}
    judges, even_more = read_records(files.judge, EvalResult, directory)
    for _, result in sorted(judges, key=lambda pr: pr[1].evaluated_at):
        data.judges[result.transcript_id] = result
    data.problems += problems + more + even_more
    if files.inputs_records is not None:
        data.cases = read_test_case_summaries(files.inputs_records)
        data.problems += [
            f"inputs/{files.inputs_records.name}:{s.line}: {s.error}" for s in data.cases if s.error
        ]
    else:
        data.problems.append("no test case records in inputs/")
    return data


def _missing(run: str | Path) -> str:
    return f"no run named {run}"


# ---------------------------------------------------------------------------
# summarize_run
# ---------------------------------------------------------------------------


def _pathway(assessment: PathwayAssessment, target: str | None) -> PathwayResult:
    return PathwayResult(
        pathway_id=assessment.pathway_id,
        pathway_did=assessment.pathway_did,
        label=assessment.label,
        target=assessment.pathway_id == target,
        support=assessment.support,
        gate_status=assessment.gate_status,
        applies=pathway_applies(assessment),
        instances=len(assessment.instances),
        gates=[
            GateResult(
                gate=g.gate,
                part_id=g.part_id,
                label=g.label,
                verdict=g.verdict,
                basis=g.basis,
                conflict=g.conflict,
            )
            for g in assessment.gates
        ],
    )


def _transcript_result(
    data: _Run,
    batch: str,
    t: Transcript,
    target: str | None,
    calls: list[TraceCall],
) -> TranscriptResult:
    verdict, source, error = data.fidelity(t)
    out = TranscriptResult(
        transcript_id=t.transcript_id,
        system_under_test=t.system_under_test,
        sample=t.sample,
        batch=batch,
        exchanges=max((turn.turn_number for turn in t.turns or []), default=0),
        exit_reason=str(t.exit_reason) if t.exit_reason is not None else None,
        sim_failure=t.sim_failure.reason if t.sim_failure is not None else None,
        fidelity_status=verdict.status if verdict is not None else None,
        fidelity_source=source,  # type: ignore[arg-type]
        fidelity_error=error,
        calls=count_calls(calls),
    )
    if verdict is not None:
        out.strategies = [
            StrategyResult(index=s.index, adherence=s.adherence) for s in verdict.strategies
        ]
        out.expected_behaviors = [
            ExpectedBehaviorResult(
                behavior=b.behavior,
                catalog_occurred=b.catalog_occurred,
                assertion_occurred=b.assertion_occurred,
                miss=b.miss,
            )
            for b in verdict.behaviors
        ]
    result = data.judges.get(t.transcript_id)
    if result is None:
        return out
    out.judge_status = result.status
    out.judge_model = result.judge_model_id
    run = result.coralmind_run
    if run is None:
        return out
    if run.run_status != "success":
        out.judge_status = run.run_status
    meta = run.run_meta
    out.prompt_version = meta.prompt_version
    out.kl_version, out.kl_digest = meta.kl.version, meta.kl.digest
    out.pathways = [_pathway(a, target) for a in run.stage2.assessments]
    out.pathways_skipped = len(run.stage2.skipped)
    assessed = {a.pathway_id for a in run.stage2.assessments}
    out.pathways_failed = sorted(
        {
            p
            for call in run.stage2.calls
            if call.status == "judge_error"
            for p in call.pathway_ids
            if p not in assessed
        }
    )
    out.behaviors_found = [
        BehaviorFound(
            behavior_id=o.behavior_id,
            behavior_did=o.behavior_did,
            label=o.label,
            kind=o.kind,
            turns=o.turns,
        )
        for o in run.stage1.observations
    ]
    counts = meta.counts
    if "stage1_calls" in counts or "stage2_calls" in counts:
        out.judge_calls = counts.get("stage1_calls", 0) + counts.get("stage2_calls", 0)
    out.judge_retries = counts.get("retries")
    out.judge_call_failures = counts.get("call_failures")
    if meta.tokens is not None:
        out.judge_tokens = TokenCount(
            prompt=meta.tokens.get("prompt", 0),
            completion=meta.tokens.get("completion", 0),
            cached=meta.tokens.get("cached", 0),
        )
    return out


def _calls_for(data: _Run, calls: list[TraceCall]) -> dict[str, list[TraceCall]]:
    """Trace calls per transcript id, where a call can be tied to one transcript.

    A row naming a test case belongs to the transcript of that test case in the trace's batch
    folder, when there is only one. Sim rows do not name a test case; they belong to the batch
    folder's transcript when it holds only one.
    """
    by_batch: dict[str, list[Transcript]] = defaultdict(list)
    for batch, t in data.transcripts:
        by_batch[batch].append(t)
    out: dict[str, list[TraceCall]] = defaultdict(list)
    for call in calls:
        batch = batch_name(call.trace, data.directory)
        here = by_batch.get(batch, [])
        if call.test_case_id is not None:
            matching = [t for t in here if t.test_case_id == call.test_case_id]
        elif trace_stage(call.trace) == "sim":
            matching = here
        else:
            matching = []
        if len(matching) == 1:
            out[matching[0].transcript_id].append(call)
    return out


def _compare_estimate(
    manifest: RunManifest | None,
    cases: list[TestCaseSummary],
    calls: list[TraceCall],
    stages_run: set[str],
) -> tuple[list[CallComparison], str | None, list[str]]:
    """Preflight's estimate on the run's own inputs against the traced calls, per role.

    Returns (rows, the library loaded for the estimate as 'version digest', notes).
    """
    if manifest is None or not any(c.error is None for c in cases):
        return [], None, ["no estimate: the run's inputs/ lack a valid manifest or test cases"]
    library, problem = load_catalog()
    notes = [] if problem is None else [f"judge calls not estimated: {problem}"]
    kg = None
    if library is not None:
        from corallab.judging.kl_index import LibraryIndex

        kg = LibraryIndex(library)
    estimated: dict[str, list] = defaultdict(list)
    for row in estimate_calls(manifest, cases, kg):
        estimated[row.role].append(row)
    actual = Counter(c.role for c in calls)
    rows: list[CallComparison] = []
    for role in list(estimated) + [r for r in actual if r not in estimated]:
        est = estimated.get(role, [])
        total = sum(r.calls for r in est)
        upper = any(r.upper_bound for r in est)
        got = actual.get(role, 0)
        if not est:
            note = "not in the estimate"
        elif got == 0 and _stage_of(role) not in stages_run:
            note = f"no {_stage_of(role)} output in this run"
        elif got == total:
            note = "at the upper bound" if upper else "exact"
        elif upper and got < total:
            note = "within the upper bound"
        elif got > total:
            note = "more than estimated: retries, follow-up calls or re-run batches"
        else:
            note = "fewer than estimated: failed calls are not traced"
        rows.append(
            CallComparison(
                role=role,
                estimated=total,
                upper_bound=upper,
                actual=got,
                difference=got - total,
                basis="; ".join(r.basis for r in est),
                note=note,
            )
        )
    loaded = f"{library.version} {library.digest}" if library is not None else None
    return rows, loaded, notes


def _stage_of(role: str) -> str:
    if role.startswith("judge"):
        return "judge"
    if role == "elicitation_fidelity":
        return "check"
    return "sim"


def summarize_run(run_id: str | Path) -> RunSummary:
    """Per test case: fidelity status, strategy verdicts, pathway verdicts with their gate
    assessments, behaviors found, calls and tokens, and the catalog each part names. Run-wide:
    traced calls and tokens, and preflight's call estimate (worked out again on the run's own
    copied inputs) against the calls the run traced, with the difference.
    """
    data = _load(run_id)
    if data is None:
        return RunSummary(
            run_id=str(run_id), path="", location="elsewhere", problems=[_missing(run_id)]
        )
    calls = trace_calls(data.files.traces)
    per_transcript = _calls_for(data, calls)
    targets = {c.id: c.pathway_id for c in data.cases if c.error is None}

    grouped: dict[str, list[TranscriptResult]] = defaultdict(list)
    for batch, t in data.transcripts:
        grouped[t.test_case_id].append(
            _transcript_result(
                data, batch, t, targets.get(t.test_case_id), per_transcript[t.transcript_id]
            )
        )
    fidelity = Counter(r.fidelity_status or "unchecked" for rs in grouped.values() for r in rs)

    stages_run = {"sim"} if data.files.sim else set()
    if data.checks or any(t.fidelity is not None for _, t in data.transcripts):
        stages_run.add("check")
    if data.judges:
        stages_run.add("judge")
    estimate, loaded, notes = _compare_estimate(data.manifest, data.cases, calls, stages_run)

    judge_pins = sorted(
        {
            f"{r.coralmind_run.run_meta.kl.version} {r.coralmind_run.run_meta.kl.digest}"
            for r in data.judges.values()
            if r.coralmind_run is not None
        }
    )
    produced = sorted(
        {
            f"{p.knowledge_library.name} {p.knowledge_library.version}"
            for p in [t.produced_by for _, t in data.transcripts]
            + [r.produced_by for r in data.judges.values()]
            if p is not None
        }
    )
    pins = sorted(
        {f"{c.kl_version} {c.kl_digest}" for c in data.cases if c.error is None and c.kl_digest}
    )
    used = judge_pins or pins
    if loaded and used and loaded not in used:
        notes.append(
            f"the judge estimate counts definition groups of the library loaded now ({loaded}); "
            f"the run used {', '.join(used)}"
        )

    manifest = data.manifest
    summary_calls = count_calls(calls)
    return RunSummary(
        run_id=data.run_id,
        path=relative_to_data(data.directory),
        location=location_of(data.directory),
        manifest_run_id=manifest.run_id if manifest is not None else None,
        evaluation_level=manifest.evaluation_level if manifest is not None else None,
        system_under_test=[s.model_id for s in manifest.system_under_test] if manifest else [],
        judge_models=[j.model.model_id for j in manifest.judges if j.enabled] if manifest else [],
        transcripts=len(data.transcripts),
        checked=sum(1 for _, t in data.transcripts if data.fidelity(t)[0] is not None),
        judged=sum(1 for _, t in data.transcripts if judged_ok(data.judges.get(t.transcript_id))),
        fidelity=dict(fidelity),
        test_cases=[
            CaseResult(test_case_id=tc, pathway_id=targets.get(tc), transcripts=results)
            for tc, results in grouped.items()
        ],
        calls=summary_calls,
        tokens=TokenCount(
            prompt=sum(c.prompt_tokens for c in summary_calls),
            completion=sum(c.completion_tokens for c in summary_calls),
            cached=sum(c.cached_tokens for c in summary_calls),
        ),
        catalog=CatalogUse(
            test_case_pins=pins, judge=judge_pins, produced_by=produced, loaded=loaded
        ),
        estimate=estimate,
        notes=notes,
        problems=data.problems,
    )


# ---------------------------------------------------------------------------
# get_transcript, get_verdicts
# ---------------------------------------------------------------------------


def get_transcript(
    run_id: str | Path,
    test_case_id: str,
    *,
    system_under_test: str | None = None,
    sample: int | None = None,
) -> TranscriptsResult:
    """Every transcript of one test case in a run (one per system under test and sample),
    with its turns and its fidelity verdict. ``system_under_test`` (the output key, e.g.
    ``azure-gpt-5-4-mini``) and ``sample`` narrow the list."""
    data = _load(run_id)
    if data is None:
        return TranscriptsResult(
            run_id=str(run_id), test_case_id=test_case_id, problems=[_missing(run_id)]
        )
    views: list[TranscriptView] = []
    for batch, t in data.transcripts:
        if t.test_case_id != test_case_id:
            continue
        if system_under_test is not None and t.system_under_test != system_under_test:
            continue
        if sample is not None and t.sample != sample:
            continue
        verdict, source, _ = data.fidelity(t)
        views.append(
            TranscriptView(
                batch=batch,
                transcript=t,
                fidelity=verdict,
                fidelity_source=source,  # type: ignore[arg-type]
            )
        )
    problems = list(data.problems)
    if not views:
        problems.append(f"no transcript of {test_case_id} in {data.run_id}")
    return TranscriptsResult(
        run_id=data.run_id, test_case_id=test_case_id, transcripts=views, problems=problems
    )


def _slim(result: EvalResult, include_raw: bool) -> EvalResult:
    """The judge result without its copy of the conversation, and without raw model output
    unless ``include_raw``."""
    slim = result.model_copy(update={"conversation_transcript": None}, deep=True)
    if not include_raw and slim.coralmind_run is not None:
        for call in slim.coralmind_run.stage1.calls:
            call.raw_response = None
        for call2 in slim.coralmind_run.stage2.calls:
            call2.raw_response = None
        slim.judge_prompt_snapshot = None
    return slim


def get_verdicts(
    run_id: str | Path, test_case_id: str | None = None, *, include_raw: bool = False
) -> VerdictsResult:
    """The fidelity verdict and judge result of each transcript, or of one test case's.

    The judge result is the full ``EvalResult`` without the conversation (``get_transcript``
    has it); raw model responses and prompt snapshots only with ``include_raw``.
    """
    data = _load(run_id)
    if data is None:
        return VerdictsResult(run_id=str(run_id), problems=[_missing(run_id)])
    verdicts: list[TranscriptVerdicts] = []
    for _, t in data.transcripts:
        if test_case_id is not None and t.test_case_id != test_case_id:
            continue
        verdict, _, error = data.fidelity(t)
        judge = data.judges.get(t.transcript_id)
        verdicts.append(
            TranscriptVerdicts(
                transcript_id=t.transcript_id,
                test_case_id=t.test_case_id,
                system_under_test=t.system_under_test,
                fidelity=verdict,
                fidelity_error=error,
                judge=_slim(judge, include_raw) if judge is not None else None,
            )
        )
    problems = list(data.problems)
    if test_case_id is not None and not verdicts:
        problems.append(f"no transcript of {test_case_id} in {data.run_id}")
    return VerdictsResult(run_id=data.run_id, verdicts=verdicts, problems=problems)


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


@dataclass
class _Rates:
    header: RunHeader
    pathways: dict[str, PathwayRate]
    labels: dict[str, str]
    behaviors: dict[str, int]
    behavior_info: dict[str, tuple[str, str]]
    test_cases: set[str]


def _rates(data: _Run, fidelity_valid_only: bool) -> _Rates:
    counted: list[EvalResult] = []
    fidelity: Counter[str] = Counter()
    for _, t in data.transcripts:
        verdict = data.fidelity(t)[0]
        fidelity[verdict.status if verdict is not None else "unchecked"] += 1
        result = data.judges.get(t.transcript_id)
        if not judged_ok(result):
            continue
        if fidelity_valid_only and (verdict is None or verdict.status != "valid"):
            continue
        assert result is not None
        counted.append(result)

    n = len(counted)
    pathways: dict[str, PathwayRate] = {}
    labels: dict[str, str] = {}
    behaviors: Counter[str] = Counter()
    info: dict[str, tuple[str, str]] = {}
    for result in counted:
        run = result.coralmind_run
        assert run is not None
        for a in run.stage2.assessments:
            labels[a.pathway_id] = a.label
            rate = pathways.setdefault(a.pathway_id, PathwayRate(judged=n, assessed=0))
            rate.assessed += 1
            setattr(rate, a.support, getattr(rate, a.support) + 1)
            rate.applies += pathway_applies(a)
        for behavior in {o.behavior_id for o in run.stage1.observations}:
            behaviors[behavior] += 1
        for o in run.stage1.observations:
            info[o.behavior_id] = (o.label, o.kind)
    for rate in pathways.values():
        rate.rate = rate.applies / n if n else None
    metas = [r.coralmind_run.run_meta for r in data.judges.values() if r.coralmind_run]
    manifest = data.manifest
    header = RunHeader(
        run_id=data.run_id,
        manifest_run_id=manifest.run_id if manifest is not None else None,
        system_under_test=[s.model_id for s in manifest.system_under_test] if manifest else [],
        judge_models=sorted({m.model for m in metas}),
        prompt_versions=sorted({m.prompt_version for m in metas}),
        catalog=sorted({f"{m.kl.version} {m.kl.digest}" for m in metas}),
        transcripts=len(data.transcripts),
        judged=sum(1 for r in data.judges.values() if judged_ok(r)),
        counted=n,
        fidelity=dict(fidelity),
    )
    return _Rates(
        header=header,
        pathways=pathways,
        labels=labels,
        behaviors=dict(behaviors),
        behavior_info=info,
        test_cases={t.test_case_id for _, t in data.transcripts},
    )


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def compare_runs(
    a: str | Path, b: str | Path, *, fidelity_valid_only: bool = False
) -> RunComparison:
    """Pathway support and behavior rates of two runs side by side.

    A rate counts each run's successfully judged transcripts (with ``fidelity_valid_only``,
    only those whose fidelity check is ``valid``). A pathway applies when it has a supported
    instance and every gate holds, as the viewer counts it; a behavior is found when Stage 1
    observed it. ``notes`` names differences in judge model, prompt, catalog or test cases
    that make the rates less comparable.
    """
    first, second = _load(a), _load(b)
    missing = [_missing(r) for r, d in ((a, first), (b, second)) if d is None]
    if first is None or second is None:
        empty = RunHeader(run_id="", counted=0)
        return RunComparison(a=empty, b=empty, problems=missing)
    ra, rb = _rates(first, fidelity_valid_only), _rates(second, fidelity_valid_only)

    pathways = []
    for pid in sorted(set(ra.pathways) | set(rb.pathways)):
        pa = ra.pathways.get(pid, PathwayRate(judged=ra.header.counted, assessed=0))
        pb = rb.pathways.get(pid, PathwayRate(judged=rb.header.counted, assessed=0))
        for rate, n in ((pa, ra.header.counted), (pb, rb.header.counted)):
            rate.rate = rate.applies / n if n else None
        pathways.append(
            PathwayComparison(
                pathway_id=pid,
                label=ra.labels.get(pid) or rb.labels.get(pid, ""),
                a=pa,
                b=pb,
                difference=_difference(pa.rate, pb.rate),
            )
        )
    behaviors = []
    for bid in sorted(set(ra.behaviors) | set(rb.behaviors)):
        ba = _behavior_rate(ra.behaviors.get(bid, 0), ra.header.counted)
        bb = _behavior_rate(rb.behaviors.get(bid, 0), rb.header.counted)
        label, kind = ra.behavior_info.get(bid) or rb.behavior_info.get(bid, ("", ""))
        behaviors.append(
            BehaviorComparison(
                behavior_id=bid,
                label=label,
                kind=kind,
                a=ba,
                b=bb,
                difference=_difference(ba.rate, bb.rate),
            )
        )

    notes = []
    for what in ("judge_models", "prompt_versions", "catalog"):
        va, vb = getattr(ra.header, what), getattr(rb.header, what)
        if va and vb and va != vb:
            notes.append(f"{what} differ: {', '.join(va)} vs {', '.join(vb)}")
    if ra.test_cases != rb.test_cases:
        notes.append("the runs hold different test cases; rates are over different sets")
    notes += [
        f"{h.run_id} has no judged transcript to count"
        for h in (ra.header, rb.header)
        if h.counted == 0
    ]
    return RunComparison(
        a=ra.header,
        b=rb.header,
        pathways=pathways,
        behaviors=behaviors,
        test_cases_only_in_a=sorted(ra.test_cases - rb.test_cases),
        test_cases_only_in_b=sorted(rb.test_cases - ra.test_cases),
        notes=notes,
        problems=[f"{first.run_id}: {p}" for p in first.problems]
        + [f"{second.run_id}: {p}" for p in second.problems],
    )


def _behavior_rate(found: int, judged: int) -> BehaviorRate:
    return BehaviorRate(found=found, judged=judged, rate=found / judged if judged else None)
