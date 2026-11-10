"""Synthetic results folders for the run API tests, built from the toolkit's own schemas.

Every record is validated with ``Transcript``, ``FidelityCheckRecord`` or ``EvalResult`` before
it is written, so a builder that drifts from the schema fails here and not in the code tested.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml
from _records import fixture_pin

from corallab.schemas.eval.transcript import FidelityCheckRecord, Transcript
from corallab.schemas.judge.eval_result import EvalResult

SUT_KEY = "remote-model"
TARGET = "kl-900102"
OTHER = "kl-900101"
BEHAVIOR = "kl-900501"


def turns(n: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for number in range(1, n + 1):
        out.append({"turn_number": number, "role": "user", "content": f"hello {number}"})
        out.append({"turn_number": number, "role": "assistant", "content": f"reply {number}"})
    return out


def fidelity(status: str = "valid", adherence: Sequence[str] = ("followed",)) -> dict[str, Any]:
    cite = {"turn": 1, "speaker": "user", "quote": "hello 1"}
    occurred = status != "invalid"
    behavior: dict[str, Any] = {
        "behavior": BEHAVIOR,
        "catalog_occurred": occurred,
        "assertion_occurred": occurred,
        "citations": [cite] if occurred else [],
    }
    if not occurred:
        behavior["miss"] = "not_attempted"
    return {
        "checker": {"model": "local/model", "prompt_version": "elicitation-fidelity-v2"},
        "behaviors": [behavior],
        "strategies": [
            {"index": i, "adherence": a, "citations": [cite] if a == "followed" else []}
            for i, a in enumerate(adherence, start=1)
        ],
        "status": status,
        "reasons": [] if occurred else [{"behavior": BEHAVIOR, "miss": "not_attempted"}],
    }


def transcript(
    test_case_id: str,
    *,
    run_id: str = "api-test",
    sample: int = 1,
    exchanges: int = 2,
    sut_key: str = SUT_KEY,
    sim_failure: str | None = None,
    inline_fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "transcript_id": f"{run_id}:{sut_key}:{test_case_id}:{sample}",
        "test_case_id": test_case_id,
        "test_case_version": "1.0.0",
        "run_id": run_id,
        "system_under_test": sut_key,
        "sample": sample,
        "turns": None if sim_failure else turns(exchanges),
        "exit_reason": None if sim_failure else "max_turns",
        "sim_failure": {"reason": sim_failure, "detail": "boom"} if sim_failure else None,
        "fidelity": inline_fidelity,
    }
    return Transcript.model_validate(record).model_dump(mode="json")


def check(t: dict[str, Any], status: str = "valid", **kwargs: Any) -> dict[str, Any]:
    record = {
        "transcript_id": t["transcript_id"],
        "test_case_id": t["test_case_id"],
        "system_under_test": t["system_under_test"],
        "fidelity": fidelity(status, **kwargs) if status else None,
        "error": None if status else "the check could not be completed; see the log",
    }
    return FidelityCheckRecord.model_validate(record).model_dump(mode="json")


def _assessment(pathway_id: str, support: str, gate_status: str) -> dict[str, Any]:
    instances = []
    if support in ("supported", "partially_supported"):
        instances.append(
            {
                "instance_id": f"{pathway_id}#1",
                "turn_window": [1, 2],
                "support": support,
                "uncertainty": 0.1,
            }
        )
    gates = []
    if gate_status != "no_gates":
        verdict = {"met": "holds", "not_met": "does_not_hold", "unknown": "unknown"}[gate_status]
        gates.append(
            {
                "gate": "G1",
                "part_id": "kl-900301",
                "label": "a gate",
                "kind": "context_value",
                "recorded": verdict,
                "inferred": verdict,
                "verdict": verdict,
                "basis": "test_case",
            }
        )
    return {
        "pathway_id": pathway_id,
        "pathway_did": f"Pathway-{pathway_id[-3:]}",
        "label": f"pathway {pathway_id}",
        "instances": instances,
        "support": support,
        "uncertainty": 0.1,
        "gates": gates,
        "gate_status": gate_status,
    }


def judge(
    t: dict[str, Any],
    *,
    pathways: Iterable[tuple[str, str, str]] = ((TARGET, "supported", "met"),),
    observed: Iterable[str] = (BEHAVIOR,),
    status: str = "success",
    model: str = "remote/model",
    prompt_version: str = "v1",
    evaluated_at: str = "2026-11-10T10:00:00+00:00",
    stage1_calls: int = 3,
    stage2_calls: int = 1,
) -> dict[str, Any]:
    pin = fixture_pin()
    record = {
        "test_case_id": t["test_case_id"],
        "test_case_version": "1.0.0",
        "transcript_id": t["transcript_id"],
        "run_id": t["run_id"],
        "system_under_test": t["system_under_test"],
        "evaluation_level": 3,
        "evaluation_id": f"eval-{t['transcript_id']}",
        "evaluated_model_id": "remote/model",
        "evaluated_at": evaluated_at,
        "judge_model_id": model,
        "status": status,
        "conversation_transcript": t["turns"],
        "coralmind_run": {
            "run_status": "success" if status == "success" else "judge_error",
            "stage1": {
                "observations": [
                    {
                        "behavior_id": b,
                        "behavior_did": f"UserBehavior-{b[-3:]}",
                        "label": f"behavior {b}",
                        "kind": "user_behavior",
                        "turns": [1],
                        "uncertainty": 0.1,
                        "group_id": "user_behavior-g01",
                    }
                    for b in observed
                ],
                "calls": [
                    {
                        "group_id": "user_behavior-g01",
                        "kind": "user_behavior",
                        "status": "success",
                        "raw_response": "RAW",
                    }
                ],
            },
            "stage2": {"assessments": [_assessment(*p) for p in pathways]},
            "run_meta": {
                "run_id": f"judge-{t['transcript_id']}",
                "started_at": evaluated_at,
                "finished_at": evaluated_at,
                "elapsed_ms": 1000,
                "mode": "live",
                "test_case_id": t["test_case_id"],
                "transcript_hash": "sha256:0",
                "transcript_turns": len(t["turns"] or []),
                "kl": {"version": pin.version, "digest": pin.digest},
                "model": model,
                "prompt_version": prompt_version,
                "prompt_hash": "sha256:1",
                "counts": {
                    "stage1_calls": stage1_calls,
                    "stage2_calls": stage2_calls,
                    "retries": 0,
                    "call_failures": 0,
                },
                "tokens": {"prompt": 100, "completion": 20, "cached": 0},
            },
        },
    }
    return EvalResult.model_validate(record).model_dump(mode="json")


def call(role: str, test_case_id: str | None = None, model: str = "azure/model") -> dict[str, Any]:
    return {
        "type": "llm_call",
        "role": role,
        "test_case_id": test_case_id,
        "model": model,
        "resolved_litellm_model": model,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cached_tokens": 0,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def write_run(
    results: Path,
    name: str,
    *,
    manifest: dict[str, Any],
    cases: Sequence[str],
    batches: Sequence[dict[str, Any]],
) -> Path:
    """A results folder as ``corallab run`` lays it out.

    Each batch is a mapping with ``transcripts`` and optional ``checks``, ``judges``, and
    ``sim_calls`` / ``check_calls`` / ``judge_calls`` trace rows.
    """
    run = results / name
    inputs = run / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    records_name = Path(manifest["test_case_records_path"]).name
    (inputs / records_name).write_text("".join(c + "\n" for c in cases), encoding="utf-8")
    (run / f"{name}.log").write_text("2026-11-10 10:00:00,000 INFO corallab: started\n")
    for index, batch in enumerate(batches, start=1):
        folder = run / f"batch-{index:03d}"
        write_jsonl(folder / f"{name}-sim.jsonl", batch["transcripts"])
        write_jsonl(
            folder / f"{name}-sim-batch-{index:03d}.trace.jsonl", batch.get("sim_calls", [])
        )
        if "checks" in batch:
            write_jsonl(folder / f"{name}-check.jsonl", batch["checks"])
            write_jsonl(folder / f"{name}-check.trace.jsonl", batch.get("check_calls", []))
        if "judges" in batch:
            write_jsonl(folder / f"{name}-judge.jsonl", batch["judges"])
            write_jsonl(folder / f"{name}-judge.trace.jsonl", batch.get("judge_calls", []))
    return run
