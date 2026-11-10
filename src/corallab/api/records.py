"""Write test case and persona records, checked against the catalog the way the loader checks them.

Expected behaviors, the library pin, persona resolution and the pathway checks come from
``corallab.simulation.data.test_case_records``; nothing here restates those rules.
"""

from __future__ import annotations

import datetime as dt
import json
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from coralmap.models import ContextValue
from pydantic import ValidationError

from corallab import paths
from corallab.api._catalog import load_catalog
from corallab.api._manifests import relative_to_data
from corallab.api._writing import (
    data_target,
    name_problem,
    ordered_like,
    read_text,
    unified_diff,
    validation_problems,
)
from corallab.api.listing import read_test_case_summaries
from corallab.api.models import (
    BehaviorInfo,
    PersonaWriteResult,
    TestCaseWriteResult,
    WriteAction,
)
from corallab.schemas.core.kl import KlPin
from corallab.schemas.eval.persona_record import Elevation, PersonaRecord, PersonaRendering
from corallab.schemas.eval.provenance import Provenance
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.test_case_record import ScenarioStrategy, TestCaseRecord
from corallab.simulation.data.test_case_records import (
    NotAPathwayError,
    check_test_case_records,
    dimension_about,
    expected_behaviors,
    load_persona,
)

# Key order of the committed test case records; optional keys follow when set.
TEST_CASE_KEY_ORDER = (
    "test_case_id",
    "evaluation_level",
    "version",
    "pathway_id",
    "persona_ref",
    "setup_behaviors",
    "assignments",
    "language",
    "kl",
    "created_at",
    "scenario",
    "turns",
    "scenario_strategies",
    "behavior_assertions",
    "display_name",
    "description",
    "provenance",
)


def context_value_problems(
    library: Any, ids: Sequence[str], *, person: bool, what: str
) -> list[str]:
    """Problems with context values recorded on a persona (``person``) or a test case.

    Each id must be a context value. A persona's must sit on person dimensions and a test
    case's on any other (TEST_CASE_RULES T2, P3); no two may share a ``cardinality: one``
    dimension.
    """
    problems: list[str] = []
    by_dimension: dict[str, list[str]] = {}
    for value_id in ids:
        try:
            entry = library.entry(value_id)
        except KeyError:
            problems.append(f"{what} {value_id} is not in the library")
            continue
        if not isinstance(entry, ContextValue):
            problems.append(f"{what} {value_id} is not a context value")
            continue
        about = dimension_about(library, value_id)
        if person and about != "person":
            problems.append(
                f"{what} {value_id} ({entry.label}) is on a {about} dimension, not person"
            )
        if not person and about == "person":
            problems.append(
                f"{what} {value_id} ({entry.label}) is a fact about the person; it belongs on the "
                "persona"
            )
        by_dimension.setdefault(entry.body.dimension, []).append(value_id)
    for dimension_id, values in by_dimension.items():
        dimension = library.entry(dimension_id)
        if len(values) > 1 and str(getattr(dimension.body.cardinality, "value", "")) == "one":
            problems.append(
                f"{what}s {', '.join(values)} share dimension {dimension_id} ({dimension.label}), "
                "which takes one value"
            )
    return problems


def _behavior_info(library: Any, behavior_id: str, source: str) -> BehaviorInfo:
    try:
        entry = library.entry(behavior_id)
    except KeyError:
        return BehaviorInfo(id=behavior_id, label="(not in the library)", source=source)
    return BehaviorInfo(
        id=behavior_id, label=entry.label, definition=entry.definition, source=source
    )


def records_file_path(records_file: str | Path) -> Path:
    """A bare name is ``test_case_records/<name>.jsonl`` (``.jsonl`` added when missing);
    a path resolves like a manifest path, under the data repository when relative, existing
    or not."""
    value = str(records_file)
    if "/" not in value:
        return paths.eval_test_cases_dir() / (
            value if value.endswith(".jsonl") else f"{value}.jsonl"
        )
    return data_target(value)


def _new_test_case_id(taken: set[str]) -> str:
    while True:
        candidate = f"aimhef-{secrets.token_hex(4)}"
        if candidate not in taken:
            return candidate


def draft_test_case(
    records_file: str | Path,
    *,
    pathway_id: str,
    persona_ref: str,
    scenario: str,
    opening: str,
    scenario_strategies: Sequence[ScenarioStrategy | Mapping[str, Any]],
    assertions: Mapping[str, str] | None = None,
    assignments: Sequence[str] = (),
    setup_behaviors: Sequence[str] = (),
    later_turns: Sequence[str] = (),
    evaluation_level: int = 3,
    test_case_id: str | None = None,
    version: str = "1.0.0",
    created_at: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
    language: str = "en",
    persona_records_path: str = "persona_records",
) -> TestCaseWriteResult:
    """Build a test case, check it, and append it to ``records_file``.

    The library pin is the loaded library's (``CORALMIND_KL_PATH``, else the one bundled with
    coralmap). The behaviors to assert are the pathway's user-behavior parts plus
    ``setup_behaviors`` (rule T6), as ``check_test_case_records`` computes them;
    ``assertions`` maps each to the text of what this person does. Call without
    ``assertions`` to get ``expected_behaviors`` with their catalog labels; nothing is written
    until each has one assertion and no others.

    ``records_file`` is a name (``test_case_records/<name>.jsonl``) or a path; the file is
    created when missing. ``opening`` is the first user turn; ``later_turns`` are scripted
    turns after it (L2). A ``test_case_id`` is generated when not given. Never raises for a
    problem in the inputs.
    """
    target = records_file_path(records_file)
    result = TestCaseWriteResult(written=False, path=relative_to_data(target))
    if target.suffix != ".jsonl":
        return result.model_copy(update={"problems": [f"{result.path} is not a .jsonl file"]})
    library, problem = load_catalog()
    if problem is not None:
        return result.model_copy(update={"problems": [problem]})

    problems: list[str] = []
    # expected_behaviors reads only these two fields.
    probe_fields: dict[str, Any] = {
        "pathway_id": pathway_id,
        "setup_behaviors": list(setup_behaviors),
    }
    probe = TestCaseRecord.model_construct(**probe_fields)
    try:
        expected = expected_behaviors(library, probe)
    except NotAPathwayError:
        return result.model_copy(update={"problems": [f"{pathway_id} is not a pathway"]})
    except KeyError:
        return result.model_copy(
            update={"problems": [f"pathway {pathway_id} is not in the library"]}
        )
    setup = set(setup_behaviors)
    pathway_parts = sorted(expected - setup)
    infos = [_behavior_info(library, b, "pathway") for b in pathway_parts] + [
        _behavior_info(library, b, "setup_behaviors") for b in setup_behaviors
    ]
    result.expected_behaviors = infos

    given = dict(assertions or {})
    problems += [
        f"no assertion for expected behavior {info.id} ({info.label})"
        for info in infos
        if info.id not in given
    ]
    problems += [
        f"assertion for {behavior}, which this test case does not expect"
        for behavior in given
        if behavior not in expected
    ]
    problems += context_value_problems(library, assignments, person=False, what="assignment")

    existing = read_test_case_summaries(target) if target.is_file() else []
    taken = {s.id for s in existing}
    tc_id = test_case_id or _new_test_case_id(taken)
    if tc_id in taken:
        problems.append(f"{result.path} already holds test case {tc_id}")

    pin = KlPin.from_library(library)
    data: dict[str, Any] = {
        "test_case_id": tc_id,
        "evaluation_level": evaluation_level,
        "version": version,
        "pathway_id": pathway_id,
        "persona_ref": persona_ref,
        "setup_behaviors": list(setup_behaviors),
        "assignments": [{"value": v} for v in assignments],
        "language": language,
        "kl": {"version": pin.version, "digest": pin.digest},
        "created_at": created_at or dt.date.today().isoformat(),
        "scenario": scenario,
        "turns": [opening, *later_turns],
        "scenario_strategies": [
            s.model_dump() if isinstance(s, ScenarioStrategy) else dict(s)
            for s in scenario_strategies
        ],
        # Assertions in the order given, so a caller controls how the record reads.
        "behavior_assertions": [
            {"behavior": b, "assertion": text} for b, text in given.items() if b in expected
        ],
    }
    optional = {"display_name": display_name, "description": description}
    data.update({k: v for k, v in optional.items() if v is not None})
    try:
        record = TestCaseRecord.model_validate(data)
    except ValidationError as exc:
        problems += validation_problems(exc)
        return result.model_copy(update={"problems": problems})

    personas_root = data_target(persona_records_path)
    persona = load_persona(personas_root, persona_ref)
    if isinstance(persona, str):
        problems.append(persona)
    elif not problems:
        problems += check_test_case_records([RunCase(test_case=record, persona=persona)], library)
    result.test_case = record
    if problems:
        return result.model_copy(update={"problems": problems})

    ordered = {k: data[k] for k in TEST_CASE_KEY_ORDER if k in data}
    line = json.dumps(ordered, ensure_ascii=False) + "\n"
    before = read_text(target)
    prefix = "\n" if before and not before.endswith("\n") else ""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(prefix + line)
    action = WriteAction.APPENDED if before else WriteAction.CREATED
    return result.model_copy(
        update={
            "written": True,
            "action": action,
            "diff": unified_diff(before, before + prefix + line, result.path),
        }
    )


def _leaf_paths(a: Any, b: Any, prefix: str) -> list[str]:
    """Dotted paths at which two dumped mappings differ; lists compare whole."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in list(a) + [k for k in b if k not in a]:
            if a.get(key) != b.get(key):
                out += _leaf_paths(a.get(key), b.get(key), f"{prefix}.{key}")
        return out
    return [prefix] if a != b else []


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json", by_alias=True, exclude_unset=True)


def create_persona(
    persona_id: str,
    rendering: PersonaRendering | Mapping[str, Any],
    elevated: Sequence[Elevation | Mapping[str, Any]],
    *,
    version: int = 1,
    provenance: Provenance | Mapping[str, Any] | None = None,
    variant_of: str | None = None,
    changed_paths: Sequence[str] | None = None,
    overwrite: bool = False,
    directory: str = "persona_records",
) -> PersonaWriteResult:
    """Check a persona record and write it to ``<directory>/<persona_id>.json``.

    Validated as ``PersonaRecord``; each elevated value must be a context value on a person
    dimension, with at most one per ``cardinality: one`` dimension. For a counterfactual,
    ``variant_of`` is the base's pinned reference (``p-ocd-07@1``), which must resolve; the
    rendering paths that differ from it, including fields the variant removes (TEST_CASE_RULES
    P7), are worked out and used as ``changed_paths`` unless given (a mismatch with the given
    ones is a note). A rendering given as a mapping keeps its key order in the file. Refuses
    to replace an existing file unless ``overwrite``.
    """
    root = data_target(directory)
    target = root / f"{persona_id}.json"
    result = PersonaWriteResult(written=False, path=relative_to_data(target))
    problem = name_problem(persona_id, "persona_id")
    if problem:
        return result.model_copy(update={"problems": [problem]})

    rendering_data = (
        _dump(rendering) if isinstance(rendering, PersonaRendering) else dict(rendering)
    )
    problems: list[str] = []
    notes: list[str] = []
    base_text = ""
    computed: list[str] = []
    variant: dict[str, Any] | None = None
    if variant_of is not None:
        base = load_persona(root, variant_of)
        if isinstance(base, str):
            problems.append(f"variant_of: {base}")
        else:
            base_text = read_text(root / f"{base.persona_id}.json")
            computed = _leaf_paths(_dump(base.rendering), rendering_data, "rendering")
            if not computed:
                problems.append(f"the rendering does not differ from {variant_of}")
            chosen = list(changed_paths) if changed_paths is not None else computed
            if changed_paths is not None and set(chosen) != set(computed):
                notes.append(
                    f"changed_paths {sorted(chosen)} differ from the rendering's differences "
                    f"{sorted(computed)}"
                )
            variant = {"persona": variant_of, "changed_paths": chosen}
    elif changed_paths is not None:
        problems.append("changed_paths is given without variant_of")

    data: dict[str, Any] = {
        "persona_id": persona_id,
        "version": version,
        "provenance": _dump(provenance) if isinstance(provenance, Provenance) else provenance,
        "variant_of": variant,
        "rendering": rendering_data,
        "elevated": [_dump(e) if isinstance(e, Elevation) else dict(e) for e in elevated],
    }
    try:
        record = PersonaRecord.model_validate(data)
    except ValidationError as exc:
        return result.model_copy(update={"problems": problems + validation_problems(exc)})
    result.persona = record
    result.changed_paths = computed

    library, lib_problem = load_catalog()
    if lib_problem is not None:
        problems.append(lib_problem)
    else:
        values = [e.value for e in record.elevated]
        problems += context_value_problems(library, values, person=True, what="elevated value")
    if target.exists() and not overwrite:
        problems.append(f"{result.path} exists; pass overwrite=True to replace it")
    if problems:
        return result.model_copy(update={"problems": problems, "notes": notes})

    text = json.dumps(ordered_like(_dump(record), data), indent=2, ensure_ascii=False) + "\n"
    before = read_text(target)
    action = WriteAction.OVERWRITTEN if target.exists() else WriteAction.CREATED
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    base_diff = ""
    if variant is not None:
        base_diff = unified_diff(
            base_text, text, result.path, before_name=f"{directory}/{variant_of}"
        )
    return result.model_copy(
        update={
            "written": True,
            "action": action,
            "diff": unified_diff(before, text, result.path),
            "base_diff": base_diff,
            "notes": notes,
        }
    )
