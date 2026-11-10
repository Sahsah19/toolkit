"""The launch slice's persona and test case records, checked against the real library.

``data-repository/persona_records/`` and ``test_case_records/l3_launch_entrenchment.jsonl`` hold the
records the slice is simulated from. The models check each record on its own; this
module checks what only the set, or the library, can: every reference resolves, each
case's expected behaviors are its pathway's person-side parts, each person's facts are
values on person dimensions, and every context-value gate is decided from those facts.
"""

from __future__ import annotations

import json

import pytest
from coralmap import KnowledgeLibrary, load
from coralmap.models import observations

from corallab import paths
from corallab.judging.judges.kl_two_stage import gate_inputs
from corallab.judging.kl_index import LibraryIndex
from corallab.schemas.core.kl import KlPin
from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.test_case_record import TestCaseRecord
from corallab.simulation.data.test_case_records import dimension_about as _about

_REAL_KL = paths.default_repo_root().parent / "knowledge-library"
_PERSONAS = paths.data_dir() / "persona_records"
_SUITE = paths.eval_test_cases_dir() / "l3_launch_entrenchment.jsonl"


@pytest.fixture(scope="module")
def library() -> KnowledgeLibrary:
    if not (_REAL_KL / "entries").is_dir():
        pytest.skip(f"sibling Knowledge Library not found at {_REAL_KL}")
    return load(_REAL_KL)


@pytest.fixture(scope="module")
def personas() -> dict[str, PersonaRecord]:
    found = {}
    for path in sorted(_PERSONAS.glob("p-*.json")):
        record = PersonaRecord.model_validate_json(path.read_text(encoding="utf-8"))
        assert path.stem == record.persona_id, f"{path.name} holds {record.persona_id}"
        found[record.ref] = record
    return found


@pytest.fixture(scope="module")
def cases() -> list[TestCaseRecord]:
    lines = _SUITE.read_text(encoding="utf-8").splitlines()
    return [TestCaseRecord.model_validate_json(line) for line in lines if line.strip()]


def test_the_records_are_present(
    personas: dict[str, PersonaRecord], cases: list[TestCaseRecord]
) -> None:
    assert personas, "no persona records"
    assert cases, "no test case records"
    assert len({c.test_case_id for c in cases}) == len(cases), "a test case id repeats"


def test_every_reference_resolves(
    personas: dict[str, PersonaRecord], cases: list[TestCaseRecord]
) -> None:
    problems = [
        f"{c.test_case_id}: persona {c.persona_ref}" for c in cases if c.persona_ref not in personas
    ]
    problems += [
        f"{ref}: variant of {p.variant_of.persona}"
        for ref, p in personas.items()
        if p.variant_of is not None and p.variant_of.persona not in personas
    ]
    assert not problems, "\n".join(problems)


def test_cases_pin_the_library_they_are_read_from(
    library: KnowledgeLibrary, cases: list[TestCaseRecord]
) -> None:
    pin = KlPin.from_library(library)
    stale = sorted({(c.kl.version, c.kl.digest) for c in cases} - {(pin.version, pin.digest)})
    assert not stale, f"cases pin {stale}; the library is {(pin.version, pin.digest)}"


def test_each_case_asserts_exactly_its_expected_behaviors(
    library: KnowledgeLibrary, cases: list[TestCaseRecord]
) -> None:
    """Expected behaviors are the pathway's user-behavior observations plus setup_behaviors."""
    problems = []
    for case in cases:
        pathway = library.entry(case.pathway_id)
        expected = {
            p.id
            for p in observations(pathway.body)
            if str(p.kind) == "user_behavior" and p.id != "OWED"
        } | set(case.setup_behaviors)
        asserted = {a.behavior for a in case.behavior_assertions}
        if asserted != expected:
            problems.append(
                f"{case.test_case_id}: asserts {sorted(asserted)}, expected {sorted(expected)}"
            )
        not_user = [b for b in asserted if str(library.entry(b).type.value) != "user_behavior"]
        if not_user:
            problems.append(f"{case.test_case_id}: {not_user} are not user behaviors")
    assert not problems, "\n".join(problems[:20])


def test_persona_facts_are_person_values_and_case_facts_are_not(
    library: KnowledgeLibrary, personas: dict[str, PersonaRecord], cases: list[TestCaseRecord]
) -> None:
    problems = [
        f"{ref}: {e.value} is on a {_about(library, e.value)} dimension"
        for ref, p in personas.items()
        for e in p.elevated
        if _about(library, e.value) != "person"
    ]
    problems += [
        f"{c.test_case_id}: assigns {a.value}, a fact about the person"
        for c in cases
        for a in c.assignments
        if _about(library, a.value) == "person"
    ]
    assert not problems, "\n".join(problems[:20])


def test_context_value_gates_are_decided_from_the_case(
    library: KnowledgeLibrary, personas: dict[str, PersonaRecord], cases: list[TestCaseRecord]
) -> None:
    """Every context-value gate is recorded from the test case's facts, not left unknown."""
    kg = LibraryIndex(library)
    problems = []
    gated = 0
    for case in cases:
        facts = {e.value for e in personas[case.persona_ref].elevated} | {
            a.value for a in case.assignments
        }
        gates = [
            g for g in gate_inputs(kg, case.pathway_id, facts) if g.part.kind == "context_value"
        ]
        if not gates:
            continue
        gated += 1
        unknown = [g.key for g in gates if g.recorded not in {"holds", "does_not_hold"}]
        if unknown:
            problems.append(f"{case.test_case_id}: {unknown} not decided from the case")
    assert gated, "no test case tests a gated pathway"
    assert not problems, "\n".join(problems[:20])


def test_the_file_is_json_lines_with_one_record_per_line() -> None:
    for number, line in enumerate(_SUITE.read_text(encoding="utf-8").splitlines(), 1):
        assert isinstance(json.loads(line), dict), f"line {number} is not a JSON object"
