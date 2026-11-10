"""Loading a file of test case records with the persona records they name."""

from __future__ import annotations

from pathlib import Path

import pytest
from _records import make_persona, make_test_case, observed_fixture_copy, write_test_case_records
from pydantic import ValidationError

from corallab.kl import load_library
from corallab.schemas.core.kl import KlPin
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.test_case_record import TestCaseRecord
from corallab.simulation.data.test_case_records import (
    TestCaseRecordsError,
    check_test_case_records,
    expected_behaviors,
    load_test_case_records,
)


@pytest.fixture(scope="module")
def library():
    return load_library()


@pytest.fixture(scope="module")
def observed_library(tmp_path_factory: pytest.TempPathFactory):
    """The fixture library with kl-900501 composing kl-900102 instead of gating it."""
    return load_library(observed_fixture_copy(tmp_path_factory.mktemp("kl") / "kl"))


def test_loads_each_test_case_with_its_persona(tmp_path: Path, library) -> None:
    records_path, personas = write_test_case_records(tmp_path, [make_test_case()], [make_persona()])
    [case] = load_test_case_records(records_path, personas, library)
    assert case.test_case_id == "aimhef-0000f001"
    assert case.persona.ref == "p-fix-01@1"


def test_a_test_case_of_another_level_than_the_manifest_fails(tmp_path: Path, library) -> None:
    records_path, personas = write_test_case_records(tmp_path, [make_test_case()], [make_persona()])
    assert len(load_test_case_records(records_path, personas, library, evaluation_level=3)) == 1
    with pytest.raises(
        TestCaseRecordsError, match=r"aimhef-0000f001: is level 3; the manifest's level is 1"
    ):
        load_test_case_records(records_path, personas, library, evaluation_level=1)


def test_personas_are_read_once_per_reference(tmp_path: Path, library) -> None:
    cases = [make_test_case("aimhef-0000f001"), make_test_case("aimhef-0000f002")]
    records_path, personas = write_test_case_records(tmp_path, cases, [make_persona()])
    loaded = load_test_case_records(records_path, personas, library)
    assert loaded[0].persona is loaded[1].persona


def test_facts_are_elevated_values_plus_assignments() -> None:
    record = TestCaseRecord.model_validate(
        {**make_test_case().model_dump(), "assignments": [{"value": "kl-900604"}]}
    )
    case = RunCase(test_case=record, persona=make_persona(elevated=("kl-900601", "kl-900602")))
    assert case.facts == frozenset({"kl-900601", "kl-900602", "kl-900604"})


def test_run_case_rejects_the_wrong_persona() -> None:
    with pytest.raises(ValidationError, match="names p-fix-01@1"):
        RunCase(test_case=make_test_case(), persona=make_persona(persona_id="p-other"))


def test_missing_persona_fails_the_load(tmp_path: Path, library) -> None:
    records_path, personas = write_test_case_records(tmp_path, [make_test_case()], [])
    with pytest.raises(TestCaseRecordsError) as err:
        load_test_case_records(records_path, personas, library)
    assert "p-fix-01@1" in err.value.problems[0]
    assert "not found" in err.value.problems[0]


def test_persona_version_mismatch_fails_the_load(tmp_path: Path, library) -> None:
    records_path, personas = write_test_case_records(
        tmp_path, [make_test_case(persona_ref="p-fix-01@2")], [make_persona()]
    )
    with pytest.raises(TestCaseRecordsError, match="holds p-fix-01@1"):
        load_test_case_records(records_path, personas, library)


def test_pin_mismatch_fails_the_load(tmp_path: Path, library) -> None:
    stale = KlPin(version="0.0.1", digest="sha256:0000")
    records_path, personas = write_test_case_records(
        tmp_path, [make_test_case(pin=stale)], [make_persona()]
    )
    with pytest.raises(TestCaseRecordsError) as err:
        load_test_case_records(records_path, personas, library)
    assert "0.0.1" in err.value.problems[0]
    assert library.digest in err.value.problems[0]


def test_every_problem_is_listed(tmp_path: Path, library) -> None:
    stale = KlPin(version="0.0.1", digest="sha256:0000")
    cases = [make_test_case("aimhef-0000f001", pin=stale), make_test_case("aimhef-0000f002")]
    records_path, personas = write_test_case_records(tmp_path, cases, [])
    records_path.write_text(
        records_path.read_text() + '{"test_case_id": "bad"}\n', encoding="utf-8"
    )
    with pytest.raises(TestCaseRecordsError) as err:
        load_test_case_records(records_path, personas, library)
    assert len(err.value.problems) == 3


def test_repeated_test_case_id_fails_the_load(tmp_path: Path, library) -> None:
    records_path, personas = write_test_case_records(
        tmp_path, [make_test_case()] * 2, [make_persona()]
    )
    with pytest.raises(TestCaseRecordsError, match="repeats"):
        load_test_case_records(records_path, personas, library)


def test_missing_test_case_records_raise(tmp_path: Path, library) -> None:
    with pytest.raises(FileNotFoundError):
        load_test_case_records(tmp_path / "nope.jsonl", tmp_path, library)


def test_expected_behaviors_are_pathway_user_parts_plus_setup(observed_library) -> None:
    assert expected_behaviors(observed_library, make_test_case()) == {"kl-900501"}


def test_check_test_case_records_flags_an_assertion_the_pathway_does_not_state(
    observed_library,
) -> None:
    good = RunCase(test_case=make_test_case(), persona=make_persona())
    bad = RunCase(
        test_case=make_test_case("aimhef-0000f002", pathway_id="kl-900101"),
        persona=make_persona(),
    )
    problems = check_test_case_records([good, bad], observed_library)
    assert len(problems) == 1
    assert "aimhef-0000f002" in problems[0]


def test_check_test_case_records_flags_an_unknown_pathway(library) -> None:
    case = RunCase(test_case=make_test_case(pathway_id="kl-999999"), persona=make_persona())
    [problem] = check_test_case_records([case], library)
    assert "kl-999999" in problem
    assert "not in the library" in problem


def test_check_test_case_records_flags_a_pathway_id_that_names_a_behavior(library) -> None:
    case = RunCase(test_case=make_test_case(pathway_id="kl-900001"), persona=make_persona())
    [problem] = check_test_case_records([case], library)
    assert "kl-900001" in problem
    assert "not a pathway" in problem
    with pytest.raises(KeyError, match="not a pathway"):
        expected_behaviors(library, case.test_case)


def test_each_problem_carries_its_kind(tmp_path: Path, library) -> None:
    stale = KlPin(version="0.0.1", digest="sha256:0000")
    cases = [
        make_test_case("aimhef-0000f001", pin=stale),
        make_test_case("aimhef-0000f002"),
        make_test_case("aimhef-0000f003", evaluation_level=1),
        make_test_case("aimhef-0000f003"),
    ]
    records_path, personas = write_test_case_records(tmp_path, cases, [])
    records_path.write_text(
        records_path.read_text() + '{"test_case_id": "bad"}\n', encoding="utf-8"
    )
    with pytest.raises(TestCaseRecordsError) as err:
        load_test_case_records(records_path, personas, library, evaluation_level=3)
    details = err.value.details
    assert [(d.kind, d.test_case_id, d.line) for d in details] == [
        ("pin", "aimhef-0000f001", 1),
        ("persona", "aimhef-0000f002", 2),
        ("level", "aimhef-0000f003", 3),
        ("repeat", "aimhef-0000f003", 4),
        ("invalid", None, 5),
    ]
    assert [d.message for d in details] == err.value.problems


def test_an_error_built_from_strings_gets_invalid_details() -> None:
    err = TestCaseRecordsError(Path("x.jsonl"), ["something"])
    assert [(d.kind, d.message) for d in err.details] == [("invalid", "something")]


def test_load_persona_reports_a_missing_file(tmp_path: Path) -> None:
    from corallab.simulation.data.test_case_records import load_persona

    assert "not found" in str(load_persona(tmp_path, "p-ghost@1"))
