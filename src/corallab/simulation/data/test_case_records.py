"""Load a JSONL file of test case records and the persona records they name.

The file runs whole or not at all: every invalid line, unresolved persona and
library-pin mismatch is collected and raised together in one ``TestCaseRecordsError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from coralmap import KnowledgeLibrary
from coralmap.models import Pathway, observations
from pydantic import ValidationError

from corallab import paths
from corallab.schemas.core.kl import KlPin
from corallab.schemas.eval.persona_record import PersonaRecord
from corallab.schemas.eval.run_case import RunCase
from corallab.schemas.eval.test_case_record import TestCaseRecord

logger = logging.getLogger(__name__)


ProblemKind = Literal["invalid", "repeat", "pin", "level", "persona"]


@dataclass(frozen=True)
class RecordProblem:
    """One problem ``load_test_case_records`` found, with what kind it is.

    ``message`` is the same text as the matching entry of ``TestCaseRecordsError.problems``.
    """

    kind: ProblemKind
    message: str
    test_case_id: str | None = None
    line: int | None = None


class TestCaseRecordsError(ValueError):
    """A test case records file that cannot run, with every problem found."""

    __test__ = False

    def __init__(
        self, path: Path, problems: list[str], details: list[RecordProblem] | None = None
    ) -> None:
        self.path = path
        self.problems = problems
        self.details = details or [RecordProblem("invalid", p) for p in problems]
        listed = "\n  ".join(problems)
        super().__init__(f"{path}: {len(problems)} problem(s)\n  {listed}")


class NotAPathwayError(KeyError):
    """A ``pathway_id`` that names a library entry other than a pathway."""


def _errors(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())


def load_persona(personas_dir: Path, ref: str) -> PersonaRecord | str:
    """The persona ``ref`` names, or a problem describing why it cannot be read."""
    persona_id, _, _ = ref.partition("@")
    path = personas_dir / f"{persona_id}.json"
    if not path.is_file():
        return f"persona {ref}: {path} not found"
    try:
        persona = PersonaRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        return f"persona {ref}: {_errors(exc)}"
    if persona.ref != ref:
        return f"persona {ref}: {path.name} holds {persona.ref}"
    return persona


def load_test_case_records(
    records_path: str | Path,
    personas_dir: str | Path,
    library: KnowledgeLibrary,
    evaluation_level: int | None = None,
) -> list[RunCase]:
    """Every test case in ``records_path`` with its persona.

    Raises ``TestCaseRecordsError`` listing each problem.

    Paths resolve through :func:`corallab.paths.resolve_data_path`. A test case
    whose ``kl`` pin differs from ``library`` is a problem (TEST_CASE_RULES T5), and so is one
    whose ``evaluation_level`` differs from ``evaluation_level`` when that is given: the run
    follows the manifest's level, so a mismatched test case would silently run at the wrong one.
    """
    records_file = Path(paths.resolve_data_path(records_path))
    personas_root = Path(paths.resolve_data_path(personas_dir))
    if not records_file.is_file():
        raise FileNotFoundError(f"Test case records not found: {records_file}")

    pin = KlPin.from_library(library)
    details: list[RecordProblem] = []
    cases: list[RunCase] = []
    seen: set[str] = set()
    personas: dict[str, PersonaRecord] = {}
    for number, line in enumerate(records_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = TestCaseRecord.model_validate_json(line)
        except ValidationError as exc:
            details.append(RecordProblem("invalid", f"line {number}: {_errors(exc)}", None, number))
            continue
        tc_id = record.test_case_id
        if tc_id in seen:
            details.append(
                RecordProblem("repeat", f"line {number}: test case {tc_id} repeats", tc_id, number)
            )
            continue
        seen.add(tc_id)
        if (record.kl.version, record.kl.digest) != (pin.version, pin.digest):
            message = (
                f"{tc_id}: pins library {record.kl.version} {record.kl.digest}; "
                f"the loaded library is {pin.version} {pin.digest}"
            )
            details.append(RecordProblem("pin", message, tc_id, number))
            continue
        if evaluation_level is not None and record.evaluation_level != evaluation_level:
            message = (
                f"{tc_id}: is level {record.evaluation_level}; "
                f"the manifest's level is {evaluation_level}"
            )
            details.append(RecordProblem("level", message, tc_id, number))
            continue
        persona = personas.get(record.persona_ref)
        if persona is None:
            loaded = load_persona(personas_root, record.persona_ref)
            if isinstance(loaded, str):
                details.append(RecordProblem("persona", f"{tc_id}: {loaded}", tc_id, number))
                continue
            persona = personas[record.persona_ref] = loaded
        cases.append(RunCase(test_case=record, persona=persona))

    if details:
        raise TestCaseRecordsError(records_file, [d.message for d in details], details)
    logger.info("Loaded %d test case(s) from %s", len(cases), records_file)
    return cases


def dimension_about(library: KnowledgeLibrary, value_id: str) -> str:
    """What the dimension of context value ``value_id`` is about: person, occasion, exchange or
    setting (TEST_CASE_RULES section 2). Raises ``KeyError`` when either entry is missing."""
    value: Any = library.entry(value_id)
    dimension: Any = library.entry(value.body.dimension)
    return str(getattr(dimension.body.about, "value", dimension.body.about))


def expected_behaviors(library: KnowledgeLibrary, record: TestCaseRecord) -> set[str]:
    """The pathway's user-behavior parts plus the test case's setup behaviors (rule T6).

    Raises ``KeyError`` when ``pathway_id`` names no entry, ``NotAPathwayError`` when it names
    an entry that is not a pathway.
    """
    pathway = library.entry(record.pathway_id)
    if not isinstance(pathway, Pathway):
        raise NotAPathwayError(f"{record.pathway_id} is not a pathway")
    return {
        p.id
        for p in observations(pathway.body)
        if str(p.kind) == "user_behavior" and p.id != "OWED"
    } | set(record.setup_behaviors)


def check_test_case_records(cases: list[RunCase], library: KnowledgeLibrary) -> list[str]:
    """Problems with loaded test cases that only the library can show; empty when none."""
    problems: list[str] = []
    for case in cases:
        record = case.test_case
        try:
            expected = expected_behaviors(library, record)
        except NotAPathwayError:
            problems.append(f"{record.test_case_id}: {record.pathway_id} is not a pathway")
            continue
        except KeyError:
            problems.append(
                f"{record.test_case_id}: pathway {record.pathway_id} is not in the library"
            )
            continue
        asserted = {a.behavior for a in record.behavior_assertions}
        if asserted != expected:
            problems.append(
                f"{record.test_case_id}: asserts {sorted(asserted)}, "
                f"its pathway expects {sorted(expected)}"
            )
    return problems
