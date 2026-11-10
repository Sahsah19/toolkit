"""The writers rebuild committed upstream records: a manifest, test cases and personas.

Reads the committed files (never writes them) into a temporary data repository and
loads the library bundled with coralmap, which the upstream records pin. A test skips
when the installed coralmap bundles a different library.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from corallab import paths
from corallab.api.manifests import create_manifest
from corallab.api.models import ManifestOverrides
from corallab.api.records import create_persona, draft_test_case
from corallab.kl import load_library
from corallab.schemas.core.kl import KlPin
from corallab.schemas.eval.run_manifest import ModelCallConfig, RunManifest

_DATA = paths.default_repo_root() / "data-repository"
_CASES = "l3_launch_entrenchment_small.jsonl"


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty data repository beside the committed one, on the bundled library."""
    monkeypatch.delenv("CORALMIND_KL_PATH", raising=False)
    root = tmp_path / "data"
    for sub in ("run_manifests", "test_case_records", "persona_records", "endpoints"):
        (root / sub).mkdir(parents=True)
    shutil.copy(_DATA / "endpoints" / "endpoints.yaml", root / "endpoints" / "endpoints.yaml")
    monkeypatch.setenv("CORALMIND_DATA_REPOSITORY_PATH", str(root))
    return root


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _skip_unless_pinned(line: str) -> None:
    pin = KlPin.from_library(load_library())
    if json.loads(line)["kl"] != {"version": pin.version, "digest": pin.digest}:
        pytest.skip("the installed coralmap bundles a library the upstream records do not pin")


def test_rebuilds_l3_azure_from_l3_local(upstream: Path) -> None:
    shutil.copy(_DATA / "run_manifests" / "l3_local.yaml", upstream / "run_manifests")
    azure = ModelCallConfig(model_id="azure/gpt-5.4-mini", temperature=0.7)
    judge = ModelCallConfig(model_id="azure/gpt-5.4-mini", temperature=0.0)
    result = create_manifest(
        "l3_local",
        ManifestOverrides(
            run_id="launch-smoke-l3-azure",
            system_under_test=[azure],
            person_response_model=ModelCallConfig(model_id="azure/gpt-5.4-mini", temperature=0.9),
            judge_model=judge,
            run_dir="results/launch-smoke-l3-azure",
        ),
        "l3_azure",
        query_ollama=False,
        dotenv=False,
    )
    assert result.written, result.problems
    written = (upstream / "run_manifests" / "l3_azure.yaml").read_text(encoding="utf-8")
    assert written == (_DATA / "run_manifests" / "l3_azure.yaml").read_text(encoding="utf-8")


def test_rebuilds_the_launch_smoke_manifest(upstream: Path) -> None:
    shutil.copy(_DATA / "run_manifests" / "l3_local.yaml", upstream / "run_manifests")
    mini = "azure/gpt-5.4-mini"
    result = create_manifest(
        "l3_local",
        ManifestOverrides(
            run_id="l3-launch-entrenchment-smoke",
            system_under_test=[ModelCallConfig(model_id=mini, temperature=0.7)],
            test_case_records_path="test_case_records/l3_launch_entrenchment_smoke.jsonl",
            run_dir="results/l3-launch-entrenchment-smoke",
            batch_size=1,
            max_turns=10,
            person_response_model=ModelCallConfig(model_id=mini, temperature=0.9),
            fidelity_model=ModelCallConfig(model_id=mini, temperature=0.0),
            judge_model=ModelCallConfig(model_id=mini, temperature=0.0),
            set={"persona_records_path": None, "simulation.elicitation_fidelity.enabled": True},
        ),
        "smoke",
        query_ollama=False,
        dotenv=False,
    )
    assert result.written, result.problems
    committed = yaml.safe_load(
        (_DATA / "run_manifests" / "l3_launch_entrenchment_smoke.yaml").read_text()
    )
    written = yaml.safe_load((upstream / "run_manifests" / "smoke.yaml").read_text())
    assert written == committed
    assert result.manifest == RunManifest.model_validate(committed)
    # The records file is not in this repository, so preflight says so.
    assert result.preflight is not None
    assert any(i.check == "test_cases" for i in result.preflight.errors)


def test_rebuilds_upstream_test_cases(upstream: Path) -> None:
    lines = _lines(_DATA / "test_case_records" / _CASES)
    _skip_unless_pinned(lines[0])
    for line in lines:
        ref = json.loads(line)["persona_ref"].split("@")[0]
        shutil.copy(_DATA / "persona_records" / f"{ref}.json", upstream / "persona_records")
    for line in lines:
        o = json.loads(line)
        result = draft_test_case(
            _CASES,
            pathway_id=o["pathway_id"],
            persona_ref=o["persona_ref"],
            scenario=o["scenario"],
            opening=o["turns"][0],
            later_turns=o["turns"][1:],
            scenario_strategies=o["scenario_strategies"],
            assertions={a["behavior"]: a["assertion"] for a in o["behavior_assertions"]},
            assignments=[a["value"] for a in o["assignments"]],
            setup_behaviors=o["setup_behaviors"],
            evaluation_level=o["evaluation_level"],
            test_case_id=o["test_case_id"],
            version=o["version"],
            created_at=o["created_at"],
            display_name=o.get("display_name"),
            description=o.get("description"),
            language=o["language"],
        )
        assert result.written, (o["test_case_id"], result.problems)
    assert _lines(upstream / "test_case_records" / _CASES) == lines


def test_rebuilds_every_upstream_persona(upstream: Path) -> None:
    committed = sorted((_DATA / "persona_records").glob("p-*.json"))
    _skip_unless_pinned(_lines(_DATA / "test_case_records" / _CASES)[0])
    records = [json.loads(p.read_text(encoding="utf-8")) for p in committed]
    # Bases before their variants ('p-ocd-01', '-nt', '-nt-np'), so each base resolves.
    for record in sorted(records, key=lambda r: r["persona_id"].count("-")):
        variant = record["variant_of"]
        if variant is not None:
            base_id = variant["persona"].split("@")[0]
            assert (upstream / "persona_records" / f"{base_id}.json").is_file()
        result = create_persona(
            record["persona_id"],
            record["rendering"],
            record["elevated"],
            version=record["version"],
            provenance=record["provenance"],
            variant_of=variant["persona"] if variant else None,
            changed_paths=variant["changed_paths"] if variant else None,
        )
        assert result.written, (record["persona_id"], result.problems)
    for path in committed:
        rebuilt = (upstream / "persona_records" / path.name).read_text(encoding="utf-8")
        assert rebuilt == path.read_text(encoding="utf-8"), path.name
