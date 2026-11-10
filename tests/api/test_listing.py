"""list_manifests, list_test_cases, list_personas, list_models."""

from __future__ import annotations

import pytest
from _records import make_persona, make_test_case

from corallab.api import list_manifests, list_models, list_personas, list_test_cases
from tests.api._repo import DataRepo, manifest_data


def test_list_manifests_summarises_a_valid_manifest(data_repo: DataRepo) -> None:
    [summary] = list_manifests()
    assert summary.id == "smoke"
    assert summary.path == "run_manifests/smoke.yaml"
    assert summary.valid
    assert summary.evaluation_level == 3
    assert summary.system_under_test == ["remote/model"]
    assert summary.person_response_model == "local/model"
    # An unset fidelity model falls back to the person model.
    assert summary.fidelity_model == "local/model"
    assert summary.judge_models == ["remote/model"]
    assert summary.max_turns == 4


def test_list_manifests_marks_an_invalid_manifest(data_repo: DataRepo) -> None:
    data_repo.write_manifest("broken", manifest_data(surprise_key=1))
    data_repo.write_manifest("not-yaml", "run_id: [unclosed")
    by_id = {s.id: s for s in list_manifests()}
    assert not by_id["broken"].valid
    assert "surprise_key" in (by_id["broken"].error or "")
    assert by_id["broken"].run_id == "api-test"
    assert not by_id["not-yaml"].valid
    assert "not valid YAML" in (by_id["not-yaml"].error or "")


def test_list_manifests_accepts_a_missing_run_id_like_the_cli(data_repo: DataRepo) -> None:
    data = manifest_data()
    del data["run_id"]
    data_repo.write_manifest("smoke", data)
    [summary] = list_manifests()
    assert summary.valid


def test_list_manifests_without_fidelity_at_level_one(data_repo: DataRepo) -> None:
    data_repo.write_manifest(
        "smoke", manifest_data(evaluation_level=1, simulation={"batch_size": 1})
    )
    [summary] = list_manifests()
    assert summary.fidelity_model is None
    assert summary.person_response_model is None


def test_list_test_cases_reads_every_file(data_repo: DataRepo) -> None:
    data_repo.write_cases(
        [make_test_case(test_case_id="aimhef-0000f002").model_dump_json(), '{"bad": 1}'],
        name="more.jsonl",
    )
    summaries = list_test_cases()
    assert [s.id for s in summaries] == [
        "aimhef-0000f001",
        "aimhef-0000f002",
        "test_case_records/more.jsonl:2",
    ]
    first = summaries[0]
    assert first.path == "test_case_records/cases.jsonl"
    assert first.pathway_id == "kl-900102"
    assert first.persona_ref == "p-fix-01@1"
    assert first.behaviors == ["kl-900501"]
    assert first.n_scripted_turns == 3
    assert first.n_strategies == 2
    assert first.kl_digest
    assert summaries[2].error


def test_list_test_cases_reads_one_named_file(data_repo: DataRepo) -> None:
    data_repo.write_cases(
        [make_test_case(test_case_id="aimhef-0000f009").model_dump_json()], "x.jsonl"
    )
    assert [s.id for s in list_test_cases("test_case_records/x.jsonl")] == ["aimhef-0000f009"]


def test_list_test_cases_skips_schema_directories(data_repo: DataRepo) -> None:
    schema = data_repo.root / "test_case_records" / "schema"
    schema.mkdir()
    (schema / "ignored.jsonl").write_text("{}\n", encoding="utf-8")
    assert [s.id for s in list_test_cases()] == ["aimhef-0000f001"]


def test_list_test_cases_raises_for_a_missing_file(data_repo: DataRepo) -> None:
    with pytest.raises(FileNotFoundError):
        list_test_cases("test_case_records/nope.jsonl")


def test_list_personas(data_repo: DataRepo) -> None:
    variant = make_persona(persona_id="p-fix-01-nt").model_dump(by_alias=True)
    variant["variant_of"] = {"persona": "p-fix-01@1", "changed_paths": ["rendering.traits"]}
    import json

    (data_repo.root / "persona_records" / "p-fix-01-nt.json").write_text(json.dumps(variant))
    (data_repo.root / "persona_records" / "p-broken.json").write_text("{}")
    by_id = {s.id: s for s in list_personas()}
    assert by_id["p-fix-01@1"].age == 41
    assert by_id["p-fix-01@1"].elevated == ["kl-900601"]
    assert by_id["p-fix-01-nt@1"].variant_of == "p-fix-01@1"
    assert by_id["p-broken"].error


def test_list_personas_raises_for_a_missing_directory(data_repo: DataRepo) -> None:
    with pytest.raises(FileNotFoundError):
        list_personas("nowhere")


def test_list_models_names_env_vars_and_never_resolves_them(
    data_repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_API_BASE", "https://secret-host.example")
    monkeypatch.setenv("TEST_API_KEY", "sk-do-not-print")
    models = {m.id: m for m in list_models()}
    remote = models["remote/model"]
    assert remote.env_vars == ["TEST_API_KEY", "TEST_API_BASE"]
    assert remote.api_base is None
    assert not remote.local
    local = models["local/model"]
    assert local.local
    assert local.env_vars == []
    assert local.api_base == "http://localhost:11434"
    dumped = "".join(m.model_dump_json() for m in models.values())
    assert "secret-host" not in dumped
    assert "sk-do-not-print" not in dumped
