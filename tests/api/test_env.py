"""check_env and load_manifest."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from corallab.api import check_env, load_manifest
from corallab.api.env import is_set
from corallab.schemas.eval.run_manifest import RunManifest
from tests.api._repo import DataRepo, manifest_data


def test_without_a_manifest_every_model_is_reported(data_repo: DataRepo) -> None:
    report = check_env()
    assert {m.model_id for m in report.models} == {"remote/model", "local/model"}
    assert report.missing == ["TEST_API_BASE", "TEST_API_KEY"]
    assert all(m.roles == [] for m in report.models)
    assert not report.dotenv_loaded


def test_with_a_manifest_only_its_models_with_roles(
    data_repo: DataRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_API_BASE", "https://example.invalid")
    report = check_env("smoke")
    by_id = {m.model_id: m for m in report.models}
    assert by_id["remote/model"].roles == ["system_under_test", "judge"]
    assert by_id["local/model"].roles == ["person_response", "elicitation_fidelity"]
    assert by_id["local/model"].ready
    assert not by_id["remote/model"].ready
    assert report.missing == ["TEST_API_KEY"]


def test_values_never_appear(data_repo: DataRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-do-not-print")
    monkeypatch.setenv("TEST_API_BASE", "https://secret-host.example")
    report = check_env("smoke")
    assert report.missing == []
    text = report.model_dump_json()
    assert "sk-do-not-print" not in text
    assert "secret-host" not in text


def test_an_empty_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_EMPTY_VAR", "")
    assert not is_set("SOME_EMPTY_VAR")


def test_an_unregistered_model_is_not_ready(data_repo: DataRepo) -> None:
    data = manifest_data(system_under_test=[{"model_id": "nobody/model"}])
    report = check_env(RunManifest.model_validate(data))
    by_id = {m.model_id: m for m in report.models}
    assert not by_id["nobody/model"].ready
    assert "not in the endpoint registry" in (by_id["nobody/model"].error or "")


def test_dotenv_is_loaded_when_asked(data_repo: DataRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("corallab.api.env.load_env_file", lambda: True)
    assert check_env(dotenv=True).dotenv_loaded
    assert not check_env(dotenv=False).dotenv_loaded


def test_load_manifest_by_id_path_or_model(data_repo: DataRepo) -> None:
    by_id = load_manifest("smoke")
    by_path = load_manifest(data_repo.root / "run_manifests" / "smoke.yaml")
    assert by_id == by_path
    assert load_manifest(by_id) is by_id


def test_load_manifest_errors(data_repo: DataRepo) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest("missing")
    data_repo.write_manifest("bad", manifest_data(evaluation_level=9))
    with pytest.raises(ValidationError):
        load_manifest("bad")
