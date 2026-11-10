"""Manifest helpers: path resolution and role walking."""

from __future__ import annotations

from pathlib import Path

import pytest

from corallab.api._manifests import (
    fidelity_model,
    manifest_roles,
    model_roles,
    read_run_manifest,
    relative_to_data,
    resolve_manifest_path,
)
from corallab.schemas.eval.run_manifest import RunManifest
from tests.api._repo import DataRepo, manifest_data


def test_resolve_by_id_relative_and_absolute(data_repo: DataRepo) -> None:
    target = data_repo.root / "run_manifests" / "smoke.yaml"
    assert resolve_manifest_path("smoke") == target
    assert resolve_manifest_path("run_manifests/smoke.yaml") == target
    assert resolve_manifest_path("data-repository/run_manifests/smoke.yaml") == target
    assert resolve_manifest_path(target) == target
    assert resolve_manifest_path("missing") == Path("missing")


def test_relative_to_data(data_repo: DataRepo, tmp_path: Path) -> None:
    assert relative_to_data(data_repo.root / "a" / "b.yaml") == "a/b.yaml"
    outside = tmp_path / "elsewhere.yaml"
    assert relative_to_data(outside) == str(outside)


def test_read_run_manifest_rejects_a_list(tmp_path: Path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text("- 1\n")
    with pytest.raises(ValueError, match="not a run manifest mapping"):
        read_run_manifest(path)


def test_roles_in_pipeline_order() -> None:
    data = manifest_data()
    data["simulation"]["elicitation_fidelity"] = {"model": {"model_id": "remote/model"}}
    manifest = RunManifest.model_validate(data)
    assert [r for r, _ in manifest_roles(manifest)] == [
        "system_under_test",
        "person_response",
        "elicitation_fidelity",
        "judge",
    ]
    assert model_roles(manifest) == {
        "remote/model": ["system_under_test", "elicitation_fidelity", "judge"],
        "local/model": ["person_response"],
    }


def test_fidelity_only_at_l3_when_enabled() -> None:
    l1 = RunManifest.model_validate(manifest_data(evaluation_level=1))
    assert fidelity_model(l1) is None
    data = manifest_data()
    data["simulation"]["elicitation_fidelity"] = {"enabled": False}
    assert fidelity_model(RunManifest.model_validate(data)) is None
    fallback = fidelity_model(RunManifest.model_validate(manifest_data()))
    assert fallback is not None
    assert fallback.model_id == "local/model"


def test_disabled_judges_have_no_role() -> None:
    data = manifest_data()
    data["judges"][0]["enabled"] = False
    roles = [r for r, _ in manifest_roles(RunManifest.model_validate(data))]
    assert "judge" not in roles
