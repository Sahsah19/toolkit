"""create_manifest: overrides, validation, the written layout, refusal to overwrite, preflight."""

from __future__ import annotations

import pytest
import yaml

from corallab.api.manifests import (
    SCHEMA_LINE,
    apply_overrides,
    create_manifest,
    dump_manifest,
)
from corallab.api.models import ManifestOverrides, WriteAction
from corallab.schemas.eval.run_manifest import ModelCallConfig
from tests.api._repo import DataRepo, manifest_data


def test_dump_manifest_matches_committed_layout() -> None:
    text = dump_manifest(
        {
            "run_id": "x",
            "system_under_test": [{"model_id": "m", "temperature": 0.7}],
            "simulation": {"person_response_model": {"model_id": "p"}},
            "judges": [{"judge_type": "kl_two_stage", "model": {"model_id": "j"}}],
        }
    )
    assert text == (
        SCHEMA_LINE + "run_id: x\n"
        "system_under_test:\n"
        "- model_id: m\n"
        "  temperature: 0.7\n"
        "simulation:\n"
        "    person_response_model:\n"
        "        model_id: p\n"
        "judges:\n"
        "- judge_type: kl_two_stage\n"
        "  model:\n"
        "      model_id: j\n"
    )


def test_apply_overrides_sets_roles_and_paths() -> None:
    data = apply_overrides(
        manifest_data(),
        ManifestOverrides(
            run_id="new",
            system_under_test=["a/one", ModelCallConfig(model_id="b/two", temperature=1.0)],
            person_response_model="p/m",
            fidelity_model="f/m",
            judge_model=ModelCallConfig(model_id="j/m", temperature=0.0),
            max_turns=6,
            batch_size=2,
            run_dir="results/new",
            test_case_records_path="test_case_records/other.jsonl",
        ),
    )
    assert data["run_id"] == "new"
    assert data["system_under_test"] == [
        {"model_id": "a/one"},
        {"model_id": "b/two", "temperature": 1.0},
    ]
    sim = data["simulation"]
    assert sim["person_response_model"] == {"model_id": "p/m"}
    assert sim["elicitation_fidelity"] == {"model": {"model_id": "f/m"}}
    assert (sim["max_turns"], sim["batch_size"]) == (6, 2)
    assert data["judges"][0]["model"] == {"model_id": "j/m", "temperature": 0.0}
    assert data["judges"][0]["options"] == {"group_size": 8}
    assert data["run_dir"] == "results/new"


def test_apply_overrides_leaves_the_template_alone() -> None:
    template = manifest_data()
    apply_overrides(template, ManifestOverrides(max_turns=9, set={"judges.0.options.x": 1}))
    assert template == manifest_data()


def test_set_paths_index_lists_and_null_removes() -> None:
    data = apply_overrides(
        manifest_data(),
        ManifestOverrides(
            set={"judges.0.options.max_concurrency": 1, "judges.0.options.group_size": None}
        ),
    )
    assert data["judges"][0]["options"] == {"max_concurrency": 1}


def test_bad_set_path_is_a_problem(data_repo: DataRepo) -> None:
    result = create_manifest(
        "smoke", ManifestOverrides(set={"judges.5.model": {}}), "out", query_ollama=False
    )
    assert not result.written
    assert "judges.5.model" in result.problems[0]


def test_create_writes_validates_and_preflights(data_repo: DataRepo) -> None:
    data_repo.write_manifest("smoke", dump_manifest(manifest_data()))
    result = create_manifest(
        "smoke",
        ManifestOverrides(run_id="copy", run_dir="results/copy", max_turns=6),
        "copy",
        query_ollama=False,
    )
    assert result.written, result.problems
    assert result.action == WriteAction.CREATED
    path = data_repo.root / "run_manifests" / "copy.yaml"
    text = path.read_text(encoding="utf-8")
    assert text.startswith(SCHEMA_LINE)
    assert yaml.safe_load(text)["simulation"]["max_turns"] == 6
    assert result.manifest is not None
    assert result.manifest.run_id == "copy"
    changed = [
        line
        for line in result.template_diff.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    assert changed == [
        "-run_id: api-test",
        "+run_id: copy",
        "-run_dir: results/api-test",
        "+run_dir: results/copy",
        "-    max_turns: 4",
        "+    max_turns: 6",
    ]
    assert result.diff.count("\n+") > 5  # a new file: every line added
    assert result.preflight is not None
    assert result.preflight.manifest_path == str(path)
    assert result.notes == []


def test_template_run_id_and_run_dir_are_noted(data_repo: DataRepo) -> None:
    result = create_manifest("smoke", None, "copy", query_ollama=False)
    assert result.written
    assert [n.split(" ")[0] for n in result.notes] == ["run_id", "run_dir"]


def test_defaults_the_template_omits_stay_omitted(data_repo: DataRepo) -> None:
    create_manifest("smoke", ManifestOverrides(run_id="c"), "c", query_ollama=False)
    written = yaml.safe_load((data_repo.root / "run_manifests" / "c.yaml").read_text())
    assert "persona_records_path" not in written
    assert "retry_max" not in written


def test_invalid_result_is_not_written(data_repo: DataRepo) -> None:
    result = create_manifest(
        "smoke", ManifestOverrides(set={"simulation.person_response_model": None}), "bad"
    )
    assert not result.written
    assert any("person_response_model" in p for p in result.problems)
    assert not (data_repo.root / "run_manifests" / "bad.yaml").exists()
    assert result.preflight is None


def test_refuses_to_overwrite_unless_asked(data_repo: DataRepo) -> None:
    first = create_manifest("smoke", ManifestOverrides(max_turns=5), "copy", query_ollama=False)
    assert first.written
    again = create_manifest("smoke", ManifestOverrides(max_turns=7), "copy", query_ollama=False)
    assert not again.written
    assert "overwrite=True" in again.problems[0]
    replaced = create_manifest(
        "smoke", ManifestOverrides(max_turns=7), "copy", overwrite=True, query_ollama=False
    )
    assert replaced.action == WriteAction.OVERWRITTEN
    assert "-    max_turns: 5" in replaced.diff
    assert "+    max_turns: 7" in replaced.diff


def test_overwriting_the_template_by_name_needs_overwrite(data_repo: DataRepo) -> None:
    result = create_manifest("smoke", ManifestOverrides(max_turns=7), "smoke")
    assert not result.written


@pytest.mark.parametrize("name", ["", "../escape", "a/b", ".hidden"])
def test_out_name_must_be_a_plain_name(data_repo: DataRepo, name: str) -> None:
    result = create_manifest("smoke", None, name)
    assert not result.written
    assert "out_name" in result.problems[0]


def test_out_name_may_carry_yaml_suffix(data_repo: DataRepo) -> None:
    result = create_manifest("smoke", ManifestOverrides(run_id="n"), "n.yaml", query_ollama=False)
    assert result.written
    assert (data_repo.root / "run_manifests" / "n.yaml").is_file()


def test_missing_template(data_repo: DataRepo) -> None:
    result = create_manifest("nope", None, "out")
    assert result.problems == ["template not found: nope"]


def test_preflight_errors_are_reported_not_raised(data_repo: DataRepo) -> None:
    result = create_manifest(
        "smoke", ManifestOverrides(judge_model="unknown/model"), "u", query_ollama=False
    )
    assert result.written
    assert result.preflight is not None
    assert not result.preflight.ok
    assert any(i.check == "models" for i in result.preflight.errors)
