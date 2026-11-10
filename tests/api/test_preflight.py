"""preflight against a temporary data repository and a fake Ollama server."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from _records import make_test_case

from corallab.api import Severity, preflight
from corallab.api.preflight import ASSUMED_OLLAMA_CONTEXT, suggested_num_ctx
from corallab.schemas.core.kl import KlPin
from tests.api._repo import DataRepo, manifest_data

Fetch = Callable[..., Any]


@pytest.fixture
def keys_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-do-not-print")
    monkeypatch.setenv("TEST_API_BASE", "https://secret-host.example")


def _checks(issues: list[Any]) -> list[str]:
    return [i.check for i in issues]


def test_a_clean_manifest_passes(data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch) -> None:
    report = preflight("smoke", fetch=ollama_fetch)
    assert report.ok, report.errors
    assert report.errors == []
    assert report.run_id == "api-test"
    assert report.test_case_count == 1
    assert report.catalog is not None
    assert report.catalog.matches
    assert report.catalog.kl_path is not None
    assert report.env is not None
    assert report.env.missing == []
    text = report.model_dump_json()
    assert "sk-do-not-print" not in text
    assert "secret-host" not in text


def test_a_missing_key_is_an_error_naming_the_var(data_repo: DataRepo, ollama_fetch: Fetch) -> None:
    report = preflight("smoke", fetch=ollama_fetch)
    assert not report.ok
    env = [i for i in report.errors if i.check == "env"]
    assert {i.subject for i in env} == {"TEST_API_KEY", "TEST_API_BASE"}
    assert all(i.severity == Severity.ERROR for i in env)
    assert "remote/model (system_under_test, judge)" in env[0].message


def test_a_missing_manifest(data_repo: DataRepo) -> None:
    report = preflight("nope")
    assert not report.ok
    assert _checks(report.errors) == ["manifest"]


def test_an_invalid_manifest_lists_each_field(data_repo: DataRepo) -> None:
    data_repo.write_manifest("bad", manifest_data(evaluation_level=7, extra=1))
    report = preflight("bad")
    messages = " ".join(i.message for i in report.errors)
    assert "evaluation_level" in messages
    assert "extra" in messages
    assert report.calls == []


def test_unparseable_yaml(data_repo: DataRepo) -> None:
    data_repo.write_manifest("bad", "run_id: [")
    assert _checks(preflight("bad").errors) == ["manifest"]


def test_an_empty_run_dir_is_an_error(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_manifest("bad", manifest_data(run_dir=""))
    report = preflight("bad", query_ollama=False)
    assert any(i.check == "manifest" and "run_dir" in i.message for i in report.errors)


def test_an_unregistered_model(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_manifest("m", manifest_data(system_under_test=[{"model_id": "ghost/model"}]))
    report = preflight("m", query_ollama=False)
    models = [i for i in report.errors if i.check == "models"]
    assert [i.subject for i in models] == ["ghost/model"]
    assert "UnknownModelError" in models[0].message


def test_a_pin_mismatch_names_the_test_case_and_the_unset_kl_path(
    data_repo: DataRepo, keys_set: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = make_test_case(pin=KlPin(version="0.0.1", digest="sha256:0000000000000000"))
    data_repo.write_cases([stale.model_dump_json()])
    report = preflight("smoke", query_ollama=False)
    pins = [i for i in report.errors if i.check == "catalog_pin"]
    assert [i.subject for i in pins] == ["aimhef-0000f001"]
    assert report.catalog is not None
    assert not report.catalog.matches
    assert report.catalog.test_case_pins == ["0.0.1 sha256:0000000000000000"]
    assert "CORALMIND_KL_PATH is unset" not in pins[0].message


def test_a_pin_mismatch_does_not_hide_a_missing_persona(
    data_repo: DataRepo, keys_set: None
) -> None:
    stale = make_test_case(
        persona_ref="p-ghost@1", pin=KlPin(version="0.0.1", digest="sha256:0000000000000000")
    )
    data_repo.write_cases([stale.model_dump_json()])
    report = preflight("smoke", query_ollama=False)
    assert set(_checks(report.errors)) == {"catalog_pin", "personas"}


def test_a_missing_persona_is_reported_once(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_cases([make_test_case(persona_ref="p-ghost@1").model_dump_json()])
    report = preflight("smoke", query_ollama=False)
    personas = [i for i in report.errors if i.check == "personas"]
    assert len(personas) == 1
    assert "p-ghost" in personas[0].message


def test_a_missing_persona_directory(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_manifest("m", manifest_data(persona_records_path="nowhere"))
    report = preflight("m", query_ollama=False)
    assert any("persona records directory" in i.message for i in report.errors)


def test_a_level_mismatch(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_cases([make_test_case(evaluation_level=1).model_dump_json()])
    report = preflight("smoke", query_ollama=False)
    assert _checks(report.errors) == ["test_cases"]
    assert "the manifest's level is 3" in report.errors[0].message


def test_a_pathway_mismatch(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_cases([make_test_case(pathway_id="kl-900101").model_dump_json()])
    report = preflight("smoke", query_ollama=False)
    pathways = [i for i in report.errors if i.check == "pathways"]
    assert [i.subject for i in pathways] == ["aimhef-0000f001"]


def test_missing_and_empty_records(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_manifest("m", manifest_data(test_case_records_path="test_case_records/x.jsonl"))
    assert "not found" in preflight("m", query_ollama=False).errors[0].message
    data_repo.write_cases(['{"nope": 1}'])
    report = preflight("smoke", query_ollama=False)
    assert any("no valid test case" in i.message for i in report.errors)


def test_a_catalog_that_does_not_load(
    data_repo: DataRepo, keys_set: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CORALMIND_KL_PATH", str(data_repo.root / "no-library-here"))
    report = preflight("smoke", query_ollama=False)
    assert "catalog" in _checks(report.errors)
    assert any("judge calls are not estimated" in i.message for i in report.warnings)
    assert report.catalog is not None
    assert report.catalog.version is None


def test_a_bad_fidelity_prompt_and_judge_options(data_repo: DataRepo, keys_set: None) -> None:
    data = manifest_data()
    data["simulation"]["elicitation_fidelity"] = {"prompt_version": "no-such-prompt"}
    data["judges"][0]["options"] = {"group_size": 0, "mystery": 1}
    data_repo.write_manifest("m", data)
    report = preflight("m", query_ollama=False)
    assert "fidelity_prompt" in _checks(report.errors)
    assert any(i.check == "judge_options" for i in report.errors)
    assert any("mystery" in i.message for i in report.warnings)


def test_an_unknown_judge_prompt_version(data_repo: DataRepo, keys_set: None) -> None:
    data = manifest_data()
    data["judges"][0]["options"] = {"prompt_version": "nope"}
    data_repo.write_manifest("m", data)
    report = preflight("m", query_ollama=False)
    assert any("not a prompt directory" in i.message for i in report.errors)


def test_call_estimates_at_l3(data_repo: DataRepo, keys_set: None) -> None:
    data_repo.write_cases(
        [
            make_test_case().model_dump_json(),
            make_test_case(test_case_id="aimhef-0000f002").model_dump_json(),
        ]
    )
    report = preflight("smoke", query_ollama=False)
    rows = {r.role: r for r in report.calls}
    assert rows["system_under_test"].calls == 2 * 4
    assert rows["system_under_test"].upper_bound
    assert rows["person_response"].calls == 2 * 3
    assert rows["elicitation_fidelity"].calls == 2
    assert rows["elicitation_fidelity"].model_id == "local/model"
    assert not rows["judge_stage1"].upper_bound
    assert rows["judge_stage1"].calls % 2 == 0
    assert rows["judge_stage1"].calls > 0
    assert rows["judge_stage2"].upper_bound
    total = sum(r.calls for r in report.calls)
    assert sum(report.calls_by_model.values()) == total


def test_call_estimates_multiply_by_sut_entries(data_repo: DataRepo, keys_set: None) -> None:
    single = {r.role: r.calls for r in preflight("smoke", query_ollama=False).calls}
    data_repo.write_manifest(
        "two",
        manifest_data(
            system_under_test=[
                {"model_id": "remote/model"},
                {"model_id": "remote/model", "label": "again"},
            ]
        ),
    )
    double = preflight("two", query_ollama=False).calls
    assert [r.role for r in double].count("system_under_test") == 2
    by_role = {r.role: r.calls for r in double}
    for role in ("person_response", "elicitation_fidelity", "judge_stage1", "judge_stage2"):
        assert by_role[role] == 2 * single[role]


def test_call_estimates_at_l1_and_l2(data_repo: DataRepo, keys_set: None) -> None:
    for level, expected in ((1, 1), (2, 3)):
        data_repo.write_cases([make_test_case(evaluation_level=level).model_dump_json()])
        data_repo.write_manifest(
            "m", manifest_data(evaluation_level=level, simulation={"batch_size": 1})
        )
        report = preflight("m", query_ollama=False)
        assert report.ok, report.errors
        rows = {r.role: r for r in report.calls}
        assert rows["system_under_test"].calls == expected
        assert not rows["system_under_test"].upper_bound
        assert "person_response" not in rows
        assert "elicitation_fidelity" not in rows


def test_disabled_fidelity_and_judges_make_no_calls(data_repo: DataRepo, keys_set: None) -> None:
    data = manifest_data()
    data["simulation"]["elicitation_fidelity"] = {"enabled": False}
    data["judges"][0]["enabled"] = False
    data_repo.write_manifest("m", data)
    roles = {r.role for r in preflight("m", query_ollama=False).calls}
    assert roles == {"system_under_test", "person_response"}


def test_ollama_prompts_over_the_assumed_context_warn(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch
) -> None:
    report = preflight("smoke", fetch=ollama_fetch)
    [local] = report.ollama
    assert local.model_id == "local/model"
    assert local.roles == ["person_response", "elicitation_fidelity"]
    assert local.reachable
    assert local.max_context_length == 131072
    roles = {p.role: p for p in local.prompts}
    assert set(roles) == {"person_response", "elicitation_fidelity"}
    for estimate in roles.values():
        assert estimate.context_length == ASSUMED_OLLAMA_CONTEXT
        assert estimate.context_source == "assumed"
        assert estimate.tokens > 0
        assert estimate.fits == (estimate.tokens <= ASSUMED_OLLAMA_CONTEXT)
    # Warnings never make the report fail.
    assert report.ok


def test_a_long_transcript_overflows_and_suggests_num_ctx(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch
) -> None:
    from corallab.api import TranscriptAssumption

    report = preflight(
        "smoke",
        fetch=ollama_fetch,
        transcript=TranscriptAssumption(user_tokens=200, assistant_tokens=1500),
    )
    assert report.ok
    overflow = [i for i in report.warnings if i.check == "ollama"]
    assert overflow
    assert "num_ctx" in overflow[0].message
    assert all(not p.fits for p in report.ollama[0].prompts)


def test_num_ctx_in_extra_params_is_used(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch
) -> None:
    data = manifest_data()
    data["simulation"]["person_response_model"] = {
        "model_id": "local/model",
        "extra_params": {"num_ctx": 65536},
    }
    data_repo.write_manifest("m", data)
    report = preflight("m", fetch=ollama_fetch)
    [estimate] = [p for p in report.ollama[0].prompts if p.role == "person_response"]
    assert estimate.context_length == 65536
    assert estimate.context_source == "extra_params"
    assert estimate.fits


def test_modelfile_num_ctx_is_used(data_repo: DataRepo, keys_set: None) -> None:
    def fetch(url: str, body: Any, timeout: float) -> dict[str, Any]:
        return {"parameters": "num_ctx 32768", "model_info": {}} if "show" in url else {}

    report = preflight("smoke", fetch=fetch)
    assert {p.context_source for p in report.ollama[0].prompts} == {"modelfile"}
    assert report.ollama[0].modelfile_num_ctx == 32768


def test_judge_on_ollama_sizes_both_stages(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch
) -> None:
    data = manifest_data()
    data["judges"][0]["model"] = {"model_id": "local/model"}
    data_repo.write_manifest("m", data)
    report = preflight("m", fetch=ollama_fetch)
    roles = {p.role for p in report.ollama[0].prompts}
    assert {"judge_stage1", "judge_stage2"} <= roles


def test_sut_on_ollama_with_injected_context(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch
) -> None:
    data = manifest_data(system_under_test=[{"model_id": "local/model"}])
    data["simulation"]["inject_context_into_sut"] = True
    data_repo.write_manifest("m", data)
    report = preflight("m", fetch=ollama_fetch)
    assert "system_under_test" in {p.role for p in report.ollama[0].prompts}


def test_an_unreachable_ollama_warns(
    data_repo: DataRepo, keys_set: None, ollama_down: Fetch
) -> None:
    report = preflight("smoke", fetch=ollama_down)
    assert report.ok
    assert any("did not answer" in i.message for i in report.warnings)
    assert report.ollama[0].reachable is False


def test_ollama_not_queried(data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch) -> None:
    report = preflight("smoke", query_ollama=False, fetch=ollama_fetch)
    assert ollama_fetch.calls == []  # type: ignore[attr-defined]
    assert report.ollama[0].reachable is None


def test_no_prompt_sizes_when_cases_fail(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch
) -> None:
    data_repo.write_cases([make_test_case(persona_ref="p-ghost@1").model_dump_json()])
    report = preflight("smoke", fetch=ollama_fetch)
    assert report.ollama[0].prompts == []
    assert any("prompt sizes are not estimated" in i.message for i in report.warnings)


def test_a_sizing_failure_is_a_warning(
    data_repo: DataRepo, keys_set: None, ollama_fetch: Fetch, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setattr("corallab.api._prompts.person_prompt", boom)
    report = preflight("smoke", fetch=ollama_fetch)
    assert any("tokenizer exploded" in i.message for i in report.warnings)


def test_suggested_num_ctx() -> None:
    assert suggested_num_ctx(1000, None) == 2048
    assert suggested_num_ctx(3000, None) == 4096
    assert suggested_num_ctx(16070, 131072) == 32768
    assert suggested_num_ctx(200000, 131072) == 131072


def test_a_manifest_by_path(data_repo: DataRepo, keys_set: None) -> None:
    path = data_repo.root / "run_manifests" / "smoke.yaml"
    assert preflight(str(path), query_ollama=False).manifest_path == str(path)
