"""add_model: insertion that keeps the file, refusals, and the env vars reported."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from corallab import paths
from corallab.api.models import WriteAction
from corallab.api.registry import add_model, insert_model
from corallab.llm.registry_loader import load_registry
from tests.api._repo import DataRepo

_COMMITTED = paths.default_repo_root() / "data-repository" / "endpoints" / "endpoints.yaml"


@pytest.fixture
def committed_copy(tmp_path: Path) -> Path:
    """A copy of the committed registry; the real file is only read."""
    copy = tmp_path / "endpoints.yaml"
    shutil.copy(_COMMITTED, copy)
    return copy


def test_keeps_every_line_of_the_committed_registry(committed_copy: Path) -> None:
    before = committed_copy.read_text(encoding="utf-8")
    result = add_model(
        "local/qwen3", "ollama-local", "ollama_chat/qwen3", registry=committed_copy, dotenv=False
    )
    assert result.written, result.problems
    assert result.action == WriteAction.UPDATED
    after = committed_copy.read_text(encoding="utf-8")
    lines = after.splitlines(keepends=True)
    at = lines.index("    local/qwen3:\n")
    assert lines[at : at + 3] == [
        "    local/qwen3:\n",
        "        endpoint: ollama-local\n",
        "        litellm_model: ollama_chat/qwen3\n",
    ]
    # Removing the three new lines gives back the file exactly.
    assert "".join(lines[:at] + lines[at + 3 :]) == before
    # The commented-out example at the end of models: stays last.
    assert after.rstrip().endswith("#   litellm_model: claude-sonnet-4-6")


def test_diff_and_summary(committed_copy: Path) -> None:
    result = add_model(
        "local/qwen3", "ollama-local", "ollama_chat/qwen3", registry=committed_copy, dotenv=False
    )
    assert "+    local/qwen3:" in result.diff
    changed = [
        line
        for line in result.diff.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 3
    assert all(line.startswith("+") for line in changed)
    assert result.model is not None
    assert (result.model.provider, result.model.local) == ("ollama", True)
    assert result.env is not None
    assert result.env.env_vars == []
    assert result.env.ready


def test_names_the_env_vars_to_set(committed_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_API_BASE", "https://secret.example")
    result = add_model(
        "azure/gpt-9",
        "azure-default",
        "azure/gpt-9",
        base_model="azure/gpt-9",
        registry=committed_copy,
        dotenv=False,
    )
    assert result.written
    assert result.env is not None
    assert {s.name: s.is_set for s in result.env.env_vars} == {
        "AZURE_API_KEY": False,
        "AZURE_API_BASE": True,
    }
    assert result.notes == ["set AZURE_API_KEY before using azure/gpt-9"]
    assert "secret.example" not in result.model_dump_json()
    route = load_registry(committed_copy, interpolate=False).models["azure/gpt-9"]
    assert route.base_model == "azure/gpt-9"


def test_refuses_a_duplicate(committed_copy: Path) -> None:
    before = committed_copy.read_text(encoding="utf-8")
    result = add_model(
        "local/llama3.2", "ollama-local", "ollama_chat/x", registry=committed_copy, dotenv=False
    )
    assert not result.written
    assert "already declared" in result.problems[0]
    assert committed_copy.read_text(encoding="utf-8") == before


def test_refuses_an_unknown_endpoint(committed_copy: Path) -> None:
    result = add_model("x/y", "nowhere", "x/y", registry=committed_copy, dotenv=False)
    assert not result.written
    assert "nowhere" in result.problems[0]
    assert "ollama-local" in result.problems[0]


def test_missing_registry(tmp_path: Path) -> None:
    result = add_model("x/y", "e", "x/y", registry=tmp_path / "none.yaml", dotenv=False)
    assert not result.written
    assert "not found" in result.problems[0]


def test_default_registry_is_the_data_repositorys(data_repo: DataRepo) -> None:
    result = add_model("local/other", "ollama-local", "ollama_chat/other", dotenv=False)
    assert result.written
    reg = load_registry(data_repo.root / "endpoints" / "endpoints.yaml", interpolate=False)
    assert "local/other" in reg.models


def test_insert_quotes_ids_that_need_it() -> None:
    text = "endpoints:\n  e:\n    provider: ollama\nmodels:\n  a:\n    endpoint: e\n"
    out = insert_model(text, "odd: id", {"endpoint": "e", "litellm_model": "x"})
    assert out == text + "  'odd: id':\n    endpoint: e\n    litellm_model: x\n"


def test_insert_into_empty_block_and_before_next_top_key() -> None:
    text = "models:\n# none yet\nendpoints: {}\n"
    out = insert_model(text, "a", {"endpoint": "e", "litellm_model": "x"})
    entry = "    a:\n        endpoint: e\n        litellm_model: x\n"
    assert out == "models:\n" + entry + "# none yet\nendpoints: {}\n"


def test_insert_without_trailing_newline() -> None:
    out = insert_model("models:\n    a:\n        endpoint: e", "b", {"endpoint": "e"})
    assert out == "models:\n    a:\n        endpoint: e\n    b:\n        endpoint: e\n"


def test_insert_needs_a_models_block() -> None:
    with pytest.raises(ValueError, match="models"):
        insert_model("endpoints: {}\n", "a", {"endpoint": "e"})


def test_a_bad_insert_restores_the_file(
    committed_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = committed_copy.read_text(encoding="utf-8")
    monkeypatch.setattr(
        "corallab.api.registry.insert_model", lambda text, *_: text + "models: {}\n"
    )
    result = add_model("x/y", "ollama-local", "x", registry=committed_copy, dotenv=False)
    assert not result.written
    assert "restored" in result.problems[0]
    assert committed_copy.read_text(encoding="utf-8") == before
