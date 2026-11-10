"""A throwaway data repository for the API tests: registry, persona, test cases, manifests.

Records are pinned to a copy of the fixture library whose pathway kl-900102 observes
kl-900501, so ``check_test_case_records`` passes on them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from _records import make_persona, make_test_case, observed_fixture_copy

from tests.api._repo import REGISTRY, DataRepo, manifest_data


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never load the developer's real .env in these tests."""
    monkeypatch.setattr("corallab.api.env.load_env_file", lambda: False)


@pytest.fixture
def data_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataRepo:
    kl = observed_fixture_copy(tmp_path / "kl")
    monkeypatch.setenv("CORALMIND_KL_PATH", str(kl))
    root = tmp_path / "data"
    for sub in ("endpoints", "run_manifests", "test_case_records", "persona_records"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("CORALMIND_DATA_REPOSITORY_PATH", str(root))
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.delenv("TEST_API_BASE", raising=False)
    (root / "endpoints" / "endpoints.yaml").write_text(yaml.safe_dump(REGISTRY), encoding="utf-8")
    persona = make_persona()
    (root / "persona_records" / "p-fix-01.json").write_text(
        persona.model_dump_json(by_alias=True), encoding="utf-8"
    )
    repo = DataRepo(root)
    repo.write_cases([make_test_case().model_dump_json()])
    repo.write_manifest("smoke", manifest_data())
    return repo


@pytest.fixture
def ollama_fetch() -> Callable[..., Any]:
    """A fake Ollama server: llama3.2 with no num_ctx, a 131072 trained maximum, not loaded."""
    calls: list[str] = []

    def fetch(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any] | None:
        calls.append(url)
        if url.endswith("/api/show"):
            return {
                "parameters": 'stop "<|eot_id|>"',
                "model_info": {"general.architecture": "llama", "llama.context_length": 131072},
            }
        if url.endswith("/api/ps"):
            return {"models": []}
        return None

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


@pytest.fixture
def ollama_down() -> Callable[..., Any]:
    def fetch(url: str, body: dict[str, Any] | None, timeout: float) -> None:
        return None

    return fetch
