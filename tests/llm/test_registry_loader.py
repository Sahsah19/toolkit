"""Tests for llm/registry_loader.py.

Covers:
- A missing file raises. An empty file produces an empty registry.
- ${ENV_VAR} interpolation in api_base, including missing var -> None.
- api_key_env name -> os.environ lookup.
- Validation errors: unknown top-level keys, model points at unknown endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from corallab.llm.registry_loader import endpoint_env_vars, load_registry, resolve_api_key


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "endpoints.yaml"
    path.write_text(body)
    return path


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"does_not_exist\.yaml"):
        load_registry(tmp_path / "does_not_exist.yaml")


def test_empty_yaml_returns_empty_registry(tmp_path: Path) -> None:
    registry = load_registry(_write(tmp_path, ""))
    assert registry.endpoints == {}
    assert registry.models == {}


def test_env_var_interpolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MY_AZURE_BASE", "https://eastus.openai.azure.com")
    body = """
endpoints:
  azure-eastus:
    provider: azure
    api_base: ${MY_AZURE_BASE}
    api_key_env: AZURE_EASTUS_KEY
    api_version: "2024-12-01-preview"
"""
    registry = load_registry(_write(tmp_path, body))
    endpoint = registry.endpoints["azure-eastus"]
    assert endpoint.api_base == "https://eastus.openai.azure.com"
    assert endpoint.api_version == "2024-12-01-preview"
    assert endpoint.api_key_env == "AZURE_EASTUS_KEY"


def test_missing_env_var_resolves_to_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("UNSET_BASE_VAR", raising=False)
    body = """
endpoints:
  unset-endpoint:
    provider: azure
    api_base: ${UNSET_BASE_VAR}
    api_key_env: UNSET_KEY_VAR
"""
    registry = load_registry(_write(tmp_path, body))
    assert registry.endpoints["unset-endpoint"].api_base is None
    assert resolve_api_key(registry.endpoints["unset-endpoint"]) is None


def test_partial_interpolation_in_string_raises_when_var_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PARTIAL_HOST", raising=False)
    body = """
endpoints:
  partial:
    provider: openai
    api_base: https://${PARTIAL_HOST}/v1
"""
    with pytest.raises(KeyError, match="PARTIAL_HOST"):
        load_registry(_write(tmp_path, body))


def test_resolve_api_key_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ROUTING_TEST_KEY", "secret-123")
    body = """
endpoints:
  ep:
    provider: openai
    api_key_env: ROUTING_TEST_KEY
"""
    registry = load_registry(_write(tmp_path, body))
    assert resolve_api_key(registry.endpoints["ep"]) == "secret-123"


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    body = """
endpoints: {}
models: {}
typo_section: {}
"""
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_registry(_write(tmp_path, body))


def test_model_referencing_unknown_endpoint_rejected(tmp_path: Path) -> None:
    body = """
endpoints:
  ep-a:
    provider: openai
models:
  good:
    endpoint: ep-a
    litellm_model: gpt-4o
  bad:
    endpoint: ep-missing
    litellm_model: gpt-4o
"""
    with pytest.raises(ValueError, match="unknown endpoints"):
        load_registry(_write(tmp_path, body))


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    body = "- not\n- a\n- mapping\n"
    with pytest.raises(ValueError, match="mapping at the top level"):
        load_registry(_write(tmp_path, body))


def test_azure_api_base_strips_foundry_v1_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Foundry's portal copies .../openai/v1; LiteLLM azure/ cannot use that path.

    The Azure OpenAI client appends /openai/deployments/{model}/..., so a base
    that already ends in /openai/v1 becomes .../openai/v1/openai/deployments/...
    and Azure returns 404 Resource not found.
    """
    monkeypatch.setenv(
        "MY_AZURE_BASE",
        "https://eastus.openai.azure.com/openai/v1",
    )
    body = """
endpoints:
  azure-eastus:
    provider: azure
    api_base: ${MY_AZURE_BASE}
    api_key_env: AZURE_EASTUS_KEY
"""
    registry = load_registry(_write(tmp_path, body))
    assert registry.endpoints["azure-eastus"].api_base == "https://eastus.openai.azure.com"


def test_non_azure_api_base_keeps_openai_v1_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MY_OPENAI_BASE", "https://proxy.example/openai/v1")
    body = """
endpoints:
  proxy:
    provider: openai
    api_base: ${MY_OPENAI_BASE}
    api_key_env: PROXY_KEY
"""
    registry = load_registry(_write(tmp_path, body))
    assert registry.endpoints["proxy"].api_base == "https://proxy.example/openai/v1"


def test_committed_registry_includes_ollama_local() -> None:
    """The committed registry routes local/llama3.2 to ollama-local."""
    from corallab.llm.registry_loader import load_registry
    from corallab.paths import endpoints_registry_path

    registry = load_registry(endpoints_registry_path())
    endpoint = registry.endpoints["ollama-local"]
    assert endpoint.provider == "ollama"
    assert endpoint.api_base == "http://localhost:11434"
    assert endpoint.api_key_env is None

    route = registry.models["local/llama3.2"]
    assert route.endpoint == "ollama-local"
    assert route.litellm_model == "ollama_chat/llama3.2"


def test_uninterpolated_load_keeps_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOME_BASE", "https://resolved.example")
    path = _write(
        tmp_path,
        """
endpoints:
  e:
    provider: openai
    api_base: ${SOME_BASE}/v1
    api_key_env: SOME_KEY
models:
  m:
    endpoint: e
    litellm_model: openai/m
""",
    )
    raw = load_registry(path, interpolate=False)
    assert raw.endpoints["e"].api_base == "${SOME_BASE}/v1"
    assert endpoint_env_vars(raw.endpoints["e"]) == ["SOME_KEY", "SOME_BASE"]
    resolved = load_registry(path)
    assert resolved.endpoints["e"].api_base == "https://resolved.example/v1"
    assert endpoint_env_vars(resolved.endpoints["e"]) == ["SOME_KEY"]


def test_uninterpolated_load_does_not_fail_on_a_missing_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("UNSET_HOST", raising=False)
    path = _write(
        tmp_path,
        """
endpoints:
  e:
    provider: openai
    api_base: https://${UNSET_HOST}/api
models: {}
""",
    )
    with pytest.raises(KeyError):
        load_registry(path)
    assert load_registry(path, interpolate=False).endpoints["e"].api_base == (
        "https://${UNSET_HOST}/api"
    )


def test_endpoint_env_vars_without_any() -> None:
    from corallab.llm.registry_schema import EndpointConfig

    assert endpoint_env_vars(EndpointConfig(provider="ollama", api_base="http://x")) == []
