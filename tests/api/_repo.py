"""Shared builders for the API tests' temporary data repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REGISTRY = {
    "endpoints": {
        "remote": {
            "provider": "azure",
            "api_base": "${TEST_API_BASE}",
            "api_key_env": "TEST_API_KEY",  # pragma: allowlist secret
            "api_version": "2024-12-01-preview",
        },
        "ollama-local": {
            "provider": "ollama",
            "api_base": "http://localhost:11434",
            "api_key_env": None,
        },
    },
    "models": {
        "remote/model": {"endpoint": "remote", "litellm_model": "azure/model"},
        "local/model": {"endpoint": "ollama-local", "litellm_model": "ollama_chat/llama3.2"},
    },
}


def manifest_data(**overrides: Any) -> dict[str, Any]:
    """An L3 manifest on the temporary repository; keyword arguments replace top-level keys."""
    data: dict[str, Any] = {
        "run_id": "api-test",
        "evaluation_level": 3,
        "system_under_test": [{"model_id": "remote/model", "temperature": 0.7}],
        "test_case_records_path": "test_case_records/cases.jsonl",
        "run_dir": "results/api-test",
        "simulation": {
            "batch_size": 1,
            "max_turns": 4,
            "person_response_model": {"model_id": "local/model"},
        },
        "judges": [
            {
                "judge_type": "kl_two_stage",
                "model": {"model_id": "remote/model"},
                "options": {"group_size": 8},
            }
        ],
    }
    data.update(overrides)
    return data


class DataRepo:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_manifest(self, name: str, data: dict[str, Any] | str) -> Path:
        path = self.root / "run_manifests" / f"{name}.yaml"
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        return path

    def write_cases(self, lines: list[str], name: str = "cases.jsonl") -> Path:
        path = self.root / "test_case_records" / name
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        return path
