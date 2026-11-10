"""Typed Python layer over the toolkit, for agents. See README.md in this directory.

Every function takes plain arguments and returns pydantic models. The CLI and
an MCP server are meant to be thin layers over these functions.
"""

from __future__ import annotations

from corallab.api._prompts import TranscriptAssumption
from corallab.api.env import check_env, load_manifest
from corallab.api.listing import list_manifests, list_models, list_personas, list_test_cases
from corallab.api.models import (
    CatalogInfo,
    EnvReport,
    EnvVarStatus,
    Issue,
    ManifestSummary,
    ModelEnvStatus,
    ModelSummary,
    OllamaContext,
    PersonaSummary,
    PreflightReport,
    PromptEstimate,
    RoleCalls,
    Severity,
    TestCaseSummary,
)
from corallab.api.preflight import preflight

__all__ = [
    "CatalogInfo",
    "EnvReport",
    "EnvVarStatus",
    "Issue",
    "ManifestSummary",
    "ModelEnvStatus",
    "ModelSummary",
    "OllamaContext",
    "PersonaSummary",
    "PreflightReport",
    "PromptEstimate",
    "RoleCalls",
    "Severity",
    "TestCaseSummary",
    "TranscriptAssumption",
    "check_env",
    "list_manifests",
    "list_models",
    "list_personas",
    "list_test_cases",
    "load_manifest",
    "preflight",
]
