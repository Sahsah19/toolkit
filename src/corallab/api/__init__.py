"""Typed Python layer over the toolkit, for agents. See README.md in this directory.

Every function takes plain arguments and returns pydantic models. The CLI and
an MCP server are meant to be thin layers over these functions.
"""

from __future__ import annotations

from corallab.api._prompts import TranscriptAssumption
from corallab.api.env import check_env, load_manifest
from corallab.api.listing import list_manifests, list_models, list_personas, list_test_cases
from corallab.api.manifests import create_manifest
from corallab.api.models import (
    BehaviorInfo,
    CatalogInfo,
    EnvReport,
    EnvVarStatus,
    Issue,
    ManifestOverrides,
    ManifestSummary,
    ManifestWriteResult,
    ModelAddResult,
    ModelEnvStatus,
    ModelSummary,
    OllamaContext,
    PersonaSummary,
    PersonaWriteResult,
    PreflightReport,
    PromptEstimate,
    RoleCalls,
    Severity,
    TestCaseSummary,
    TestCaseWriteResult,
    WriteAction,
    WriteResult,
)
from corallab.api.preflight import preflight
from corallab.api.records import create_persona, draft_test_case
from corallab.api.registry import add_model

__all__ = [
    "BehaviorInfo",
    "CatalogInfo",
    "EnvReport",
    "EnvVarStatus",
    "Issue",
    "ManifestOverrides",
    "ManifestSummary",
    "ManifestWriteResult",
    "ModelAddResult",
    "ModelEnvStatus",
    "ModelSummary",
    "OllamaContext",
    "PersonaSummary",
    "PersonaWriteResult",
    "PreflightReport",
    "PromptEstimate",
    "RoleCalls",
    "Severity",
    "TestCaseSummary",
    "TestCaseWriteResult",
    "TranscriptAssumption",
    "WriteAction",
    "WriteResult",
    "add_model",
    "check_env",
    "create_manifest",
    "create_persona",
    "draft_test_case",
    "list_manifests",
    "list_models",
    "list_personas",
    "list_test_cases",
    "load_manifest",
    "preflight",
]
