"""Add a model to the endpoint registry, keeping the file's comments and order."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from corallab import paths
from corallab.api._manifests import relative_to_data
from corallab.api._writing import unified_diff, validation_problems
from corallab.api.env import load_env_file, model_env_status
from corallab.api.listing import list_models
from corallab.api.models import ModelAddResult, WriteAction
from corallab.llm.registry_loader import load_registry
from corallab.llm.registry_schema import ModelRoute

_TOP_KEY = re.compile(r"^[^\s#]")
_MODELS_LINE = re.compile(r"^models:\s*(#.*)?$")


def _scalar(value: str) -> str:
    """``value`` as a YAML scalar, quoted only when it must be."""
    return str(yaml.safe_dump([value], default_flow_style=True, width=10_000)).strip()[1:-1]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def insert_model(text: str, model_id: str, route: dict[str, str]) -> str:
    """``text`` with ``model_id`` added as the last entry of its top-level ``models:`` mapping.

    Every other line is kept as written, comments included. The entry follows the last
    uncommented line of the block, so a trailing commented-out example stays last.
    Raises ``ValueError`` when the file has no block ``models:`` mapping.
    """
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if _MODELS_LINE.match(line.rstrip("\n"))), None)
    if start is None:
        raise ValueError("no block 'models:' mapping at the top level")
    end = next((i for i in range(start + 1, len(lines)) if _TOP_KEY.match(lines[i])), len(lines))
    content = [
        i
        for i in range(start + 1, end)
        if lines[i].strip() and not lines[i].lstrip().startswith("#")
    ]
    key_indent, step = 4, 4
    if content:
        key_indent = _indent(lines[content[0]])
        deeper = [_indent(lines[i]) for i in content if _indent(lines[i]) > key_indent]
        if deeper:
            step = min(deeper) - key_indent
    entry = f"{' ' * key_indent}{_scalar(model_id)}:\n" + "".join(
        f"{' ' * (key_indent + step)}{k}: {_scalar(v)}\n" for k, v in route.items()
    )
    at = content[-1] + 1 if content else start + 1
    if at > 0 and not lines[at - 1].endswith("\n"):
        lines[at - 1] += "\n"
    return "".join(lines[:at]) + entry + "".join(lines[at:])


def add_model(
    model_id: str,
    endpoint: str,
    litellm_model: str,
    *,
    base_model: str | None = None,
    registry: str | Path | None = None,
    dotenv: bool = True,
) -> ModelAddResult:
    """Add ``model_id`` to ``endpoints.yaml`` under ``models:``, on an existing ``endpoint``.

    Keeps every other line, comments and order included, and checks the result reloads with
    only that model added; otherwise the file is restored. Refuses an id already declared.
    ``env`` names the env vars the endpoint needs and whether each is set (values are
    never read out); ``dotenv`` loads ``.env`` first, as the CLI does.
    """
    path = Path(registry) if registry is not None else paths.endpoints_registry_path()
    result = ModelAddResult(written=False, path=relative_to_data(path))

    def refuse(*problems: str) -> ModelAddResult:
        return result.model_copy(update={"problems": list(problems)})

    try:
        before_reg = load_registry(path, interpolate=False)
    except (FileNotFoundError, ValueError) as exc:
        return refuse(str(exc))
    if model_id in before_reg.models:
        return refuse(f"{model_id!r} is already declared in {result.path}")
    if endpoint not in before_reg.endpoints:
        known = ", ".join(sorted(before_reg.endpoints))
        return refuse(f"endpoint {endpoint!r} is not declared; declared endpoints: {known}")
    route = {"endpoint": endpoint, "litellm_model": litellm_model}
    if base_model is not None:
        route["base_model"] = base_model
    try:
        ModelRoute.model_validate(route)
    except ValidationError as exc:
        return refuse(*validation_problems(exc))

    before = path.read_text(encoding="utf-8")
    try:
        after = insert_model(before, model_id, route)
    except ValueError as exc:
        return refuse(f"{result.path}: {exc}")
    path.write_text(after, encoding="utf-8")
    try:
        after_reg = load_registry(path, interpolate=False)
        expected = {**before_reg.models, model_id: ModelRoute.model_validate(route)}
        if after_reg.models != expected or after_reg.endpoints != before_reg.endpoints:
            raise ValueError("the file reloads with more changed than the new model")
    except Exception as exc:  # restore whatever went wrong, then report it
        path.write_text(before, encoding="utf-8")
        return refuse(f"{result.path} was restored: {exc}")

    if dotenv:
        load_env_file()
    model = next(m for m in list_models(path) if m.id == model_id)
    env = model_env_status(after_reg, model_id, [])
    notes = [f"set {s.name} before using {model_id}" for s in env.env_vars if not s.is_set]
    return result.model_copy(
        update={
            "written": True,
            "action": WriteAction.UPDATED,
            "diff": unified_diff(before, after, result.path),
            "model": model,
            "env": env,
            "notes": notes,
        }
    )
