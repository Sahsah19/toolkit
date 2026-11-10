"""Write a run manifest from an existing one plus overrides, then preflight it."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from corallab import paths
from corallab.api._manifests import read_run_manifest, relative_to_data, resolve_manifest_path
from corallab.api._writing import name_problem, read_text, unified_diff, validation_problems
from corallab.api.models import ManifestOverrides, ManifestWriteResult, ModelChoice, WriteAction
from corallab.api.preflight import preflight
from corallab.schemas.eval.run_manifest import ModelCallConfig, RunManifest

SCHEMA_LINE = "# yaml-language-server: $schema=./schema/run_manifest.schema.json\n"


class _ManifestDumper(yaml.SafeDumper):
    """The committed manifests' layout: 4-space mappings, a list item's mapping 2 in.

    ``- model_id: x`` then ``  temperature: 0.7`` at the dash's column plus two; nested
    mappings 4 further. Plain PyYAML at ``indent=4`` writes ``-   model_id``.
    """

    _seq_item = False

    def expect_block_sequence_item(self, first: bool = False) -> None:
        if not first and isinstance(self.event, yaml.SequenceEndEvent):
            self.indent = self.indents.pop()
            self.state = self.states.pop()
            return
        self.write_indent()
        self.write_indicator("-", True, False, True)  # need_whitespace, whitespace, indent
        self.states.append(self.expect_block_sequence_item)
        self._seq_item = True
        self.expect_node(sequence=True)

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        if self._seq_item and not flow:
            self._seq_item = False
            self.indents.append(self.indent)
            self.indent = (self.indent or 0) + 2
            return
        self._seq_item = False
        super().increase_indent(flow, indentless)


def dump_manifest(data: dict[str, Any]) -> str:
    """Manifest YAML in the committed files' layout, keys in ``data``'s order."""
    body = yaml.dump(
        data,
        Dumper=_ManifestDumper,
        sort_keys=False,
        indent=4,
        allow_unicode=True,
        default_flow_style=False,
    )
    return SCHEMA_LINE + str(body)


def _model(choice: ModelChoice) -> dict[str, Any]:
    cfg = ModelCallConfig(model_id=choice) if isinstance(choice, str) else choice
    return cfg.model_dump(mode="json", exclude_unset=True)


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``value`` at a dotted path with list indexes, creating mappings; None removes it."""
    keys = dotted.split(".")
    node: Any = data
    for key in keys[:-1]:
        if isinstance(node, list):
            node = node[int(key)]
        else:
            node = node.setdefault(key, {})
            if node is None:
                raise ValueError(f"{dotted}: {key} is null")
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    elif value is None:
        node.pop(last, None)
    else:
        node[last] = value


def apply_overrides(template: dict[str, Any], overrides: ManifestOverrides) -> dict[str, Any]:
    """A copy of a raw manifest mapping with ``overrides`` applied. Raises ``ValueError`` for a
    ``set`` path that does not fit the mapping."""
    data = copy.deepcopy(template)
    for key in (
        "run_id",
        "evaluation_level",
        "test_case_records_path",
        "persona_records_path",
        "run_dir",
    ):
        value = getattr(overrides, key)
        if value is not None:
            data[key] = value
    if overrides.system_under_test is not None:
        data["system_under_test"] = [_model(c) for c in overrides.system_under_test]
    models = {
        "simulation.person_response_model": overrides.person_response_model,
        "simulation.elicitation_fidelity.model": overrides.fidelity_model,
    }
    for dotted, choice in models.items():
        if choice is not None:
            _set_path(data, dotted, _model(choice))
    counts = {
        "simulation.max_turns": overrides.max_turns,
        "simulation.batch_size": overrides.batch_size,
    }
    for dotted, count in counts.items():
        if count is not None:
            _set_path(data, dotted, count)
    if overrides.judge_model is not None:
        for judge in data.get("judges") or []:
            judge["model"] = _model(overrides.judge_model)
    for dotted, value in overrides.set.items():
        try:
            _set_path(data, dotted, value)
        except (IndexError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"set {dotted!r}: {exc}") from exc
    return data


def dropped_settings(template: dict[str, Any], overrides: ManifestOverrides) -> list[str]:
    """One note per role whose model is given as a bare id while the template set more than
    ``model_id`` for it: those settings are dropped and the role default applies."""
    sim = template.get("simulation") or {}
    fidelity = sim.get("elicitation_fidelity") or {}
    sut = template.get("system_under_test") or []
    roles: list[tuple[str, ModelChoice | None, Any]] = [
        (f"system_under_test[{i}]", choice, sut[i] if i < len(sut) else None)
        for i, choice in enumerate(overrides.system_under_test or [])
    ]
    roles += [
        (
            "simulation.person_response_model",
            overrides.person_response_model,
            sim.get("person_response_model"),
        ),
        ("simulation.elicitation_fidelity.model", overrides.fidelity_model, fidelity.get("model")),
    ]
    roles += [
        (f"judges[{i}].model", overrides.judge_model, judge.get("model"))
        for i, judge in enumerate(template.get("judges") or [])
    ]
    notes = []
    for role, choice, old in roles:
        if not isinstance(choice, str) or not isinstance(old, dict):
            continue
        dropped = [key for key in old if key != "model_id"]
        if dropped:
            notes.append(
                f"{role}: given as the bare id {choice!r}, so the template's "
                f"{', '.join(dropped)} were dropped and the role default applies"
            )
    return notes


def create_manifest(
    from_template: str | Path,
    overrides: ManifestOverrides | None = None,
    out_name: str = "",
    *,
    overwrite: bool = False,
    query_ollama: bool = True,
    dotenv: bool = True,
) -> ManifestWriteResult:
    """Build a run manifest from ``from_template`` plus ``overrides`` and write it to
    ``run_manifests/<out_name>.yaml``, then preflight it.

    ``from_template`` is a path or a manifest id (``l3_local``). Keys the template leaves out
    stay out, so defaults are not written. A model given as a bare id drops the template's
    other settings for that role; ``notes`` names each one dropped. Refuses to replace an
    existing file unless ``overwrite``. Comments in the template other than the schema line
    are not kept. Never raises for a problem in the inputs; nothing is written when
    ``problems`` is not empty.
    ``query_ollama`` and ``dotenv`` are passed to ``preflight``.
    """
    overrides = overrides or ManifestOverrides()
    name = out_name.removesuffix(".yaml")
    target = paths.eval_run_configs_dir() / f"{name}.yaml"
    result = ManifestWriteResult(written=False, path=relative_to_data(target))
    problem = name_problem(name, "out_name")
    if problem:
        return result.model_copy(update={"problems": [problem]})

    template_path = resolve_manifest_path(from_template)
    if not template_path.is_file():
        return result.model_copy(update={"problems": [f"template not found: {from_template}"]})
    result.template_path = relative_to_data(template_path)
    try:
        template = read_run_manifest(template_path)
        data = apply_overrides(template, overrides)
    except ValueError as exc:
        return result.model_copy(update={"problems": [str(exc)]})

    checked = dict(data)
    checked.setdefault("run_id", "run-generated")  # the CLI fills a missing run_id
    try:
        manifest = RunManifest.model_validate(checked)
    except ValidationError as exc:
        return result.model_copy(update={"problems": validation_problems(exc)})

    text = dump_manifest(data)
    result.manifest = manifest
    result.template_diff = unified_diff(
        read_text(template_path),
        text,
        result.path,
        before_name=result.template_path,
    )
    if target.exists() and not overwrite:
        return result.model_copy(
            update={"problems": [f"{result.path} exists; pass overwrite=True to replace it"]}
        )
    notes = [
        f"{key} is the template's ({data[key]}); runs of both manifests share it"
        for key in ("run_id", "run_dir")
        if key in data and getattr(overrides, key) is None and key not in overrides.set
    ]
    notes += dropped_settings(template, overrides)
    before = read_text(target)
    action = WriteAction.OVERWRITTEN if target.exists() else WriteAction.CREATED
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return result.model_copy(
        update={
            "written": True,
            "action": action,
            "diff": unified_diff(before, text, result.path),
            "notes": notes,
            "preflight": preflight(target, query_ollama=query_ollama, dotenv=dotenv),
        }
    )
