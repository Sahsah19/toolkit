# Stage 2: writers for manifests, models, test cases and personas

Branch `agent-interface`, code commit `9ee9d32`, on top of Stage 1. Everything was checked against the upstream records (`main`), not the student homework-help work.

## What was built

Four writers in `src/corallab/api/`, exported from `corallab.api`:

| Module | Holds |
| --- | --- |
| `manifests.py` | `create_manifest`, `apply_overrides`, `dropped_settings`, `dump_manifest` |
| `registry.py` | `add_model`, `insert_model` |
| `records.py` | `draft_test_case`, `create_persona`, `context_value_problems`, `records_file_path`, `TEST_CASE_KEY_ORDER` |
| `_writing.py` | `unified_diff`, `data_target`, `name_problem`, `validation_problems`, `ordered_like` |
| `_catalog.py` | `load_catalog() -> (library, problem)`, moved out of `preflight.py` so the writers share it |
| `models.py` | `WriteResult`, `WriteAction`, `ManifestOverrides`, `ManifestWriteResult`, `ModelAddResult`, `BehaviorInfo`, `TestCaseWriteResult`, `PersonaWriteResult` |

Changes outside `api/`:

- `simulation/data/test_case_records.py`: new `dimension_about(library, value_id) -> str` (what a context value's dimension is `about`). It was `_about` in `tests/simulation/data/test_launch_records.py`; that test now imports it. Nothing else changed.
- `preflight.py`: `_load_catalog` wraps `load_catalog`. Same messages.

No result format, schema or viewer input changed. No new dependency.

## Signatures

```python
create_manifest(from_template: str | Path, overrides: ManifestOverrides | None = None,
                out_name: str = "", *, overwrite: bool = False,
                query_ollama: bool = True, dotenv: bool = True) -> ManifestWriteResult
add_model(model_id: str, endpoint: str, litellm_model: str, *, base_model: str | None = None,
          registry: str | Path | None = None, dotenv: bool = True) -> ModelAddResult
draft_test_case(records_file: str | Path, *, pathway_id: str, persona_ref: str, scenario: str,
                opening: str, scenario_strategies: Sequence[ScenarioStrategy | Mapping],
                assertions: Mapping[str, str] | None = None, assignments: Sequence[str] = (),
                setup_behaviors: Sequence[str] = (), later_turns: Sequence[str] = (),
                evaluation_level: int = 3, test_case_id: str | None = None,
                version: str = "1.0.0", created_at: str | None = None,
                display_name: str | None = None, description: str | None = None,
                language: str = "en", persona_records_path: str = "persona_records"
                ) -> TestCaseWriteResult
create_persona(persona_id: str, rendering: PersonaRendering | Mapping,
               elevated: Sequence[Elevation | Mapping], *, version: int = 1,
               provenance: Provenance | Mapping | None = None, variant_of: str | None = None,
               changed_paths: Sequence[str] | None = None, overwrite: bool = False,
               directory: str = "persona_records") -> PersonaWriteResult
```

Every result is a `WriteResult`: `written`, `path` (relative to the data repository when under it), `action` (`created`, `overwritten`, `appended`, `updated`), `diff` (unified diff of the file, before → after), `problems` (why nothing was written) and `notes` (worth a look, did not stop the write). Each adds:

- `ManifestWriteResult`: `template_path`, `template_diff` (template → new file), `manifest` (the validated `RunManifest`), `preflight` (a Stage 1 `PreflightReport` on the written file).
- `ModelAddResult`: `model` (`ModelSummary`), `env` (`ModelEnvStatus`: each env var the endpoint needs and whether it is set). `notes` says "set X before using M" for each unset one.
- `TestCaseWriteResult`: `test_case`, `expected_behaviors` (`BehaviorInfo`: id, catalog label, definition, `pathway` or `setup_behaviors`).
- `PersonaWriteResult`: `persona`, `changed_paths` (worked out for a variant), `base_diff` (base persona → variant).

`ManifestOverrides` fields: `run_id`, `evaluation_level`, `system_under_test` (list), `person_response_model`, `fidelity_model`, `judge_model` (every judge), `max_turns`, `batch_size`, `test_case_records_path`, `persona_records_path`, `run_dir`, and `set` (dotted path with list indexes → value, applied last; `null` removes the key), e.g. `{"judges.0.options.max_concurrency": 1}`.

## Decisions

- **Problems are data, and a problem means nothing is written.** Like `preflight`, no writer raises for a problem in its inputs.
- **Manifests are written in the committed layout.** `dump_manifest` is PyYAML with a small dumper subclass: 4-space mappings, list items at the parent's column (`- model_id:`, with `temperature:` two columns in on the next line). Every committed upstream manifest round-trips byte for byte through it. The template's raw mapping is edited, not a dumped `RunManifest`, so defaults the template leaves out stay out. Comments other than the schema line are dropped; the first line is always `# yaml-language-server: $schema=./schema/run_manifest.schema.json`.
- **A model given as a bare id becomes `{model_id: <id>}`.** The template's sampling params for that role are dropped and the role's default applies (SUT 0.7, person 0.9, fidelity and judge 0.0). Carrying params across models was rejected: a reasoning model needs temperature 1, and a local model's `num_ctx` belongs to that model. Give a `ModelCallConfig` to set params. Kept after review (settings belong to a model), and made visible: when the template set more than `model_id` for that role, `notes` gets one entry per role naming what was dropped (`dropped_settings`).
- **`notes` flags a `run_id` or `run_dir` copied from the template**, since two manifests would then share a results folder name.
- **`add_model` inserts text; it does not round-trip YAML.** No round-trip library is installed (`ruamel.yaml` is not in the environment). The entry goes after the last uncommented line of the `models:` block, so the commented `persona-claude` example stays last, at the block's own indentation. The file is then reloaded with `load_registry(interpolate=False)`; if anything but the new model changed, the original text is put back and the result says so. Values are quoted only when YAML needs it. Reviewed and kept: no `ruamel.yaml` dependency.
- **`draft_test_case` reuses the loader's pieces.** Expected behaviors come from `expected_behaviors` (through a `model_construct` probe with only `pathway_id` and `setup_behaviors`), the pin from `KlPin.from_library`, the persona from `load_persona`, and the final check from `check_test_case_records`. Called without `assertions`, it returns the expected behaviors with their catalog labels and writes nothing, so an agent can ask first and write second. `behavior_assertions` keep the order the caller gave.
- **Assignment and elevation checks** (`context_value_problems`): each id is a context value; a persona's sit on person dimensions and a test case's on others (rules T2, P3); no two share a `cardinality: one` dimension. Draft entries are allowed. Liveness is not checked, because the student entries are `draft`.
- **Key order and encoding match the committed records.** Test cases: the fixed key order every committed line uses (`TEST_CASE_KEY_ORDER`), optional keys after, `json.dumps(ensure_ascii=False)`. Personas: `indent=2`, `ensure_ascii=False`, trailing newline, top-level keys in model order with `provenance` and `variant_of` always present; inside the rendering, the caller's key order is kept (`ordered_like`).
- **Variants.** `variant_of` takes the base's pinned ref, which must resolve. `changed_paths` defaults to the rendering's leaf differences from the base (lists compare whole), counting a field the variant removes as changed (TEST_CASE_RULES P7: "the rendering paths that differ"). When given, they are used as given, and any mismatch is a note.
- **Where files go.** `records_file`: a bare name (with or without `.jsonl`) is `test_case_records/<name>.jsonl`; a path resolves like `resolve_data_path`, but a relative path that does not exist yet goes under the data repository, not the working directory (`data_target`). Manifests always go to `run_manifests/`; `out_name` must be a plain file name.
- **Refusals.** Manifests and personas are not replaced unless `overwrite=True`. A test case id already in the file is refused. A duplicate model id is refused.

## Testing

59 new tests in `tests/api/` (`test_create_manifest.py`, `test_add_model.py`, `test_records_write.py`, `test_recreate_upstream.py`); the API suite is 133.

The recreation tests (`test_recreate_upstream.py`) use upstream records only. They read the committed files, never write them, and build into a temporary data repository:

- `l3_azure.yaml` rebuilt from `l3_local` with overrides: **byte-identical**.
- `l3_launch_entrenchment_smoke.yaml` rebuilt from `l3_local`: the same parsed mapping and `RunManifest`. It is not byte-identical, because overrides add `enabled` and `model` after the template's `prompt_version`.
- Every line of `l3_launch_entrenchment_small.jsonl` (19 test cases) appended by `draft_test_case`: **byte-identical**.
- All 250 committed personas, bases then variants (`-nt` on the base, `-nt-np` on `-nt`), written by `create_persona` with the committed `changed_paths`: **byte-identical**.

They load the library bundled with coralmap, which the upstream records pin. They skip if a coralmap upgrade bundles another library. AGENTS.md says tests load the synthetic library, not the real entries. These tests are an accepted exception, like `test_launch_records.py`: rebuilding pinned records needs the library they pin.

`add_model` is tested on a copy of the committed `endpoints.yaml`. Deleting the three inserted lines gives back the original file exactly. No test, and no part of this work, wrote the real `endpoints.yaml`.

## `pixi run check`

- In `~/toolkit`, after switching `~/knowledge-library` from `student-homework` to `main` (with permission; the branch is untouched, `git checkout student-homework` restores it): lint, format, mypy, schemas, viewer types and yamllint pass. Tests: 779 passed, 1 failed, 4 skipped. The failure is `test_committed_run_config_validates[l3_launch_entrenchment_smoke_local.yaml]`, the untracked local manifest with no schema line, left alone as instructed.
- In a clean worktree of the branch at `9ee9d32`: every step passes, 769 passed and 14 skipped. The skips are sibling-catalog tests, because no catalog sits beside the worktree.

## Not verified

- `add_model` against registries laid out differently from the committed one (e.g. `models:` written as a flow mapping `{}`, which it refuses, or entries separated by blank lines, where the new one is not preceded by a blank line).
- A run of a manifest written by `create_manifest`. Only preflight ran on them; no model was called.
- Persona elevations read from prose are checked for value and dimension only, not that the quoted span appears in the rendering.

## Upstream finding: variants omit removed fields from `changed_paths`

All 112 upstream persona variants list the fields they change but omit the field they remove from their base:

| Variants | Count | Field removed but not listed |
| --- | --- | --- |
| `-nt` | 73 | `rendering.traits.clinical_history.treatment_engagement` |
| `-nt-np` | 39 | `rendering.traits.clinical_history.insight` |

No variant lists a path that does not differ. P7 defines `changed_paths` as the rendering paths that differ, so removals should count; the schema checks only the `rendering.` prefix, so nothing catches it. This is reported upstream separately and not fixed here: no committed persona record was edited. `create_persona` counts removals when it works out `changed_paths`; the byte-identical recreation test passes the committed lists explicitly, and `test_variant_counts_removed_fields_as_changed` covers removal detection.

## For the next stage

- The writers are not on the CLI or an MCP server yet. Their arguments are plain or pydantic, so a tool schema can come from the signatures. `ManifestOverrides` is the one composite input.
- For an agent, a test case takes two calls: `draft_test_case(..., assertions=None)` to read `expected_behaviors`, then again with one assertion per id.
- `create_manifest` runs preflight with `query_ollama=True` by default (a metadata request to a local Ollama, no model call) and `dotenv=True`. Tests pass `False` for both.
- Test fixtures: `data_repo` (Stage 1) suits the writers. `tests/api/test_recreate_upstream.py` has an `upstream` fixture: an empty data repository on the bundled library.
- Still not built: running the AI under test through headless `claude -p`; running and reading runs.
