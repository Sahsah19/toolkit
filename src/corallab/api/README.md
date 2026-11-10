# `corallab.api`

A typed Python layer for agents that configure, check and (in later stages) run evaluations. It holds the logic; the CLI and an MCP server are meant to be thin layers over it. Every function takes plain arguments and returns pydantic models from `models.py`, so a tool's input and output schema comes from these classes.

It reuses the toolkit's own loaders, validators and schemas (`RunManifest`, `load_test_case_records`, `check_test_case_records`, the endpoint registry, the judge's prompt builders). It does not change any result format.

## Functions

| Function | Returns |
| --- | --- |
| `list_manifests(directory=None)` | `list[ManifestSummary]`: each run manifest, validated, with its models and key settings |
| `list_test_cases(path=None)` | `list[TestCaseSummary]`: every test case line in one file, a directory, or all of `test_case_records/` |
| `list_personas(directory="persona_records")` | `list[PersonaSummary]` |
| `list_models(registry=None)` | `list[ModelSummary]`: each registry model, its endpoint and the env-var names it needs |
| `check_env(manifest=None, *, registry=None, dotenv=True)` | `EnvReport`: for each model, the env vars it needs and whether each is set |
| `load_manifest(manifest)` | `RunManifest` from a path, a manifest id or a `RunManifest` |
| `preflight(manifest_path, *, query_ollama=True, ollama_timeout=2.0, transcript=None, dotenv=True)` | `PreflightReport`: errors, warnings, catalog pin, env, model-call estimates, Ollama context |

### Writers

| Function | Writes |
| --- | --- |
| `create_manifest(from_template, overrides, out_name, *, overwrite=False, query_ollama=True, dotenv=True)` | `run_manifests/<out_name>.yaml` from a manifest plus `ManifestOverrides`, then runs `preflight` on it |
| `add_model(model_id, endpoint, litellm_model, *, base_model=None, registry=None, dotenv=True)` | One entry at the end of `models:` in `endpoints.yaml`; every other line kept. Reports the env vars to set |
| `draft_test_case(records_file, *, pathway_id, persona_ref, scenario, opening, scenario_strategies, assertions, ...)` | One line appended to a test case JSONL file, pinned to the loaded library, with one assertion per expected behavior |
| `create_persona(persona_id, rendering, elevated, *, version=1, provenance=None, variant_of=None, changed_paths=None, overwrite=False)` | `persona_records/<persona_id>.json`; for a variant, works out `changed_paths` |

Each returns a `WriteResult`: `written`, `path`, `action`, `diff` (the file before → after), `problems` (why nothing was written) and `notes`. A writer never raises for a problem in its inputs and writes nothing when there is one. Manifests and personas are not replaced unless `overwrite=True`. Output follows the committed files' layout: the committed manifests, test cases and personas rebuild byte for byte (`tests/api/test_recreate_upstream.py`).

A file or line that does not validate is listed with `error` set, not dropped. `preflight` reports problems instead of raising them.

## Rules this layer keeps

- No secret is read out. Env vars are reported as set or unset by name. The registry is read without `${VAR}` interpolation, so no key or configured address is resolved.
- `preflight` makes no model calls. Its only network request is an optional metadata query (`/api/show`, `/api/ps`) to a local Ollama server.
- `dotenv=True` loads `.env` the way the CLI callback does. Tests turn it off.
