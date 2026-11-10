# Stage 1: the typed API layer, listings, env check and preflight

Branch `agent-interface` (from `student-homework`). Code commit `2c7d1de`.

## What was built

`src/corallab/api/`, a package of plain functions that return pydantic models (`models.py`). It holds logic; later stages put the CLI and an MCP server over it. `src/corallab/api/README.md` states its purpose and rules.

| Module | Holds |
| --- | --- |
| `models.py` | Every input/output model |
| `listing.py` | `list_manifests`, `list_test_cases`, `list_personas`, `list_models`, `read_test_case_summaries` |
| `env.py` | `check_env`, `load_manifest`, `load_env_file`, `is_set`, `model_env_status` |
| `preflight.py` | `preflight`, `estimate_calls`, `suggested_num_ctx`, `ASSUMED_OLLAMA_CONTEXT` |
| `ollama.py` | `query_ollama`, `http_json`, `ollama_model_name`, `OllamaModelInfo` (metadata only) |
| `_prompts.py` | Prompt sizing over a synthetic transcript; `TranscriptAssumption` |
| `_manifests.py` | `read_run_manifest` (moved here from the CLI), `resolve_manifest_path`, `manifest_roles`, `model_roles`, `fidelity_model`, `relative_to_data`; role names |

Changes outside `api/`, each so the API reuses code instead of copying it:

- `simulation/data/test_case_records.py`: `load_test_case_records` now records each problem's kind as `RecordProblem(kind, message, test_case_id, line)` on `TestCaseRecordsError.details`, kind one of `invalid`, `repeat`, `pin`, `level`, `persona`. `problems` and every message are unchanged. `_load_persona` is now public as `load_persona`.
- `llm/registry_loader.py`: `load_registry(path, *, interpolate=True)`; `interpolate=False` keeps `${VAR}` tokens so nothing is read from the environment. New `endpoint_env_vars(endpoint) -> list[str]`.
- `cli/_helpers.py`: `read_run_manifest` is imported from `corallab.api._manifests`; behavior unchanged.

No result format, schema or viewer input changed.

## Signatures

```python
list_manifests(directory: str | Path | None = None) -> list[ManifestSummary]
list_test_cases(path: str | Path | None = None) -> list[TestCaseSummary]
list_personas(directory: str | Path = "persona_records") -> list[PersonaSummary]
list_models(registry: str | Path | None = None) -> list[ModelSummary]
check_env(manifest: RunManifest | str | Path | None = None, *,
          registry: str | Path | None = None, dotenv: bool = True) -> EnvReport
load_manifest(manifest: RunManifest | str | Path) -> RunManifest
preflight(manifest_path: str | Path, *, query_ollama: bool = True,
          ollama_timeout: float = 2.0, transcript: TranscriptAssumption | None = None,
          dotenv: bool = True, fetch: Fetch = http_json) -> PreflightReport
```

`manifest_path` and `manifest` accept a path, a path under the data repository, or a bare id (`l3_student_homework_smoke`).

`PreflightReport`: `ok` (no errors), `errors` and `warnings` (`Issue`: `severity`, `check`, `message`, `subject`), `catalog` (`CatalogInfo`: `kl_path`, loaded `version`/`digest`, the test cases' pins, `matches`), `env` (`EnvReport`), `calls` (`RoleCalls` per role and model: `calls`, `upper_bound`, `basis`), `calls_by_model`, `ollama` (`OllamaContext` per local model, with a `PromptEstimate` per role).

Issue `check` values: `manifest`, `models`, `env`, `catalog`, `catalog_pin`, `test_cases`, `personas`, `pathways`, `fidelity_prompt`, `judge_options`, `ollama`.

## What preflight checks

Errors: manifest missing, not YAML, or invalid (one issue per field); empty `run_dir` or `test_case_records_path`; fidelity prompt version unreadable (`prompt_problem`); judge options invalid or prompt directory missing; a model id not in `endpoints.yaml`; a needed env var unset (named, never shown); the library does not load or `CORALMIND_KL_PATH` names a missing path; records file missing or holding no valid test case; each loader problem (pin, level, persona, repeat, invalid line); each `check_test_case_records` problem (pathway and expected behaviors).

Warnings (never raise, never set `ok` false): unknown judge options; Ollama unreachable; an Ollama prompt larger than the context its role gets, with a suggested `extra_params: {num_ctx: N}`; a prompt over the model's trained maximum; estimates that could not be made.

Call estimates, before retries, multiplied by the number of `system_under_test` entries:

| Role | Count | Exact? |
| --- | --- | --- |
| system_under_test | L1: cases; L2: scripted turns; L3: cases × max_turns | L3 is an upper bound |
| person_response (L3) | cases × (max_turns − 1) | upper bound |
| elicitation_fidelity (L3, enabled) | one per transcript; `run` and `check`, not `sim` | exact |
| judge_stage1 | cases × definition groups at `group_size` (the judge's own `build_definition_groups`) | exact when every transcript has turns |
| judge_stage2 | cases × ceil(pathways / `pathways_per_call`), every pathway a candidate | upper bound |

## Results on the named manifests (run 2026-11-10)

- `l3_launch_entrenchment_smoke`: one error, `AZURE_API_KEY is not set; needed by azure/gpt-5.4-mini (system_under_test, person_response, elicitation_fidelity, judge)`. 173 calls at most, all on that model (Stage 1: 36 groups; Stage 2: up to 117).
- `l3_student_homework_smoke`, without `CORALMIND_KL_PATH`: one `catalog_pin` error (pins `sha256:ab31949defeba3f4`, the bundled library is `sha256:4564ba13128cc240`), with a note that the variable is unset. Neither the shell nor `.env` set it.
- `l3_student_homework_smoke`, with `CORALMIND_KL_PATH=~/knowledge-library`: `ok`. Stage 1 is 37 groups, matching PROPOSAL.md's "37 screening groups". At most 165 Gemini calls and 5 `local/llama3.2` calls. One warning: the student's prompt is about 2,980 tokens against an assumed 2,048 context, so the start of it (the persona block) would be cut. That may explain the off-script student in the proposal's smoke run; not verified.
- `l3_launch_entrenchment_smoke_local` (the untracked file, read only): five warnings. Judge Stage 1 prompts reach about 16,070 tokens against 2,048, consistent with the cut at about 2,050 seen in practice.

## Decisions

- **Problems are data.** Preflight never raises for a problem in the inputs. Listings keep invalid files and lines with `error` set.
- **No secret is resolved.** The registry is read with `interpolate=False`. `api_base` appears in `ModelSummary` only when written literally (the Ollama URL). Tests assert that set key values and addresses never appear in any output.
- **`.env` is loaded like the CLI** (`load_dotenv()`, existing variables win) when `dotenv=True`. The API never reads or returns its contents. Tests replace `load_env_file` with a no-op.
- **Loader authority.** Test case checks come from `load_test_case_records` and `check_test_case_records`. Because the loader stops at the first problem per test case, preflight also loads the personas of test cases whose pin or level failed, so a pin mismatch does not hide a missing persona.
- **Prompt sizes use the real builders** (`PersonResponseGenerator._build_system_prompt`/`_build_messages`, `fidelity.elicitation._build_prompt`, `build_stage1_prompt`, `build_stage2_prompt`) over a synthetic transcript of `max_turns` exchanges, counted with `litellm.token_counter` as `--dry-run` does. Defaults are about 60 tokens per user message and 350 per AI reply (`TranscriptAssumption`). Two of those builders are private; if they are renamed, `_prompts.py` breaks loudly in the tests.
- **Ollama context.** In order: `num_ctx` in the role's `extra_params`, `num_ctx` in the Modelfile (`/api/show`), the loaded context (`/api/ps`), then an assumed 2,048 (the observed cut). The server's own default (`OLLAMA_CONTEXT_LENGTH`) is not visible to a client. The suggested `num_ctx` is a power of two with 25% headroom, capped at the trained maximum.
- Only Ollama models get prompt sizing. Hosted models have large contexts and are not compared.

## Not verified

- That `extra_params: {num_ctx: N}` reaches Ollama through `LLMClient` → LiteLLM `ollama_chat`. LiteLLM's `OllamaChatConfig` lists `num_ctx`, and `complete()` forwards extra params with `drop_params=True`, but no call was made.
- The effective default context of the local Ollama 0.40.16 server. 2,048 is an assumption from the observed cut.
- Token counts are approximate: LiteLLM falls back to a generic tokenizer for `ollama_chat/*` and `gemini/*`.
- No model was called, so no estimate was compared against a real run's trace.

## `pixi run check`

In `~/toolkit` it reports 8 failures, all present before this stage and none in code this stage touched:

- 7 are the catalog-pin coupling PROPOSAL.md describes: `~/knowledge-library` is on `student-homework`, so the upstream records' pins do not match it.
- 1 is `test_committed_run_config_validates[l3_launch_entrenchment_smoke_local.yaml]`: the untracked local manifest has no schema comment line. It is to be left alone.

A clean worktree of `2c7d1de`, with no sibling catalog (those tests skip) and no untracked manifest, passed every step: ruff, format, mypy, schemas, viewer types, yamllint, and `709 passed, 14 skipped`. Stage 1 adds 75 tests (`tests/api/`, plus additions to `tests/llm/test_registry_loader.py` and `tests/simulation/data/test_load_test_case_records.py`).

## For the next stage

- The CLI does not use the API yet. `corallab check-env` still has its own scraper (`referenced_vars`); it can become a view of `check_env()`. `validate run-manifest` and `validate test-cases` could print `preflight` issues.
- An MCP server can expose the seven functions directly; their argument and return types are already pydantic or plain.
- `TestCaseRecordsError.details` gives structured loader problems to any new caller.
- Tests use `tests/api/_repo.py` (`DataRepo`, `manifest_data`, `REGISTRY`) and the `data_repo`, `ollama_fetch` and `ollama_down` fixtures in `tests/api/conftest.py`. Import helpers from `tests.api._repo`: `from conftest import` resolves to `tests/conftest.py`.
- Running the AI under test through a headless `claude -p` session is not built. It needs an endpoint type the router does not have.
- The `run_id` a manifest omits is filled with `run-generated` for validation only; the CLI still generates its own.
