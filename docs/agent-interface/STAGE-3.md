# Stage 3: starting, watching and reading runs

Branch `agent-interface`, code commit `80fc0e8`, on top of Stage 2. Built and tested against upstream data only: the synthetic fixture library, temporary data repositories, and the one published sample run (read only).

## What was built

Nine functions in `src/corallab/api/`, exported from `corallab.api`:

| Module | Holds |
| --- | --- |
| `runs.py` | `start_run`, `run_status`, `cancel_run`, `delete_run`, `list_runs`, `run_command` |
| `results.py` | `summarize_run`, `get_transcript`, `get_verdicts`, `compare_runs`, `pathway_applies` |
| `_results.py` | Finding a run folder (`find_run`), listing its files (`run_files`), reading records with the toolkit's schemas (`read_records`), reading trace calls (`trace_calls`, `call_role`) |
| `_burr.py` | Burr tracker attribution (`attribute`, `named_app_pattern`) and `--db-path` persister rows |
| `_child.py` | `python -m corallab.api._child EXIT -- CMD`: runs a command and writes its exit code |
| `models.py` | `StartRunResult`, `RunStatus`, `CancelResult`, `DeleteRunResult`, `BurrApp`, `RunListing`, `CallCount`, `RunSummary`, `CaseResult`, `TranscriptResult`, `PathwayResult`, `GateResult`, `BehaviorFound`, `StrategyResult`, `ExpectedBehaviorResult`, `CatalogUse`, `CallComparison`, `TokenCount`, `TranscriptsResult`, `TranscriptView`, `VerdictsResult`, `TranscriptVerdicts`, `RunComparison`, `RunHeader`, `PathwayComparison`, `PathwayRate`, `BehaviorComparison`, `BehaviorRate` |

Nothing outside `api/` changed. No result format, schema, CLI command or viewer input changed. No new dependency.

## Signatures

```python
start_run(manifest_path: str | Path, test_case_ids: list[str] | None = None, *,
          wait_seconds: float = 120.0, query_ollama: bool = True,
          dotenv: bool = True) -> StartRunResult
run_status(run_id: str) -> RunStatus
cancel_run(run_id: str, *, grace_seconds: float = 10.0) -> CancelResult
delete_run(run_id: str, include_burr: bool = True, *, dry_run: bool = True,
           burr_dir: str | Path | None = None,
           db_path: str | Path | None = None) -> DeleteRunResult
list_runs(*, include_samples: bool = False) -> list[RunListing]
summarize_run(run_id: str | Path) -> RunSummary
get_transcript(run_id: str | Path, test_case_id: str, *,
               system_under_test: str | None = None, sample: int | None = None
               ) -> TranscriptsResult
get_verdicts(run_id: str | Path, test_case_id: str | None = None, *,
             include_raw: bool = False) -> VerdictsResult
compare_runs(a: str | Path, b: str | Path, *,
             fidelity_valid_only: bool = False) -> RunComparison
```

A `run_id` is the results folder's name, e.g. `api-test-20261110-100000-2`. It is looked up in `start_run`'s records, then `results/`, then `sample_results/`. The four reading functions also take a path to a run folder. `delete_run` takes an id only.

### What each returns

- `StartRunResult`: `started`, `run_id`, `results_dir` (relative to the data repository), `pid`, `run_manifest` (the manifest the run was started with), `command`, `problems` (why nothing started), `warnings`, `preflight` (the full `PreflightReport`).
- `RunStatus`: `stage` (`starting`, `sim`, `check`, `judge`, `finished`, `failed`, `cancelled`), `alive`, `exit_code`, `cancelled`, `batches_total`/`batches_simulated`, `transcripts_total`/`transcripts_done`, `checks_done`, `judged_done`, `judge_calls_done`, `calls` (per role and model, with tokens), `errors`, `started_at`, `elapsed_seconds`.
- `DeleteRunResult`: `files` and `bytes` of the results folder, `agent_files`, `burr_apps` (each with why it belongs to the run), `burr_unattributed` (kept), `db_rows`, `deleted`.
- `RunSummary`: run-wide counts, `fidelity` per status, `calls` and `tokens` from the traces, `catalog` (`test_case_pins`, `judge`, `produced_by`, `loaded`), `estimate` (`CallComparison` per role: `estimated`, `upper_bound`, `actual`, `difference`, `basis`, `note`), and `test_cases`, each with a `TranscriptResult` per conversation: fidelity status and source, strategy adherence, expected behaviors, judge status, model, prompt version, catalog version and digest, `pathways` (support, gate status, `applies`, instances, each gate's verdict, basis and conflict, `target`), skipped and failed pathways, `behaviors_found`, judge calls, retries, failures and tokens, and the trace calls tied to it.
- `RunComparison`: a `RunHeader` per run, `pathways` (per pathway: judged, assessed, counts per support level, `applies`, `rate` for each run, and the difference), `behaviors` (found, judged, rate), test cases in only one run, `notes`.

## Decisions

- **The run's id is the folder the CLI really created.** `start_run` lists the folders beside the manifest's `run_dir` that match `<name>-YYYYMMDD-HHMMSS[-N]`, starts the process, and waits for exactly one new one. Starts are serialized with a file lock in `results/.agent-runs/`, so two `start_run` calls never see each other's folder. A run started outside the API in the same moment could still make two new folders; `start_run` then refuses and stops its process. A test pre-creates the next 30 seconds' folder names and checks that the `-2` folder is returned.
- **Process records live in `results/.agent-runs/`.** One JSON file per run (pid, command, start time, output and exit files). `results/` is gitignored, and the viewer lists only folders holding transcript or judge files, so it skips this one. The run's own folder is not touched: the result layout is unchanged.
- **The child is wrapped to record its exit code.** `corallab run` is started as `python -m corallab.api._child <exit file> -- python -m corallab.cli run --run-manifest M`, in a new session. The wrapper catches SIGTERM and writes the command's exit code, so a later process (a CLI call, a restarted MCP server) can tell `finished` from `failed`. Liveness from another process uses signal 0, then `ps` to rule out a zombie or a reused pid.
- **Cancel** sends SIGTERM to the process group, then SIGKILL after `grace_seconds`. Only runs `start_run` started can be cancelled; others report that their process is not known.
- **Stage** while alive is the stage of the newest trace file (`-judge.trace.jsonl`, `-check.trace.jsonl`, else sim). After exit it is `finished` (exit 0), `failed` or `cancelled`. For a run not started by the API, `alive` is `None` and the stage is the last one written.
- **Errors** are the run log's ERROR lines (first line of each, addresses such as `api_base=…` and URLs replaced with `<address>`, since the log prints the endpoint for every call), failed simulations, checks without a verdict, judge results that are not `success`, unreadable lines, and on a non-zero exit the end of the process output, also with addresses removed. At most the last 20.
- **`test_case_ids` narrows by copying.** `corallab run` has no filter. The manifest and the records are copied to `results/.agent-runs/inputs/<token>/`, the records keeping each chosen line byte for byte under their own `test_case_records/` folder, so the viewer names the suite as it names the original. The copy's `test_case_records_path` is relative to the data repository, so no home path lands in `inputs/`. Preflight runs on the copy, so its estimate is for the narrowed run. The copies are kept for the run's life (the judge reads the records path) and removed by `delete_run`.
- **Warnings** carry preflight's warnings, other manifests sharing this one's `run_dir` (their runs get folders with the same name) or `run_id`, and a run of the same manifest still alive.
- **Burr attribution.** Named apps match `^<stem>(__<key>)?(-sim-batch-NNN|-batch-NNN|-judge)$`, which covers `corallab run`, `corallab sim`, `corallab judge` and several systems under test, and does not match a `-2` run of the same second. Nested apps (per test case, per L3 conversation, per judge pass) have generated 64-hex folder names. One is attributed only by following its own `metadata.json` (`spawning_parent_pointer`, else `parent_pointer`) up to a named app. A folder whose chain hits a missing folder, missing or unreadable metadata, or a cycle goes in `burr_unattributed` and is never deleted; a chain that ends at another run's root is left out. `partition_key` is `corallab` for every app the CLI makes, so it cannot tell runs apart and is not used. Timestamps are never used.
- **`--db-path`.** The CLI's own Burr databases sit in the run folder and go with it. For a persister file given with `--db-path`, `delete_run(db_path=...)` deletes only the rows whose `app_id` is the run's (named, or attributed through the tracker), and keeps the file.
- **`delete_run` refuses** anything under `sample_results/` (checked on the resolved path), a run still alive, a run outside `results/` that `start_run` did not record, and any id that is a path, `..`, or starts with a dot. It is a dry run unless `dry_run=False`.
- **Records are read with the toolkit's schemas.** `Transcript`, `FidelityCheckRecord` and `EvalResult`; a line that does not validate is a problem naming its file and line. When a transcript was judged more than once the latest `evaluated_at` wins, as in the viewer. The fidelity verdict comes from the check file, else from `Transcript.fidelity` (runs from before the check had its own file, like the sample).
- **`applies` is the viewer's "fires".** A supported instance (or, with no instances, a supported verdict) and gate status `met` or `no_gates`. `compare_runs` rates are `applies / judged transcripts`; `fidelity_valid_only` counts only transcripts whose check is `valid`.
- **Call counts come from the trace files**, mapped to preflight's roles (`sut` → `system_under_test`, `person_response_generator` → `person_response`, `coralmind_s1_*` → `judge_stage1`, `coralmind_s2_*` → `judge_stage2`). A trace row is written only for a response; a call that raised is not traced. Dry-run traces (`batch-NNN/dry_run/`) are not counted. Sim trace rows do not name a test case, so per-transcript sim calls are given only when the batch holds one transcript.
- **The estimate comparison** reruns Stage 1's `estimate_calls` on the run's own `inputs/` copy with the library loaded now. When that library is not the one the run pinned or judged with, a note says the judge estimate counts the wrong catalog's definition groups.
- `.json` manifests in `inputs/` are read with `json`, not YAML: the sample's copied manifest is tab-indented JSON, which YAML rejects.

## Estimates against a real run

The only real run on upstream data available is the published sample (`l3-launch-entrenchment-smoke-20261019-133536`: one test case, `max_turns` 10, written before the check had its own file, no judge results). `summarize_run` on it:

| Role | Estimated | Actual | Note |
| --- | --- | --- | --- |
| system_under_test | 10 (upper bound) | 10 | at the upper bound: the conversation ran to `max_turns` |
| person_response | 9 (upper bound) | 9 | at the upper bound |
| elicitation_fidelity | 1 | 1 | exact |
| judge_stage1 | 36 | 0 | no judge output in the sample |
| judge_stage2 | 117 (upper bound) | 0 | no judge output in the sample |

The sim and fidelity estimates match. The judge estimates are still unchecked: the sample has no judge file. The one other upstream-data run on this machine, `results/l3-launch-entrenchment-smoke-local-20261109-234943`, is a `sim --dry-run` with no transcripts. The sample's test case pins catalog `0.2.0 sha256:2624053e96bbdada`, while the installed coralmap bundles `0.1.0 sha256:4564ba13128cc240`; `summarize_run` notes this, since the judge-group count comes from the loaded catalog.

## Testing

51 new tests; the API suite is 184.

- `tests/api/_runs.py` builds synthetic runs in the `corallab run` layout. Every record is validated with `Transcript`, `FidelityCheckRecord` or `EvalResult` before it is written.
- `test_results.py`: `summarize_run`, `get_transcript`, `get_verdicts` and `compare_runs` on synthetic runs (checked and judged, unchecked, a line that does not validate, a re-judged transcript, gates that hold and do not, rates with and without `fidelity_valid_only`, differing judge model and test cases), and all four on the published sample. That test hashes every file of the sample before and after and asserts nothing changed.
- `test_runs.py`: `start_run`, `run_status` and `cancel_run` with `tests/api/_fake_run.py` standing in for `corallab run` (it names its folder with the CLI's own `timestamped_run_dir`): preflight refusal, unknown manifest and test case ids, the `-2` collision, a run followed to `finished`, a `failed` run, a process that exits before making its folder, narrowing to one test case, the shared-`run_dir` warning, cancel, and status and cancel without the `Popen` handle as another process sees them (including the zombie case). One test starts the real `corallab sim --dry-run` (no model calls), follows it to `finished`, attributes its Burr folders (two named, the nested ones through metadata, none unattributed) and deletes it. `delete_run` against a temporary tracker folder with nested, broken, cyclic and other-run apps, `include_burr=False`, a persister file, and the refusals.
- `test_run_helpers.py`: the app-name pattern, attribution through `parent_pointer`, cycles and unreadable metadata, trace parsing (including a half-written last line), run lookup order, and the exit-code wrapper.

HOME points at a temporary folder in every lifecycle test, so Burr's tracker writes there. The real `~/.burr/corallab` held the same 25 folders before and after the suite.

## `pixi run check`

In a clean temporary worktree of the branch at `80fc0e8` (no sibling catalog, no untracked files, `CORALMIND_KL_PATH` unset): every step passed, exit 0. Ruff and format clean, mypy clean on 122 files, tests 823 passed and 14 skipped (the sibling-catalog tests, as in Stage 2). The worktree was then removed. In `~/toolkit` the machine-specific failures described in Stage 1 still apply and were not touched.

## Not verified

- Judge call estimates against a real judged run (above). A real `corallab run` was never started: every run in the tests is the stand-in script or `sim --dry-run`.
- `start_run` on a manifest with several `system_under_test` entries, and `run_status` stage reporting through that fan-out (the CLI runs each model's whole pipeline in turn).
- Cancelling a real `corallab run` mid-call: the tests cancel the stand-in. Whether LiteLLM or Burr leave anything behind after SIGTERM was not checked; `delete_run` lists whatever is there.
- Burr attribution was checked on the folders a real `sim --dry-run` makes and on synthetic metadata, not on a real judge run's nested apps (they use the same `spawning_parent_pointer`, as the real `~/.burr` shows).
- Whether the nested apps write rows to a `--db-path` persister. Only rows of named or tracker-attributed apps are deleted.
- Windows: `start_run` uses `fcntl`, process groups and `ps`.
- The child process loads `.env` the way the CLI always does (`load_dotenv()` in the CLI callback). The `sim --dry-run` test therefore may load the developer's `.env` into the child; it makes no calls and prints nothing from it.

## For the next stage

- All nine functions take and return plain or pydantic values, so CLI commands and MCP tools can be generated from the signatures. `delete_run` is the one destructive tool; keep its `dry_run=True` default on every surface.
- `start_run` is the only function that leads to model calls. An MCP tool for it should say so, and the caller should read `preflight.calls_by_model` first.
- `run_command` is the single place the started command is built; tests replace it. A headless `claude -p` system under test would need an endpoint type in the router, not a change here.
- `run_status` and `list_runs` are cheap but read every sim, check and judge file; on large runs, a status tool polled often may want a lighter path.
- A narrowed run's copied inputs (`.agent-runs/inputs/<token>/`) stay until `delete_run` of that run.
