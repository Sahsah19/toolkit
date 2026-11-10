"""Stands in for ``corallab run`` in the start_run tests: ``python _fake_run.py MANIFEST MODE``.

It names its results folder the way the CLI does (``timestamped_run_dir``), copies the inputs,
and writes a log line and a trace row. Then, by MODE:

- ``fail``: exits 3 before creating any folder;
- ``sleep``: sleeps, to be cancelled;
- ``ok``: writes a transcript and exits 0;
- ``crash``: exits 1 after writing.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from corallab.api._manifests import read_run_manifest
from corallab.cli._helpers import timestamped_run_dir
from corallab.paths import resolve_data_path, resolve_run_dir


def main(manifest_path: Path, mode: str) -> int:
    if mode == "fail":
        sys.stdout.write(
            "Error: could not reach https://example.invalid/v1 (api_base=https://x.invalid)\n"
        )
        return 3
    data = read_run_manifest(manifest_path)
    out = timestamped_run_dir(resolve_run_dir(data["run_dir"]))
    out.mkdir(parents=True)
    inputs = out / "inputs"
    inputs.mkdir()
    shutil.copy2(manifest_path, inputs / manifest_path.name)
    records = resolve_data_path(data["test_case_records_path"])
    shutil.copy2(records, inputs / records.name)
    (out / f"{out.name}.log").write_text(
        "2026-11-10 10:00:00,000 INFO corallab: Log file\n"
        "2026-11-10 10:00:01,000 ERROR corallab.cli.run: Sim failed: api_base=https://x.invalid\n",
        encoding="utf-8",
    )
    batch = out / "batch-001"
    batch.mkdir()
    row = {"type": "llm_call", "role": "sut", "resolved_litellm_model": "azure/model"}
    (batch / f"{out.name}-sim-batch-001.trace.jsonl").write_text(json.dumps(row) + "\n")
    if mode == "sleep":
        time.sleep(120)
        return 0
    first = records.read_text(encoding="utf-8").splitlines()[0]
    test_case_id = json.loads(first)["test_case_id"]
    transcript = {
        "transcript_id": f"{data['run_id']}:remote-model:{test_case_id}:1",
        "test_case_id": test_case_id,
        "test_case_version": "1.0.0",
        "run_id": data["run_id"],
        "system_under_test": "remote-model",
        "sample": 1,
        "turns": [
            {"turn_number": 1, "role": "user", "content": "hi"},
            {"turn_number": 1, "role": "assistant", "content": "hello"},
        ],
        "exit_reason": "max_turns",
    }
    (batch / f"{out.name}-sim.jsonl").write_text(json.dumps(transcript) + "\n")
    return 1 if mode == "crash" else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2]))
