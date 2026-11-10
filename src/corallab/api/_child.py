"""Run a command and write its exit code to a file: ``python -m corallab.api._child EXIT -- CMD``.

``start_run`` starts ``corallab run`` through this so that a later process, which cannot wait
on the run, still learns how it ended. SIGTERM is caught here and the command, which shares
the process group, ends on it; its exit code is then written as usual.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        sys.stderr.write("usage: python -m corallab.api._child EXIT_FILE -- COMMAND...\n")
        return 2
    exit_file = Path(argv[0])
    signal.signal(signal.SIGTERM, lambda *_: None)
    code = subprocess.call(argv[2:])
    exit_file.write_text(f"{code}\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
