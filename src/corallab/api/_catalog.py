"""Loading the Knowledge Library without raising, for preflight and the record writers."""

from __future__ import annotations

from typing import Any

from corallab import paths


def load_catalog() -> tuple[Any, str | None]:
    """``(library, None)``, or ``(None, problem)`` when it does not load.

    The library is the one ``corallab.kl.load_library`` resolves: ``CORALMIND_KL_PATH``,
    else the one bundled with ``coralmap``.
    """
    from corallab.kl import load_library

    kl_path = paths.knowledge_library_path()
    if kl_path is not None and not kl_path.exists():
        return None, f"CORALMIND_KL_PATH names {kl_path}, which does not exist"
    try:
        return load_library(), None
    except Exception as exc:  # LibraryParseError and anything coralmap raises on a bad tree
        return None, f"the Knowledge Library did not load: {exc}"
