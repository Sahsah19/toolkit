"""Helpers the record and manifest writers share."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from corallab import paths

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def unified_diff(before: str, after: str, name: str, *, before_name: str | None = None) -> str:
    """A unified diff of two texts; empty when they are equal."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=before_name or name,
            tofile=name,
        )
    )


def data_target(value: str | Path) -> Path:
    """Where a writer puts ``value``: ``paths.resolve_data_path``, with a relative path that
    does not exist yet placed under the data repository rather than the working directory."""
    path = paths.resolve_data_path(value)
    return path if path.is_absolute() else paths.data_dir() / path


def read_text(path: Path) -> str:
    """The file's text, or an empty string when it does not exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def name_problem(name: str, what: str) -> str | None:
    """Why ``name`` cannot be a file name under the data repository, or None."""
    if not _NAME.match(name):
        return f"{what} {name!r} must be a plain file name: letters, digits, '_', '.', '-'"
    return None


def validation_problems(exc: ValidationError, prefix: str = "") -> list[str]:
    """One message per pydantic error, with its field path."""
    return [
        f"{prefix}{'.'.join(map(str, e['loc'])) or '(record)'}: {e['msg']}" for e in exc.errors()
    ]


def ordered_like(data: Any, like: Any) -> Any:
    """``data`` with each mapping's keys in ``like``'s order where ``like`` has them.

    Keys ``like`` lacks follow in ``data``'s own order. Keeps a caller's key order when a
    record is validated and dumped again.
    """
    if isinstance(data, dict) and isinstance(like, dict):
        keys = [k for k in like if k in data] + [k for k in data if k not in like]
        return {k: ordered_like(data[k], like.get(k)) for k in keys}
    if isinstance(data, list) and isinstance(like, list) and len(data) == len(like):
        return [ordered_like(d, w) for d, w in zip(data, like, strict=True)]
    return data
