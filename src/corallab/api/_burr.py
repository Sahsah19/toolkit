"""Which Burr tracker folders and SQLite persister rows belong to a run.

The CLI names each Burr application after the run's file stem (``with_identifiers(app_id=...)``
in ``cli/_runners.py``): ``<stem>-sim-batch-NNN`` and ``<stem>-judge`` from ``corallab run``,
``<stem>-batch-NNN`` from ``corallab sim``, with ``__<key>`` after the stem when a manifest
lists several systems under test. Applications Burr spawns inside them (one per test case, one
per L3 conversation, one per judge pass) get generated folder names. Such a folder is
attributed to a run only through its own ``metadata.json``: the spawning or parent application
it names, followed up to a named one. A folder whose chain cannot be followed is reported,
never attributed by its timestamp or its neighbours.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

PERSISTER_TABLE = "corallab_state"
"""The table ``cli/_runners.py`` gives ``AsyncSQLitePersister``."""


def default_burr_dir() -> Path:
    """Where the local tracker writes for project ``corallab``: ``~/.burr/corallab``."""
    return Path("~/.burr").expanduser() / "corallab"


def named_app_pattern(run_stem: str) -> re.Pattern[str]:
    """The application ids the CLI gives a run with this stem."""
    stem = re.escape(run_stem)
    return re.compile(rf"^{stem}(?:__[A-Za-z0-9._-]+?)?(?:(?:-sim)?-batch-\d+|-judge)$")


@dataclass(frozen=True)
class AppFolder:
    app_id: str
    path: Path
    parent: str | None
    """The spawning (else parent) application its metadata names; None for a root."""
    error: str | None = None


def _read_folder(path: Path) -> AppFolder:
    meta = path / "metadata.json"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AppFolder(path.name, path, None, "no metadata.json")
    except (OSError, json.JSONDecodeError) as exc:
        return AppFolder(path.name, path, None, f"metadata.json unreadable: {exc}")
    if not isinstance(data, dict):
        return AppFolder(path.name, path, None, "metadata.json is not a mapping")
    for key in ("spawning_parent_pointer", "parent_pointer"):
        pointer = data.get(key)
        if isinstance(pointer, dict) and pointer.get("app_id"):
            return AppFolder(path.name, path, str(pointer["app_id"]))
    return AppFolder(path.name, path, None)


@dataclass
class Attribution:
    apps: list[tuple[AppFolder, str]]
    """Folders of the run, each with the reason."""
    unattributed: list[tuple[AppFolder, str]]
    """Folders whose chain ends somewhere unreadable, each with the reason."""


def attribute(burr_dir: Path, run_stem: str) -> Attribution:
    """Sort the tracker's folders into this run's and the ones that cannot be placed.

    A folder is the run's when its id is one the CLI gives the run, or its metadata chain
    reaches such an id. It cannot be placed when its chain reaches a folder that is missing
    or has no readable metadata. Folders that reach another run are left out.
    """
    result = Attribution(apps=[], unattributed=[])
    if not burr_dir.is_dir():
        return result
    folders = {p.name: _read_folder(p) for p in sorted(burr_dir.iterdir()) if p.is_dir()}
    named = named_app_pattern(run_stem)

    def root_of(folder: AppFolder) -> tuple[str | None, str]:
        """(the named root it reaches, or None; how)."""
        seen: set[str] = set()
        current = folder
        while True:
            if named.match(current.app_id):
                return current.app_id, ""
            if current.error is not None:
                return None, f"{current.app_id}: {current.error}"
            if current.parent is None:
                return None, "root"
            if current.app_id in seen:
                return None, f"cycle at {current.app_id}"
            seen.add(current.app_id)
            parent = folders.get(current.parent)
            if parent is None:
                return None, f"names {current.parent}, which has no folder"
            current = parent

    for app_id, folder in folders.items():
        root, how = root_of(folder)
        if root == app_id:
            result.apps.append((folder, "named for this run"))
        elif root is not None:
            result.apps.append((folder, f"spawned under {root} (metadata.json)"))
        elif how != "root":
            result.unattributed.append((folder, how))
    return result


def persister_app_ids(db_path: Path) -> list[str]:
    """Every application id with rows in a persister file; empty when there is no table."""
    if not db_path.is_file():
        return []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            rows = conn.execute(f"SELECT DISTINCT app_id FROM {PERSISTER_TABLE}").fetchall()
        except sqlite3.OperationalError:
            return []
    return sorted(str(r[0]) for r in rows)


def persister_rows(db_path: Path, app_ids: list[str]) -> int:
    """Rows of ``app_ids`` in a persister file; 0 when the file or table is missing."""
    if not app_ids or not db_path.is_file():
        return 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        try:
            marks = ",".join("?" * len(app_ids))
            row = conn.execute(
                f"SELECT COUNT(*) FROM {PERSISTER_TABLE} WHERE app_id IN ({marks})",
                app_ids,
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0])


def delete_persister_rows(db_path: Path, app_ids: list[str]) -> int:
    if not app_ids or not db_path.is_file():
        return 0
    conn = sqlite3.connect(db_path)
    try:
        marks = ",".join("?" * len(app_ids))
        try:
            cursor = conn.execute(
                f"DELETE FROM {PERSISTER_TABLE} WHERE app_id IN ({marks})",
                app_ids,
            )
        except sqlite3.OperationalError:
            return 0
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
