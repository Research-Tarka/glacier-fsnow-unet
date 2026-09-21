"""SQLite cache of downloaded scenes, used as the resume mechanism.

Purpose
-------
Record which ``(glacier, sensor, year, scene_id)`` tuples have already been
downloaded, so a re-run skips them. This is the resume mechanism for stage 4
(see ``docs/decisions/scene_download_resume.md``).

Inputs / outputs
----------------
One SQLite file per split, e.g. ``scene_cache_split7.db``.

Design
------
* SQLite in WAL mode -- process-safe across the N parallel splits (each split
  has its own file) and crash-safe mid-download.
* A ``threading.Lock`` for thread-safety within a split.
* An in-memory layer loaded in bulk per ``(glacier, sensor)`` prefix, so the
  per-scene ``is_done`` lookups that dominate a resume run are O(1) in RAM
  after one indexed range scan.

Keys
----
* scene:        ``"<glims_id>|<sensor>|<year>|<scene_id>"``
* sensor done:  ``"__SENSOR_DONE__|<glims_id>|<sensor>"``
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Optional

STATUS_OK = "OK"
STATUS_FAILED = "Failed"
STATUS_SENSOR_DONE = "DoneSensor"


class SceneCache:
    """Persistent, thread- and process-safe record of downloaded scenes."""

    def __init__(self, cache_path: str | Path) -> None:
        self.path = Path(cache_path).with_suffix(".db")
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        self._memory: dict[str, str] = {}
        self._loaded_prefixes: set[str] = set()
        self._init_db()

    # -- connection -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.path, check_same_thread=False, timeout=30
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
        return self._connection

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            con = self._connect()
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key    TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    reason TEXT
                )
                """
            )
            con.commit()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> "SceneCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- keys -----------------------------------------------------------

    @staticmethod
    def scene_key(glims_id: str, sensor: str, year: int, scene_id: str) -> str:
        return f"{glims_id}|{sensor}|{year}|{scene_id}"

    @staticmethod
    def sensor_done_key(glims_id: str, sensor: str) -> str:
        return f"__SENSOR_DONE__|{glims_id}|{sensor}"

    # -- bulk load ------------------------------------------------------

    def _ensure_loaded(self, glims_id: str, sensor: str) -> None:
        """Load every entry for one (glacier, sensor) into RAM, once."""
        prefix = f"{glims_id}|{sensor}"
        if prefix in self._loaded_prefixes:
            return
        with self._lock:
            if prefix in self._loaded_prefixes:
                return
            con = self._connect()
            # Range scan over the PRIMARY KEY B-tree: '~' sorts just after '|'.
            rows = con.execute(
                "SELECT key, status FROM cache WHERE key >= ? AND key < ?",
                (f"{prefix}|", f"{prefix}~"),
            ).fetchall()
            for key, status in rows:
                self._memory[key] = status

            done_key = self.sensor_done_key(glims_id, sensor)
            row = con.execute(
                "SELECT status FROM cache WHERE key = ?", (done_key,)
            ).fetchone()
            if row:
                self._memory[done_key] = row[0]
            self._loaded_prefixes.add(prefix)

    # -- reads ----------------------------------------------------------

    def get(self, glims_id: str, sensor: str, year: int, scene_id: str) -> Optional[str]:
        """Return the recorded status of one scene, or None if unknown."""
        self._ensure_loaded(glims_id, sensor)
        return self._memory.get(self.scene_key(glims_id, sensor, year, scene_id))

    def is_done(self, glims_id: str, sensor: str, year: int, scene_id: str) -> bool:
        """True when this scene was already downloaded successfully."""
        return self.get(glims_id, sensor, year, scene_id) == STATUS_OK

    def is_attempted(self, glims_id: str, sensor: str, year: int, scene_id: str) -> bool:
        """True when this scene was downloaded *or* recorded as a failure."""
        return self.get(glims_id, sensor, year, scene_id) is not None

    def is_sensor_done(self, glims_id: str, sensor: str) -> bool:
        """True when this glacier/sensor pair was fully processed."""
        self._ensure_loaded(glims_id, sensor)
        return self._memory.get(self.sensor_done_key(glims_id, sensor)) == STATUS_SENSOR_DONE

    def count(self, status: Optional[str] = None) -> int:
        """Total number of cached entries, optionally filtered by status."""
        with self._lock:
            con = self._connect()
            if status is None:
                return int(con.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
            return int(
                con.execute(
                    "SELECT COUNT(*) FROM cache WHERE status = ?", (status,)
                ).fetchone()[0]
            )

    # -- writes ---------------------------------------------------------

    def _write(self, key: str, status: str, reason: Optional[str] = None) -> None:
        with self._lock:
            con = self._connect()
            con.execute(
                "INSERT OR REPLACE INTO cache (key, status, reason) VALUES (?, ?, ?)",
                (key, status, reason),
            )
            con.commit()
            self._memory[key] = status

    def set_ok(self, glims_id: str, sensor: str, year: int, scene_id: str) -> None:
        self._write(self.scene_key(glims_id, sensor, year, scene_id), STATUS_OK)

    def set_failed(
        self, glims_id: str, sensor: str, year: int, scene_id: str, reason: str
    ) -> None:
        self._write(
            self.scene_key(glims_id, sensor, year, scene_id), STATUS_FAILED, reason
        )

    def set_sensor_done(self, glims_id: str, sensor: str) -> None:
        self._write(self.sensor_done_key(glims_id, sensor), STATUS_SENSOR_DONE)

    def set_many_ok(self, entries: Iterable[tuple[str, str, int, str]]) -> int:
        """Record many successful scenes in one transaction."""
        rows = [
            (self.scene_key(gid, sensor, year, scene), STATUS_OK, None)
            for gid, sensor, year, scene in entries
        ]
        if not rows:
            return 0
        with self._lock:
            con = self._connect()
            con.executemany(
                "INSERT OR REPLACE INTO cache (key, status, reason) VALUES (?, ?, ?)",
                rows,
            )
            con.commit()
            for key, status, _ in rows:
                self._memory[key] = status
        return len(rows)
