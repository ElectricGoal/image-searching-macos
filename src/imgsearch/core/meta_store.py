"""SQLite-backed metadata store: paths, hashes, mtimes, id allocation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    faiss_id   INTEGER PRIMARY KEY,
    rel_path   TEXT UNIQUE NOT NULL,
    sha1       TEXT NOT NULL,
    mtime      REAL NOT NULL,
    width      INTEGER,
    height     INTEGER,
    indexed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_path ON images(rel_path);

CREATE TABLE IF NOT EXISTS id_seq (
    id       INTEGER PRIMARY KEY CHECK (id = 0),
    next_id  INTEGER NOT NULL
);
INSERT OR IGNORE INTO id_seq (id, next_id) VALUES (0, 1);
"""


@dataclass(frozen=True)
class ImageRow:
    faiss_id: int
    rel_path: str
    sha1: str
    mtime: float
    width: int | None
    height: int | None
    indexed_at: float


class MetaStore:
    """Thin wrapper around a single SQLite file.

    All writes happen inside the caller's transaction boundary via
    ``transaction()``. Reads outside a transaction are permitted.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ----- lifecycle -----

    def open(self) -> None:
        if self._conn is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> MetaStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the connection inside an exclusive transaction."""
        assert self._conn is not None, "call open() first"
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ----- id allocation -----

    def allocate_ids(self, count: int) -> list[int]:
        """Atomically reserve `count` new faiss_ids. Monotonic, never reused."""
        assert self._conn is not None
        if count <= 0:
            return []
        with self.transaction() as conn:
            row = conn.execute("SELECT next_id FROM id_seq WHERE id = 0").fetchone()
            start = int(row["next_id"])
            conn.execute(
                "UPDATE id_seq SET next_id = ? WHERE id = 0", (start + count,)
            )
        return list(range(start, start + count))

    # ----- reads -----

    def get_by_path(self, rel_path: str) -> ImageRow | None:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM images WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return _row_to_image(row) if row else None

    def count(self) -> int:
        assert self._conn is not None
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def all_rel_paths(self) -> set[str]:
        assert self._conn is not None
        return {
            r["rel_path"]
            for r in self._conn.execute("SELECT rel_path FROM images")
        }

    def fetch_paths_for_ids(self, ids: Iterable[int]) -> dict[int, str]:
        """Return a mapping of faiss_id -> rel_path for the given ids."""
        assert self._conn is not None
        ids = list(ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT faiss_id, rel_path FROM images WHERE faiss_id IN ({placeholders})",
            ids,
        ).fetchall()
        return {int(r["faiss_id"]): r["rel_path"] for r in rows}

    def all_ids(self) -> list[int]:
        assert self._conn is not None
        return [int(r["faiss_id"]) for r in self._conn.execute("SELECT faiss_id FROM images")]

    # ----- writes (must be inside transaction()) -----

    def upsert(
        self,
        conn: sqlite3.Connection,
        *,
        faiss_id: int,
        rel_path: str,
        sha1: str,
        mtime: float,
        width: int | None,
        height: int | None,
        indexed_at: float,
    ) -> None:
        conn.execute(
            """
            INSERT INTO images (faiss_id, rel_path, sha1, mtime, width, height, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rel_path) DO UPDATE SET
                faiss_id = excluded.faiss_id,
                sha1     = excluded.sha1,
                mtime    = excluded.mtime,
                width    = excluded.width,
                height   = excluded.height,
                indexed_at = excluded.indexed_at
            """,
            (faiss_id, rel_path, sha1, mtime, width, height, indexed_at),
        )

    def update_mtime(
        self, conn: sqlite3.Connection, rel_path: str, mtime: float
    ) -> None:
        conn.execute(
            "UPDATE images SET mtime = ? WHERE rel_path = ?", (mtime, rel_path)
        )

    def delete_by_ids(self, conn: sqlite3.Connection, ids: Iterable[int]) -> None:
        ids = list(ids)
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM images WHERE faiss_id IN ({placeholders})", ids)


def _row_to_image(row: sqlite3.Row) -> ImageRow:
    return ImageRow(
        faiss_id=int(row["faiss_id"]),
        rel_path=row["rel_path"],
        sha1=row["sha1"],
        mtime=float(row["mtime"]),
        width=row["width"],
        height=row["height"],
        indexed_at=float(row["indexed_at"]),
    )
