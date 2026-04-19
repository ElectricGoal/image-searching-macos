"""High-level Index facade coordinating MetaStore + VectorStore + Manifest."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imgsearch.config import (
    FAISS_FILE,
    INDEX_DIR_NAME,
    MANIFEST_FILE,
    META_DB_FILE,
    ModelSpec,
)
from imgsearch.core.manifest import Manifest, ensure_compatible
from imgsearch.core.meta_store import MetaStore
from imgsearch.core.scanner import DiscoveredFile, sha1_of_file
from imgsearch.core.vector_store import VectorStore


@dataclass
class IndexPaths:
    root: Path
    index_dir: Path
    manifest: Path
    meta_db: Path
    faiss: Path

    @classmethod
    def for_root(cls, root: Path) -> IndexPaths:
        d = root / INDEX_DIR_NAME
        return cls(
            root=root,
            index_dir=d,
            manifest=d / MANIFEST_FILE,
            meta_db=d / META_DB_FILE,
            faiss=d / FAISS_FILE,
        )


@dataclass(frozen=True)
class SearchHit:
    rel_path: str
    similarity: float


@dataclass
class Plan:
    """What index() needs to do for a scan snapshot."""

    to_embed: list[DiscoveredFile]  # new or changed
    to_delete_ids: list[int]  # stale rows (file gone)
    to_touch_mtime: list[tuple[str, float]]  # mtime changed but content same
    unchanged: int


class Index:
    """Façade used by CLI commands. Owns open MetaStore + VectorStore + Manifest."""

    def __init__(self, root: Path, spec: ModelSpec, alias: str) -> None:
        self.paths = IndexPaths.for_root(root.resolve())
        self.spec = spec
        self.alias = alias
        self._meta = MetaStore(self.paths.meta_db)
        self._vectors = VectorStore(self.paths.faiss, spec.dim)
        self._manifest: Manifest | None = None

    # ----- lifecycle -----

    def open(self, *, create: bool = False) -> None:
        """Open (and optionally create) the index directory."""
        if self.paths.manifest.exists():
            self._manifest = Manifest.load(self.paths.manifest)
            ensure_compatible(self._manifest, self.spec)
        else:
            if not create:
                raise FileNotFoundError(
                    f"No index found at {self.paths.index_dir}. "
                    f"Run `imgsearch index {self.paths.root}` first."
                )
            self.paths.index_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = Manifest.from_model(self.spec, self.alias)

        self._meta.open()
        self._vectors.open()
        self._reconcile()

    def close(self) -> None:
        self._meta.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ----- reconciliation -----

    def _reconcile(self) -> None:
        """Cross-check SQLite vs FAISS. Drop orphan SQLite rows whose ids are
        no longer in FAISS (can happen if we crashed after SQLite commit but
        before FAISS write)."""
        meta_ids = set(self._meta.all_ids())
        vec_ids = set(self._vectors.all_ids())
        if not meta_ids:
            return
        orphans = meta_ids - vec_ids
        if not orphans:
            return
        # Drop orphan metadata rows — they'll be re-embedded on next index.
        with self._meta.transaction() as conn:
            self._meta.delete_by_ids(conn, orphans)

    # ----- planning -----

    def plan(self, discovered: list[DiscoveredFile]) -> Plan:
        """Compute what to do for a fresh scan. Uses mtime fast-path, sha1 slow-path."""
        existing_paths = self._meta.all_rel_paths()
        seen_paths: set[str] = set()

        to_embed: list[DiscoveredFile] = []
        to_touch: list[tuple[str, float]] = []
        unchanged = 0

        for f in discovered:
            seen_paths.add(f.rel_path)
            row = self._meta.get_by_path(f.rel_path)
            if row is None:
                to_embed.append(f)
                continue
            if row.mtime == f.mtime:
                unchanged += 1
                continue
            # mtime changed — hash to decide
            new_sha = sha1_of_file(f.abs_path)
            if new_sha == row.sha1:
                to_touch.append((f.rel_path, f.mtime))
                unchanged += 1
            else:
                to_embed.append(f)

        missing = existing_paths - seen_paths
        to_delete_ids: list[int] = []
        if missing:
            for p in missing:
                row = self._meta.get_by_path(p)
                if row is not None:
                    to_delete_ids.append(row.faiss_id)

        return Plan(
            to_embed=to_embed,
            to_delete_ids=to_delete_ids,
            to_touch_mtime=to_touch,
            unchanged=unchanged,
        )

    # ----- mutations -----

    def apply_deletes(self, ids: list[int]) -> None:
        if not ids:
            return
        self._vectors.remove(ids)
        with self._meta.transaction() as conn:
            self._meta.delete_by_ids(conn, ids)

    def apply_mtime_touches(self, touches: list[tuple[str, float]]) -> None:
        if not touches:
            return
        with self._meta.transaction() as conn:
            for rel_path, mtime in touches:
                self._meta.update_mtime(conn, rel_path, mtime)

    def add_batch(
        self,
        files: list[DiscoveredFile],
        vectors: np.ndarray,
        sha1s: list[str],
        dims: list[tuple[int | None, int | None]],
    ) -> None:
        """Append embeddings for a batch of new/changed files."""
        if not files:
            return
        assert vectors.shape[0] == len(files), "vectors/files length mismatch"
        assert len(sha1s) == len(files)
        assert len(dims) == len(files)

        ids = self._meta.allocate_ids(len(files))
        now = time.time()

        # Write SQLite first — if FAISS save later fails, reconciliation on
        # next open will drop the orphan rows.
        with self._meta.transaction() as conn:
            for faiss_id, f, sha, (w, h) in zip(ids, files, sha1s, dims, strict=True):
                self._meta.upsert(
                    conn,
                    faiss_id=faiss_id,
                    rel_path=f.rel_path,
                    sha1=sha,
                    mtime=f.mtime,
                    width=w,
                    height=h,
                    indexed_at=now,
                )

        self._vectors.add(np.asarray(ids, dtype=np.int64), vectors)

    def commit(self) -> None:
        """Persist FAISS to disk and update manifest."""
        assert self._manifest is not None
        self._vectors.save()
        self._manifest.touch(count=self._meta.count())
        self._manifest.save(self.paths.manifest)

    # ----- queries -----

    def search(self, query_vec: np.ndarray, k: int) -> list[SearchHit]:
        distances, ids = self._vectors.search(query_vec, k)
        if ids.size == 0:
            return []
        id_list = [int(i) for i in ids[0] if i != -1]
        path_map = self._meta.fetch_paths_for_ids(id_list)
        hits: list[SearchHit] = []
        for sim, fid in zip(distances[0], ids[0], strict=True):
            fid_int = int(fid)
            if fid_int == -1:
                continue
            rel = path_map.get(fid_int)
            if rel is None:
                continue  # orphan — reconciliation will clean up on next open
            hits.append(SearchHit(rel_path=rel, similarity=float(sim)))
        return hits

    @property
    def manifest(self) -> Manifest:
        assert self._manifest is not None
        return self._manifest

    @property
    def count(self) -> int:
        return self._meta.count()
