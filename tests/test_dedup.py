"""Tests for duplicate_finder: exact groups, near groups, keeper selection, merging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from imgsearch.core.duplicate_finder import (  # noqa: E402
    _UnionFind,
    build_groups,
    find_exact_groups,
    find_near_groups,
    pick_keeper,
)
from imgsearch.core.meta_store import MetaStore  # noqa: E402
from imgsearch.core.vector_store import VectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(tmp_path: Path, rows: list[dict]) -> MetaStore:
    store = MetaStore(tmp_path / "meta.db")
    store.open()
    with store.transaction() as conn:
        for r in rows:
            store.upsert(
                conn,
                faiss_id=r["faiss_id"],
                rel_path=r["rel_path"],
                sha1=r.get("sha1", "aabbcc"),
                mtime=r.get("mtime", 1.0),
                width=r.get("width"),
                height=r.get("height"),
                indexed_at=1.0,
            )
    return store


def _make_vec_store(tmp_path: Path, vecs: np.ndarray, ids: list[int]) -> VectorStore:
    vs = VectorStore(tmp_path / "index.faiss", vecs.shape[1])
    vs.open()
    vs.add(np.array(ids, dtype=np.int64), vecs.astype(np.float32))
    return vs


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

def test_union_find_basic() -> None:
    uf = _UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    groups = uf.groups()
    assert len(groups) == 1
    members = list(groups.values())[0]
    assert set(members) == {"a", "b", "c"}


def test_union_find_no_singleton() -> None:
    uf = _UnionFind()
    uf.union("x", "x")
    assert uf.groups() == {}


# ---------------------------------------------------------------------------
# Exact duplicates
# ---------------------------------------------------------------------------

def test_find_exact_groups(tmp_path: Path) -> None:
    meta = _make_meta(tmp_path, [
        {"faiss_id": 1, "rel_path": "a.png", "sha1": "same"},
        {"faiss_id": 2, "rel_path": "b.png", "sha1": "same"},
        {"faiss_id": 3, "rel_path": "c.png", "sha1": "different"},
    ])
    groups = find_exact_groups(meta)
    assert len(groups) == 1
    assert set(groups[0]) == {"a.png", "b.png"}
    meta.close()


def test_find_exact_groups_no_dups(tmp_path: Path) -> None:
    meta = _make_meta(tmp_path, [
        {"faiss_id": 1, "rel_path": "a.png", "sha1": "h1"},
        {"faiss_id": 2, "rel_path": "b.png", "sha1": "h2"},
    ])
    assert find_exact_groups(meta) == []
    meta.close()


# ---------------------------------------------------------------------------
# Near duplicates
# ---------------------------------------------------------------------------

def test_find_near_groups_above_threshold(tmp_path: Path) -> None:
    base = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
    almost = _unit(np.array([1.0, 0.01, 0.0, 0.0]))   # cosine ~0.9999
    unrelated = _unit(np.array([0.0, 1.0, 0.0, 0.0]))  # cosine 0.0

    vecs = np.stack([base, almost, unrelated])
    meta = _make_meta(tmp_path, [
        {"faiss_id": 1, "rel_path": "base.png"},
        {"faiss_id": 2, "rel_path": "almost.png"},
        {"faiss_id": 3, "rel_path": "unrelated.png"},
    ])
    vs = _make_vec_store(tmp_path, vecs, [1, 2, 3])

    groups = find_near_groups(vs, meta, threshold=0.99)
    assert len(groups) == 1
    paths, sim = groups[0]
    assert set(paths) == {"base.png", "almost.png"}
    assert sim > 0.99
    meta.close()


def test_find_near_groups_below_threshold(tmp_path: Path) -> None:
    v1 = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
    v2 = _unit(np.array([0.0, 1.0, 0.0, 0.0]))
    vecs = np.stack([v1, v2])
    meta = _make_meta(tmp_path, [
        {"faiss_id": 1, "rel_path": "a.png"},
        {"faiss_id": 2, "rel_path": "b.png"},
    ])
    vs = _make_vec_store(tmp_path, vecs, [1, 2])
    assert find_near_groups(vs, meta, threshold=0.99) == []
    meta.close()


# ---------------------------------------------------------------------------
# Keeper selection
# ---------------------------------------------------------------------------

def test_pick_keeper_largest(tmp_path: Path) -> None:
    big = tmp_path / "big.png"
    small = tmp_path / "small.png"
    big.write_bytes(b"x" * 1000)
    small.write_bytes(b"x" * 10)
    keeper = pick_keeper(["big.png", "small.png"], tmp_path, "largest")
    assert keeper == "big.png"


def test_pick_keeper_newest(tmp_path: Path) -> None:
    import time
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    old.write_bytes(b"x")
    time.sleep(0.05)
    new.write_bytes(b"x")
    keeper = pick_keeper(["old.png", "new.png"], tmp_path, "newest")
    assert keeper == "new.png"


def test_pick_keeper_oldest(tmp_path: Path) -> None:
    import time
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    old.write_bytes(b"x")
    time.sleep(0.05)
    new.write_bytes(b"x")
    keeper = pick_keeper(["old.png", "new.png"], tmp_path, "oldest")
    assert keeper == "old.png"


# ---------------------------------------------------------------------------
# build_groups integration
# ---------------------------------------------------------------------------

def test_build_groups_merges_exact_and_near(tmp_path: Path) -> None:
    # exact group: a.png == b.png (same sha1)
    # near group: b.png ~ c.png (high cosine)
    # result should be one merged group {a, b, c}
    exact = [["a.png", "b.png"]]
    near = [(["b.png", "c.png"], 0.995)]

    for name in ["a.png", "b.png", "c.png"]:
        (tmp_path / name).write_bytes(b"x" * 100)

    groups = build_groups(tmp_path, exact, near, strategy="largest")
    assert len(groups) == 1
    assert set(groups[0].all_paths) == {"a.png", "b.png", "c.png"}
