from __future__ import annotations

from pathlib import Path

from imgsearch.core.meta_store import MetaStore


def test_open_creates_schema(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "meta.db")
    store.open()
    assert store.count() == 0
    store.close()


def test_id_allocation_monotonic(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "meta.db")
    store.open()
    a = store.allocate_ids(3)
    b = store.allocate_ids(2)
    assert a == [1, 2, 3]
    assert b == [4, 5]
    store.close()


def test_upsert_and_lookup(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "meta.db")
    store.open()
    with store.transaction() as conn:
        store.upsert(
            conn,
            faiss_id=10,
            rel_path="a/b.png",
            sha1="deadbeef",
            mtime=1.0,
            width=10,
            height=20,
            indexed_at=2.0,
        )
    row = store.get_by_path("a/b.png")
    assert row is not None
    assert row.faiss_id == 10
    assert row.sha1 == "deadbeef"
    assert store.count() == 1
    assert store.all_rel_paths() == {"a/b.png"}
    store.close()


def test_delete_and_fetch_map(tmp_path: Path) -> None:
    store = MetaStore(tmp_path / "meta.db")
    store.open()
    with store.transaction() as conn:
        for fid, name in [(1, "x.png"), (2, "y.png"), (3, "z.png")]:
            store.upsert(
                conn,
                faiss_id=fid,
                rel_path=name,
                sha1="h",
                mtime=1.0,
                width=1,
                height=1,
                indexed_at=1.0,
            )
    mapping = store.fetch_paths_for_ids([1, 3])
    assert mapping == {1: "x.png", 3: "z.png"}

    with store.transaction() as conn:
        store.delete_by_ids(conn, [2])
    assert store.count() == 2
    assert store.get_by_path("y.png") is None
    store.close()
