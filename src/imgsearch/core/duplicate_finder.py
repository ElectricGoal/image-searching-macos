"""Duplicate image detection using SHA-1 exact matching and embedding cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imgsearch.core.meta_store import MetaStore
from imgsearch.core.vector_store import VectorStore


@dataclass
class DuplicateGroup:
    kind: str  # "exact" | "similar"
    similarity: float  # 1.0 for exact, cosine score for similar
    keeper: str  # rel_path of the image to keep
    duplicates: list[str]  # rel_paths to remove

    @property
    def all_paths(self) -> list[str]:
        return [self.keeper] + self.duplicates


# ---------------------------------------------------------------------------
# Union-Find (path-compressed, union-by-rank) for clustering pairs
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return {r: members for r, members in result.items() if len(members) >= 2}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def find_exact_groups(meta: MetaStore) -> list[list[str]]:
    """Return groups of rel_paths that share the same SHA-1 hash."""
    assert meta._conn is not None
    rows = meta._conn.execute("SELECT sha1, rel_path FROM images ORDER BY sha1").fetchall()
    by_hash: dict[str, list[str]] = {}
    for row in rows:
        by_hash.setdefault(row["sha1"], []).append(row["rel_path"])
    return [paths for paths in by_hash.values() if len(paths) >= 2]


def find_near_groups(
    vector_store: VectorStore,
    meta: MetaStore,
    threshold: float,
) -> list[tuple[list[str], float]]:
    """Return (group_paths, max_similarity) for near-duplicate clusters.

    Loads all stored vectors, computes the full cosine matrix (already L2-normalised
    so dot-product == cosine), and clusters pairs above `threshold` via Union-Find.
    """
    n = vector_store.count
    if n < 2:
        return []

    if n > 50_000:
        raise ValueError(
            f"Near-duplicate detection on {n} images would require a "
            f"{n*n*4/1e9:.1f} GB matrix. Use --exact-only for large collections."
        )

    # Reconstruct all vectors in FAISS-internal order.
    import faiss  # type: ignore[import-not-found]

    index = vector_store._index
    dim = vector_store._dim
    all_vecs = np.empty((n, dim), dtype=np.float32)
    # IndexIDMap2 wraps an inner flat index — reconstruct via inner index.
    inner = faiss.downcast_index(index.index)
    inner.reconstruct_n(0, n, all_vecs)

    # All vectors are L2-normalised → dot == cosine similarity.
    scores: np.ndarray = all_vecs @ all_vecs.T  # (n, n)

    # Retrieve ordered list of ids so we can map matrix row → rel_path.
    id_list = [int(index.id_map.at(i)) for i in range(n)]
    path_map = meta.fetch_paths_for_ids(id_list)

    uf = _UnionFind()
    group_score: dict[tuple[str, str], float] = {}

    upper = np.triu(scores, k=1)
    rows_idx, cols_idx = np.where(upper >= threshold)

    for r, c in zip(rows_idx, cols_idx):
        pa = path_map.get(id_list[int(r)])
        pb = path_map.get(id_list[int(c)])
        if pa is None or pb is None:
            continue
        uf.union(pa, pb)
        key = (min(pa, pb), max(pa, pb))
        group_score[key] = float(upper[r, c])

    result: list[tuple[list[str], float]] = []
    for members in uf.groups().values():
        max_sim = max(
            group_score.get((min(a, b), max(a, b)), threshold)
            for a in members
            for b in members
            if a != b
        )
        result.append((sorted(members), max_sim))
    return result


# ---------------------------------------------------------------------------
# Keeper selection
# ---------------------------------------------------------------------------

def pick_keeper(paths: list[str], root: Path, strategy: str) -> str:
    """Select which rel_path to keep from a duplicate group."""

    def _stat(rel: str):
        try:
            return (root / rel).stat()
        except OSError:
            return None

    if strategy == "largest":
        return max(paths, key=lambda p: (_stat(p) or type("", (), {"st_size": 0})()).st_size)
    if strategy == "newest":
        return max(paths, key=lambda p: (_stat(p) or type("", (), {"st_mtime": 0.0})()).st_mtime)
    if strategy == "oldest":
        return min(paths, key=lambda p: (_stat(p) or type("", (), {"st_mtime": float("inf")})()).st_mtime)
    if strategy == "highest-res":
        from PIL import Image
        def _res(rel: str) -> int:
            try:
                with Image.open(root / rel) as img:
                    return img.width * img.height
            except Exception:
                return 0
        return max(paths, key=_res)
    raise ValueError(f"Unknown keep strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Merge + build final groups
# ---------------------------------------------------------------------------

def build_groups(
    root: Path,
    exact_raw: list[list[str]],
    near_raw: list[tuple[list[str], float]],
    strategy: str,
) -> list[DuplicateGroup]:
    """Merge exact and near groups, deduplicate overlapping members, assign keeper."""
    uf = _UnionFind()
    sim_map: dict[str, float] = {}  # root → best similarity
    kind_map: dict[str, str] = {}   # root → "exact" | "similar"

    for group in exact_raw:
        for a, b in zip(group, group[1:]):
            uf.union(a, b)
        root_node = uf.find(group[0])
        sim_map[root_node] = 1.0
        kind_map[root_node] = "exact"

    for members, sim in near_raw:
        for a, b in zip(members, members[1:]):
            uf.union(a, b)
        root_node = uf.find(members[0])
        if sim_map.get(root_node, 0.0) < sim:
            sim_map[root_node] = sim
        kind_map.setdefault(root_node, "similar")

    result: list[DuplicateGroup] = []
    for root_node, members in uf.groups().items():
        keeper = pick_keeper(members, root, strategy)
        dups = [m for m in sorted(members) if m != keeper]
        result.append(DuplicateGroup(
            kind=kind_map.get(uf.find(root_node), "similar"),
            similarity=sim_map.get(uf.find(root_node), 1.0),
            keeper=keeper,
            duplicates=dups,
        ))

    result.sort(key=lambda g: (-g.similarity, g.keeper))
    return result
