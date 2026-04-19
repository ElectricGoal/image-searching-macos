"""FAISS-backed vector store: add, remove, search, persist."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class VectorStore:
    """Wraps a FAISS `IndexIDMap2(IndexFlatIP)` for exact cosine search.

    Vectors must be L2-normalized by the caller; inner product then equals
    cosine similarity. IDs are 64-bit integers allocated by MetaStore.
    """

    def __init__(self, index_path: Path, dim: int) -> None:
        self._index_path = index_path
        self._dim = dim
        self._index = None  # type: ignore[assignment]
        self._faiss = None

    # ----- lifecycle -----

    def _import_faiss(self):
        if self._faiss is None:
            import faiss  # type: ignore[import-not-found]

            self._faiss = faiss
        return self._faiss

    def open(self) -> None:
        """Load existing index or create an empty one."""
        if self._index is not None:
            return
        faiss = self._import_faiss()
        if self._index_path.exists():
            index = faiss.read_index(str(self._index_path))
            if index.d != self._dim:
                raise RuntimeError(
                    f"FAISS index dim {index.d} does not match expected {self._dim}"
                )
            self._index = index
        else:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            inner = faiss.IndexFlatIP(self._dim)
            self._index = faiss.IndexIDMap2(inner)

    @property
    def count(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    # ----- mutations -----

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add vectors with explicit int64 ids. Vectors must be (N, dim) float32."""
        assert self._index is not None
        if len(ids) == 0:
            return
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        if vectors.shape != (len(ids), self._dim):
            raise ValueError(
                f"vectors shape {vectors.shape} does not match ({len(ids)}, {self._dim})"
            )
        id_array = np.asarray(ids, dtype=np.int64)
        self._index.add_with_ids(vectors, id_array)

    def remove(self, ids: np.ndarray | list[int]) -> int:
        """Remove vectors by id. Returns number actually removed."""
        assert self._index is not None
        faiss = self._import_faiss()
        id_array = np.asarray(list(ids), dtype=np.int64)
        if len(id_array) == 0:
            return 0
        selector = faiss.IDSelectorBatch(id_array.size, faiss.swig_ptr(id_array))
        return int(self._index.remove_ids(selector))

    # ----- search -----

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, ids), each shape (1, k). Empty index yields empty arrays."""
        assert self._index is not None
        if self._index.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        if query.ndim == 1:
            query = query[None, :]
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)
        k = min(k, int(self._index.ntotal))
        distances, ids = self._index.search(query, k)
        return distances, ids

    # ----- persistence -----

    def save(self) -> None:
        """Atomically persist the index to disk."""
        assert self._index is not None
        faiss = self._import_faiss()
        tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp))
        os.replace(tmp, self._index_path)

    def all_ids(self) -> list[int]:
        """Return all ids currently stored. Used for reconciliation."""
        assert self._index is not None
        n = int(self._index.ntotal)
        if n == 0:
            return []
        # IndexIDMap2 stores ids in `id_map`; expose via reconstruct hack:
        # easier: iterate via id_map vector exposed by faiss python bindings.
        try:
            return [int(self._index.id_map.at(i)) for i in range(n)]
        except Exception:
            # Fallback: not all wrappers expose .at(); return empty for safety.
            return []
