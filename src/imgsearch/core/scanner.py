"""Recursive image discovery with content hashing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from imgsearch.config import INDEX_DIR_NAME, SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class DiscoveredFile:
    """A candidate image found on disk."""

    abs_path: Path
    rel_path: str  # POSIX-style relative to scan root
    mtime: float
    size: int


def is_image(path: Path) -> bool:
    """Return True if the path has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def scan(root: Path, recursive: bool = True) -> Iterator[DiscoveredFile]:
    """Walk `root` yielding image files. Skips the hidden index directory.

    The scanner is read-only and does not open files. Callers decide when to
    hash or embed based on mtime staleness.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        # Skip the index dir entirely
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if parts and parts[0] == INDEX_DIR_NAME:
            continue
        if not path.is_file():
            continue
        if not is_image(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            # File disappeared between listing and stat — skip silently
            continue
        yield DiscoveredFile(
            abs_path=path,
            rel_path=path.relative_to(root).as_posix(),
            mtime=stat.st_mtime,
            size=stat.st_size,
        )


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA1 digest of a file. Used only when mtime indicates a change."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
