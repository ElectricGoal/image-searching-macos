from __future__ import annotations

from pathlib import Path

from imgsearch.config import INDEX_DIR_NAME
from imgsearch.core.scanner import is_image, scan, sha1_of_file


def test_scan_finds_supported_formats(tmp_image_dir: Path) -> None:
    files = list(scan(tmp_image_dir))
    rel = sorted(f.rel_path for f in files)
    assert rel == sorted(["red.png", "green.png", "blue.png", "sub/gray.png"])


def test_scan_skips_index_dir(tmp_image_dir: Path) -> None:
    idx = tmp_image_dir / INDEX_DIR_NAME
    idx.mkdir()
    (idx / "pretend.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    rel = {f.rel_path for f in scan(tmp_image_dir)}
    assert INDEX_DIR_NAME not in " ".join(rel)


def test_non_recursive(tmp_image_dir: Path) -> None:
    files = list(scan(tmp_image_dir, recursive=False))
    rel = {f.rel_path for f in files}
    assert "sub/gray.png" not in rel
    assert "red.png" in rel


def test_is_image_filters() -> None:
    assert is_image(Path("a.JPG"))
    assert is_image(Path("a.heic"))
    assert not is_image(Path("a.txt"))
    assert not is_image(Path("a"))


def test_sha1_stable(tmp_image_dir: Path) -> None:
    p = tmp_image_dir / "red.png"
    h1 = sha1_of_file(p)
    h2 = sha1_of_file(p)
    assert h1 == h2
    assert len(h1) == 40
