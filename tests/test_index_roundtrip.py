"""End-to-end test of the Index facade using a deterministic fake embedder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # skip if faiss not installed yet

from imgsearch.config import ModelSpec  # noqa: E402
from imgsearch.core.index import Index  # noqa: E402
from imgsearch.core.scanner import scan, sha1_of_file  # noqa: E402


def _embed_all(embedder, files):
    paths = [f.abs_path for f in files]
    vecs = embedder.encode_images(paths)
    sha1s = [sha1_of_file(p) for p in paths]
    dims = [(1, 1)] * len(files)
    return vecs, sha1s, dims


def test_round_trip_and_search(tmp_image_dir: Path, fake_spec: ModelSpec, fake_embedder) -> None:
    with Index(tmp_image_dir, fake_spec, "fake") as idx:
        idx.open(create=True)
        files = list(scan(tmp_image_dir))
        plan = idx.plan(files)
        assert len(plan.to_embed) == len(files)

        vecs, sha1s, dims = _embed_all(fake_embedder, plan.to_embed)
        idx.add_batch(plan.to_embed, vecs, sha1s, dims)
        idx.commit()

        assert idx.count == len(files)

        # Querying with the exact same token used for embedding should rank first.
        q_red = fake_embedder.encode_text("red")
        hits = idx.search(q_red, k=len(files))
        assert hits[0].rel_path == "red.png"
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_rescan_unchanged(tmp_image_dir: Path, fake_spec: ModelSpec, fake_embedder) -> None:
    with Index(tmp_image_dir, fake_spec, "fake") as idx:
        idx.open(create=True)
        files = list(scan(tmp_image_dir))
        plan = idx.plan(files)
        vecs, sha1s, dims = _embed_all(fake_embedder, plan.to_embed)
        idx.add_batch(plan.to_embed, vecs, sha1s, dims)
        idx.commit()

    with Index(tmp_image_dir, fake_spec, "fake") as idx:
        idx.open(create=True)
        files = list(scan(tmp_image_dir))
        plan2 = idx.plan(files)
        assert plan2.to_embed == []
        assert plan2.to_delete_ids == []
        assert plan2.unchanged == len(files)


def test_delete_missing(tmp_image_dir: Path, fake_spec: ModelSpec, fake_embedder) -> None:
    with Index(tmp_image_dir, fake_spec, "fake") as idx:
        idx.open(create=True)
        files = list(scan(tmp_image_dir))
        vecs, sha1s, dims = _embed_all(fake_embedder, files)
        idx.add_batch(files, vecs, sha1s, dims)
        idx.commit()

    (tmp_image_dir / "red.png").unlink()

    with Index(tmp_image_dir, fake_spec, "fake") as idx:
        idx.open(create=True)
        files = list(scan(tmp_image_dir))
        plan = idx.plan(files)
        assert len(plan.to_delete_ids) == 1
        idx.apply_deletes(plan.to_delete_ids)
        idx.commit()
        assert idx.count == 3


def test_model_mismatch_refused(tmp_image_dir: Path, fake_spec: ModelSpec, fake_embedder) -> None:
    with Index(tmp_image_dir, fake_spec, "fake") as idx:
        idx.open(create=True)
        files = list(scan(tmp_image_dir))
        vecs, sha1s, dims = _embed_all(fake_embedder, files)
        idx.add_batch(files, vecs, sha1s, dims)
        idx.commit()

    other = ModelSpec(
        id="different/model", dim=8, image_size=16, family="test", display_name="Other"
    )
    with pytest.raises(Exception) as excinfo:
        with Index(tmp_image_dir, other, "other") as idx:
            idx.open(create=True)
    assert "model" in str(excinfo.value).lower()
