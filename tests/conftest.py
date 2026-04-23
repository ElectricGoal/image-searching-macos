"""Shared pytest fixtures: synthetic images and a fake embedder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imgsearch.config import ModelSpec
from imgsearch.core.embedder import EmbedderProtocol


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a folder with a few tiny colored PNGs."""
    colors = {
        "red.png": (255, 0, 0),
        "green.png": (0, 255, 0),
        "blue.png": (0, 0, 255),
    }
    for name, color in colors.items():
        img = Image.new("RGB", (32, 32), color)
        img.save(tmp_path / name)
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (32, 32), (128, 128, 128)).save(tmp_path / "sub" / "gray.png")
    return tmp_path


@pytest.fixture
def fake_spec() -> ModelSpec:
    return ModelSpec(
        alias="fake",
        id="test/fake-model",
        dim=8,
        image_size=16,
        family="test",
        display_name="Fake Model",
        backend="test",
    )


class FakeEmbedder:
    """Deterministic pseudo-embedder used in tests.

    Maps a filename or string to a stable 8-dim unit vector so that equality
    of identifier implies cosine similarity of 1.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def _vec_for_token(self, token: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(token)) % (2**32))
        v = rng.standard_normal(self.spec.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def load(self) -> None:  # pragma: no cover - noop
        pass

    def encode_images(self, paths) -> np.ndarray:
        return np.stack([self._vec_for_token(Path(p).stem) for p in paths])

    def encode_pil(self, images) -> np.ndarray:
        # Derive a stable token from the image content size as a proxy for identity.
        return np.stack([self._vec_for_token(str(img.size)) for img in images])

    def encode_text(self, text: str) -> np.ndarray:
        return self._vec_for_token(text.strip().lower())


@pytest.fixture
def fake_embedder(fake_spec: ModelSpec) -> EmbedderProtocol:
    return FakeEmbedder(fake_spec)  # type: ignore[return-value]
