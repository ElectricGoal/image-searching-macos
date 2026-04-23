from __future__ import annotations

import numpy as np

from imgsearch.config import resolve_model
from imgsearch.core.embedder import (
    MLXEmbedder,
    TorchEmbedder,
    _coerce_feature_output,
    create_embedder,
)


def test_create_embedder_selects_torch_backend() -> None:
    spec = resolve_model("clip-vit-b32", backend="torch")
    embedder = create_embedder(spec)
    assert isinstance(embedder, TorchEmbedder)


def test_create_embedder_selects_mlx_backend() -> None:
    spec = resolve_model("clip-vit-b32", backend="mlx")
    embedder = create_embedder(spec)
    assert isinstance(embedder, MLXEmbedder)


class _FakeModelOutput:
    def __init__(self, array: np.ndarray) -> None:
        self.pooler_output = array


def test_coerce_feature_output_unwraps_model_output() -> None:
    arr = np.ones((1, 512), dtype=np.float32)
    coerced = _coerce_feature_output(_FakeModelOutput(arr), kind="image")
    assert coerced is arr
