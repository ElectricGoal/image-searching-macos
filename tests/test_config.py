from __future__ import annotations

import pytest

from imgsearch.config import (
    DEFAULT_MODEL_ALIAS,
    MODEL_REGISTRY,
    resolve_model,
)


def test_default_alias_registered() -> None:
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_by_alias() -> None:
    spec = resolve_model(DEFAULT_MODEL_ALIAS)
    assert spec.dim > 0


def test_resolve_by_id() -> None:
    spec = resolve_model(MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].id)
    assert spec.id == MODEL_REGISTRY[DEFAULT_MODEL_ALIAS].id


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_model("nope")
