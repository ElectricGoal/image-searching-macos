from __future__ import annotations

import pytest

from imgsearch.config import (
    DEFAULT_MODEL_ALIAS,
    MODEL_REGISTRY,
    PlatformError,
    default_backend,
    is_supported_python,
    resolve_model,
)


def test_default_alias_registered() -> None:
    assert DEFAULT_MODEL_ALIAS in MODEL_REGISTRY


def test_resolve_by_alias() -> None:
    spec = resolve_model(DEFAULT_MODEL_ALIAS, backend="torch")
    assert spec.dim > 0
    assert spec.backend == "torch"


def test_resolve_by_id() -> None:
    expected = MODEL_REGISTRY[DEFAULT_MODEL_ALIAS]["mlx"]
    spec = resolve_model(expected.id)
    assert spec.id == expected.id


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_model("nope")


def test_default_backend_linux() -> None:
    assert default_backend("Linux", "x86_64") == "torch"


def test_default_backend_macos_arm() -> None:
    assert default_backend("Darwin", "arm64") == "mlx"


def test_default_backend_unsupported_raises() -> None:
    with pytest.raises(PlatformError):
        default_backend("Windows", "AMD64")


def test_supported_python_versions() -> None:
    assert is_supported_python((3, 11, 0))
    assert is_supported_python((3, 12, 9))
    assert not is_supported_python((3, 13, 0))
