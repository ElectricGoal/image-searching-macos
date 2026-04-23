"""Central configuration: runtime selection, model registry, supported formats."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Final

INDEX_DIR_NAME: Final = ".imgsearch"
MANIFEST_FILE: Final = "manifest.json"
META_DB_FILE: Final = "meta.db"
FAISS_FILE: Final = "index.faiss"
MANIFEST_VERSION: Final = 1

# File extensions are lowercase with leading dot
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an embedding model."""

    alias: str
    id: str  # HuggingFace repo id
    dim: int  # output embedding dimension
    image_size: int  # square input resolution expected by preprocessor
    family: str  # "siglip" | "clip"
    display_name: str
    backend: str  # "mlx" | "torch"


SUPPORTED_PYTHON_MIN: Final = (3, 11)
SUPPORTED_PYTHON_MAX_EXCLUSIVE: Final = (3, 13)
SUPPORTED_BACKENDS: Final[frozenset[str]] = frozenset({"mlx", "torch"})

# Registry of supported models. Each alias may map to a different backend-specific model id.
MODEL_REGISTRY: Final[dict[str, dict[str, ModelSpec]]] = {
    "siglip-so400m": {
        "mlx": ModelSpec(
            alias="siglip-so400m",
            id="mlx-community/siglip-so400m-patch14-384",
            dim=1152,
            image_size=384,
            family="siglip",
            display_name="SigLIP SO400M (patch14, 384)",
            backend="mlx",
        ),
        "torch": ModelSpec(
            alias="siglip-so400m",
            id="google/siglip-so400m-patch14-384",
            dim=1152,
            image_size=384,
            family="siglip",
            display_name="SigLIP SO400M (patch14, 384)",
            backend="torch",
        ),
    },
    "siglip-so400m-224": {
        "mlx": ModelSpec(
            alias="siglip-so400m-224",
            id="mlx-community/siglip-so400m-patch14-224",
            dim=1152,
            image_size=224,
            family="siglip",
            display_name="SigLIP SO400M (patch14, 224)",
            backend="mlx",
        ),
        "torch": ModelSpec(
            alias="siglip-so400m-224",
            id="google/siglip-so400m-patch14-224",
            dim=1152,
            image_size=224,
            family="siglip",
            display_name="SigLIP SO400M (patch14, 224)",
            backend="torch",
        ),
    },
    "siglip2-base-8bit": {
        "mlx": ModelSpec(
            alias="siglip2-base-8bit",
            id="mlx-community/siglip2-base-patch16-224-8bit",
            dim=768,
            image_size=224,
            family="siglip",
            display_name="SigLIP2 Base 8-bit (patch16, 224)",
            backend="mlx",
        ),
        "torch": ModelSpec(
            alias="siglip2-base-8bit",
            id="google/siglip2-base-patch16-224",
            dim=768,
            image_size=224,
            family="siglip",
            display_name="SigLIP2 Base (patch16, 224)",
            backend="torch",
        ),
    },
    "clip-vit-b32": {
        "mlx": ModelSpec(
            alias="clip-vit-b32",
            id="mlx-community/clip-vit-base-patch32",
            dim=512,
            image_size=224,
            family="clip",
            display_name="OpenAI CLIP ViT-B/32",
            backend="mlx",
        ),
        "torch": ModelSpec(
            alias="clip-vit-b32",
            id="openai/clip-vit-base-patch32",
            dim=512,
            image_size=224,
            family="clip",
            display_name="OpenAI CLIP ViT-B/32",
            backend="torch",
        ),
    },
    "clip-vit-l14": {
        "mlx": ModelSpec(
            alias="clip-vit-l14",
            id="mlx-community/clip-vit-large-patch14",
            dim=768,
            image_size=224,
            family="clip",
            display_name="OpenAI CLIP ViT-L/14",
            backend="mlx",
        ),
        "torch": ModelSpec(
            alias="clip-vit-l14",
            id="openai/clip-vit-large-patch14",
            dim=768,
            image_size=224,
            family="clip",
            display_name="OpenAI CLIP ViT-L/14",
            backend="torch",
        ),
    },
}

DEFAULT_MODEL_ALIAS: Final = "clip-vit-b32"
DEFAULT_BATCH_SIZE: Final = 16
DEFAULT_TOP_K: Final = 10


def _python_version_tuple(version_info: tuple[int, ...] | None = None) -> tuple[int, int]:
    if version_info is None:
        version_info = sys.version_info
    return int(version_info[0]), int(version_info[1])


def is_supported_python(version_info: tuple[int, ...] | None = None) -> bool:
    """Return whether the interpreter version is supported by the package."""
    version = _python_version_tuple(version_info)
    return SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX_EXCLUSIVE


def default_backend(system: str | None = None, machine: str | None = None) -> str:
    """Choose the preferred embedding backend for the current platform."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "mlx"
    if system == "Linux":
        return "torch"
    raise PlatformError(
        f"imgsearch supports macOS arm64 and Linux via pip-installable backends; got "
        f"{system} {machine}."
    )


def resolve_model(alias_or_id: str, backend: str | None = None) -> ModelSpec:
    """Resolve a user-provided model string (alias or full HF id) to a backend-specific spec."""
    if alias_or_id in MODEL_REGISTRY:
        chosen_backend = backend or default_backend()
        try:
            return MODEL_REGISTRY[alias_or_id][chosen_backend]
        except KeyError as exc:
            raise ValueError(
                f"Model {alias_or_id!r} is not available for backend {chosen_backend!r}."
            ) from exc
    for variants in MODEL_REGISTRY.values():
        for spec in variants.values():
            if spec.id == alias_or_id:
                return spec
    raise ValueError(
        f"Unknown model: {alias_or_id!r}. "
        f"Available aliases: {', '.join(MODEL_REGISTRY)}"
    )


class PlatformError(RuntimeError):
    """Raised when the current platform cannot run imgsearch."""


def check_platform() -> None:
    """Verify that the current runtime can select a supported embedding backend."""
    version = _python_version_tuple()
    if not is_supported_python(version):
        raise PlatformError(
            "imgsearch supports Python 3.11 or 3.12 only; "
            f"found {version[0]}.{version[1]}."
        )
    default_backend()
