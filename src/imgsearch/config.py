"""Central configuration: model registry, defaults, supported formats."""

from __future__ import annotations

import platform
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

    id: str  # HuggingFace repo id
    dim: int  # output embedding dimension
    image_size: int  # square input resolution expected by preprocessor
    family: str  # "mobileclip" | "siglip" | "clip"
    display_name: str


# Registry of supported models. Keys are short aliases used by --model.
MODEL_REGISTRY: Final[dict[str, ModelSpec]] = {
    # ── MobileCLIP (PyTorch + MPS) ──────────────────────────────────────────
    # Apple's fastest high-quality CLIP variant. Uses PyTorch MPS (Apple GPU).
    # Requires open-clip-torch + ml-mobileclip. dim=768, image_size=256.
    "mobileclip-s4": ModelSpec(
        id="apple/MobileCLIP-S4",
        dim=768,
        image_size=256,
        family="mobileclip",
        display_name="MobileCLIP-S4 (Apple)",
    ),
    # ── SigLIP (MLX) ────────────────────────────────────────────────────────
    # Confirmed working with mlx-embeddings. Best quality in the MLX family.
    "siglip-so400m": ModelSpec(
        id="mlx-community/siglip-so400m-patch14-384",
        dim=1152,
        image_size=384,
        family="siglip",
        display_name="SigLIP SO400M (patch14, 384)",
    ),
    "siglip-so400m-224": ModelSpec(
        id="mlx-community/siglip-so400m-patch14-224",
        dim=1152,
        image_size=224,
        family="siglip",
        display_name="SigLIP SO400M (patch14, 224)",
    ),
    "siglip2-base-8bit": ModelSpec(
        id="mlx-community/siglip2-base-patch16-224-8bit",
        dim=768,
        image_size=224,
        family="siglip",
        display_name="SigLIP2 Base 8-bit (patch16, 224)",
    ),
    # ── CLIP (MLX) ──────────────────────────────────────────────────────────
    "clip-vit-b32": ModelSpec(
        id="mlx-community/clip-vit-base-patch32",
        dim=512,
        image_size=224,
        family="clip",
        display_name="OpenAI CLIP ViT-B/32",
    ),
    "clip-vit-l14": ModelSpec(
        id="mlx-community/clip-vit-large-patch14",
        dim=768,
        image_size=224,
        family="clip",
        display_name="OpenAI CLIP ViT-L/14",
    ),
}

DEFAULT_MODEL_ALIAS: Final = "mobileclip-s4"
DEFAULT_BATCH_SIZE: Final = 16
DEFAULT_TOP_K: Final = 10


def resolve_model(alias_or_id: str) -> ModelSpec:
    """Resolve a user-provided model string (alias or full HF id) to a ModelSpec."""
    if alias_or_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias_or_id]
    # Allow passing a full HF id for any registered model by id as well
    for spec in MODEL_REGISTRY.values():
        if spec.id == alias_or_id:
            return spec
    raise ValueError(
        f"Unknown model: {alias_or_id!r}. "
        f"Available aliases: {', '.join(MODEL_REGISTRY)}"
    )


class PlatformError(RuntimeError):
    """Raised when the current platform cannot run imgsearch."""


def check_platform() -> None:
    """Verify we are on macOS arm64. Raises PlatformError otherwise."""
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin":
        raise PlatformError(
            f"imgsearch requires macOS (Darwin), got {system}. "
            "MLX acceleration is only available on Apple Silicon."
        )
    if machine not in {"arm64", "aarch64"}:
        raise PlatformError(
            f"imgsearch requires Apple Silicon (arm64), got {machine}. "
            "Intel Macs are not supported."
        )
