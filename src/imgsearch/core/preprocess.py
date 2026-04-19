"""Image loading and preprocessing to model-ready tensors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Register HEIC/HEIF opener on import. Safe no-op if unavailable.
try:
    import pillow_heif  # type: ignore[import-not-found]

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional at runtime
    pass

# Tolerate mildly truncated files rather than crashing a 1K-image index on one bad byte.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoadError(Exception):
    """Raised when an image cannot be decoded."""


@dataclass(frozen=True)
class PreparedImage:
    """Result of a single-pass open: image ready for the processor plus metadata."""

    image: Image.Image  # RGB, resized to model's image_size
    sha1: str
    orig_width: int
    orig_height: int


def load_and_prepare(path: Path, image_size: int, chunk_size: int = 1 << 20) -> PreparedImage:
    """Open once: compute SHA-1, capture original dims, resize — all in one read.

    Pre-resizing to `image_size` before the processor runs keeps per-batch RAM
    proportional to batch_size × image_size² instead of batch_size × orig_size².
    For 4K source photos this is a ~300× memory reduction per image.
    """
    try:
        h = hashlib.sha1()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        sha1 = h.hexdigest()

        img = Image.open(path)
        img.load()
        orig_w, orig_h = img.size
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return PreparedImage(image=img, sha1=sha1, orig_width=orig_w, orig_height=orig_h)
    except Exception as exc:
        raise ImageLoadError(f"failed to prepare {path}: {exc}") from exc


def load_rgb(path: Path) -> Image.Image:
    """Open an image as RGB. Raises ImageLoadError on failure."""
    try:
        img = Image.open(path)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as exc:
        raise ImageLoadError(f"failed to decode {path}: {exc}") from exc


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an (H, W, 3) uint8 numpy array."""
    return np.asarray(img, dtype=np.uint8)
