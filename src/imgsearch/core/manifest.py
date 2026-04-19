"""Manifest file read/write — tracks index identity and stats."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from imgsearch.config import MANIFEST_VERSION, ModelSpec


@dataclass
class Manifest:
    """On-disk index metadata. Stored as JSON next to meta.db."""

    version: int = MANIFEST_VERSION
    model_id: str = ""
    model_alias: str = ""
    dim: int = 0
    image_size: int = 0
    family: str = ""
    index_type: str = "IndexFlatIP"
    count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_model(cls, spec: ModelSpec, alias: str) -> Manifest:
        return cls(
            model_id=spec.id,
            model_alias=alias,
            dim=spec.dim,
            image_size=spec.image_size,
            family=spec.family,
        )

    def touch(self, count: int) -> None:
        self.count = count
        self.updated_at = time.time()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text())
        return cls(**data)


class ManifestMismatch(RuntimeError):
    """Raised when a requested model differs from the one used to index."""


def ensure_compatible(manifest: Manifest, spec: ModelSpec) -> None:
    """Refuse to mix vectors from different models or dimensions."""
    if manifest.version != MANIFEST_VERSION:
        raise ManifestMismatch(
            f"Index was built with manifest version {manifest.version}, "
            f"this build expects {MANIFEST_VERSION}. Run `imgsearch clean` and re-index."
        )
    if manifest.model_id != spec.id or manifest.dim != spec.dim:
        raise ManifestMismatch(
            f"Index was built with model '{manifest.model_id}' (dim={manifest.dim}), "
            f"but you requested '{spec.id}' (dim={spec.dim}). "
            f"Re-run with --model {manifest.model_alias} or `imgsearch clean` first."
        )
