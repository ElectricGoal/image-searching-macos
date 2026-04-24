"""MLX embedding model wrapper.

Thin façade over `mlx-embeddings` that exposes a stable two-method interface
(`encode_images`, `encode_text`) returning L2-normalized float32 numpy arrays.

The underlying package is imported lazily so unit tests that mock the embedder
can run on non-Mac CI and so the CLI starts fast when the user runs
`imgsearch --help` without loading the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from PIL import Image

from imgsearch.config import ModelSpec
from imgsearch.core.preprocess import load_rgb


class EmbedderProtocol(Protocol):
    """Interface used by the Index facade. Allows test doubles."""

    spec: ModelSpec

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray: ...
    def encode_pil(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_text(self, text: str) -> np.ndarray: ...


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Safe on zero rows."""
    if x.ndim == 1:
        norm = float(np.linalg.norm(x))
        return x if norm == 0.0 else (x / norm).astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (x / norms).astype(np.float32, copy=False)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert mx.array / torch.Tensor / numpy / list to a numpy array."""
    if isinstance(value, np.ndarray):
        return value
    # mlx arrays expose __array__ via np.asarray()
    try:
        return np.asarray(value)
    except Exception:
        pass
    # torch fallback
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    raise TypeError(f"Cannot convert {type(value).__name__} to numpy")


class MLXEmbedder:
    """Production embedder backed by mlx-embeddings."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._processor: Any = None

    # ----- lifecycle -----

    def load(self) -> None:
        """Download (if needed) and load model + processor. Idempotent."""
        if self._model is not None:
            return
        try:
            from mlx_embeddings.utils import load as mlx_load  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "mlx-embeddings is not installed. "
                "Install with: pip install 'mlx-embeddings>=0.0.5'"
            ) from exc
        self._model, self._processor = mlx_load(self.spec.id)

    # ----- image encoding -----

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        """Encode image files from disk. Returns (N, dim) L2-normalized float32."""
        if not paths:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        self.load()
        images = [load_rgb(p) for p in paths]
        return self.encode_pil(images)

    def encode_pil(self, images: Sequence[Any]) -> np.ndarray:
        """Encode pre-loaded PIL images. Returns (N, dim) L2-normalized float32.

        Preferred over encode_images when images are already loaded (e.g. from
        the prefetch queue), avoiding a redundant file open.
        """
        if not images:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        self.load()
        features = self._run_image_batch(images)
        return _l2_normalize(features)

    def _run_image_batch(self, images: Iterable[Any]) -> np.ndarray:
        """Invoke the underlying processor + model to get raw features."""
        assert self._model is not None and self._processor is not None
        inputs = self._processor(
            images=list(images),
            return_tensors="mlx",
        )
        pixel_values = inputs["pixel_values"]

        # Prefer the dedicated helper if present (HF CLIP/SigLIP convention)
        if hasattr(self._model, "get_image_features"):
            feats = self._model.get_image_features(pixel_values=pixel_values)
        else:
            outputs = self._model(pixel_values=pixel_values)
            feats = _extract_image_embeds(outputs)

        # Force MLX to materialise the lazy compute graph now, before the next
        # batch accumulates. Without this, MLX can defer evaluation across
        # batches causing a large memory spike and a slow flush at commit time.
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
            mx.eval(feats)
        except Exception:
            pass  # non-fatal: mx.eval is an optimisation, not a correctness requirement

        arr = _to_numpy(feats).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] != self.spec.dim:
            raise RuntimeError(
                f"Unexpected image feature shape {arr.shape}, "
                f"expected (*, {self.spec.dim})"
            )
        return arr

    # ----- text encoding -----

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text query. Returns (dim,) L2-normalized float32."""
        if not text or not text.strip():
            raise ValueError("text query must be non-empty")
        self.load()
        inputs = self._processor(
            text=[text],
            padding="max_length",
            return_tensors="mlx",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        kwargs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        if hasattr(self._model, "get_text_features"):
            feats = self._model.get_text_features(**kwargs)
        else:
            outputs = self._model(**kwargs)
            feats = _extract_text_embeds(outputs)
        arr = _to_numpy(feats).astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[0]
        if arr.shape[0] != self.spec.dim:
            raise RuntimeError(
                f"Unexpected text feature shape {arr.shape}, "
                f"expected ({self.spec.dim},)"
            )
        return _l2_normalize(arr)


class MobileCLIPEmbedder:
    """Production embedder backed by Apple MobileCLIP via open_clip + PyTorch MPS.

    Apple's MobileCLIP HF repos (e.g. apple/MobileCLIP-S4) ship a raw PyTorch
    checkpoint (*.pt), not an open_clip config bundle. The ml-mobileclip package
    registers the custom fastvit_mci* architectures with open_clip at import
    time, after which open_clip.create_model_and_transforms can build the model
    from its architecture name ("MobileCLIP2-S4") and load the checkpoint.
    """

    # Map HF repo id → (open_clip architecture name, checkpoint filename).
    # MobileCLIP-S4 uses the MobileCLIP2-S4 architecture config (identical
    # fastvit_mci4 encoder); only the training data/procedure differs.
    _MODEL_MAP: dict[str, tuple[str, str]] = {
        "apple/MobileCLIP-S4": ("MobileCLIP2-S4", "mobileclip_s4.pt"),
        "apple/MobileCLIP2-S4": ("MobileCLIP2-S4", "mobileclip2_s4.pt"),
    }

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        # Eagerly import torch here so its bundled OpenMP runtime loads *before*
        # faiss-cpu's bundled OpenMP (imported lazily by VectorStore). Loading
        # faiss first causes a silent SIGSEGV the first time we touch a torch
        # tensor on macOS arm64. Running the import here — in the constructor
        # called by commands/*.py — guarantees the correct ordering regardless
        # of when Index() lazily imports faiss.
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MobileCLIP requires torch, open-clip-torch, and ml-mobileclip. "
                "Install with: uv sync --extra mobileclip"
            ) from exc

    def load(self) -> None:
        """Download (if needed) and load model. Idempotent."""
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            import open_clip  # type: ignore[import-not-found]
            import mobileclip  # type: ignore[import-not-found]  # noqa: F401  (registers architectures)
            from mobileclip.modules.common.mobileone import (  # type: ignore[import-not-found]
                reparameterize_model,
            )
            from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MobileCLIP requires torch, open-clip-torch, and ml-mobileclip. "
                "Install with: uv sync --extra mobileclip"
            ) from exc

        entry = self._MODEL_MAP.get(self.spec.id)
        if entry is None:
            raise RuntimeError(
                f"No MobileCLIP load recipe for HF repo {self.spec.id!r}. "
                f"Known repos: {', '.join(self._MODEL_MAP)}"
            )
        arch_name, ckpt_filename = entry

        ckpt_path = hf_hub_download(repo_id=self.spec.id, filename=ckpt_filename)

        # Apple's README passes these kwargs; they disable open_clip's default
        # ImageNet normalization because the MobileCLIP preprocessing pipeline
        # already handles scaling.
        model_kwargs = {"image_mean": (0.0, 0.0, 0.0), "image_std": (1.0, 1.0, 1.0)}
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch_name, pretrained=ckpt_path, **model_kwargs
        )
        model.eval()
        # Fuse reparameterizable branches (MobileOne / FastViT) for inference speed.
        model = reparameterize_model(model)

        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(self._device)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(arch_name)

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        """Encode image files from disk. Returns (N, dim) L2-normalized float32."""
        if not paths:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        self.load()
        images = [load_rgb(p) for p in paths]
        return self.encode_pil(images)

    def encode_pil(self, images: Sequence[Any]) -> np.ndarray:
        """Encode pre-loaded PIL images. Returns (N, dim) L2-normalized float32."""
        if not images:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        self.load()
        import torch  # type: ignore[import-not-found]

        tensors = torch.stack([self._preprocess(img) for img in images]).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().float().numpy()
        if arr.ndim != 2 or arr.shape[1] != self.spec.dim:
            raise RuntimeError(
                f"Unexpected image feature shape {arr.shape}, "
                f"expected (*, {self.spec.dim})"
            )
        return arr.astype(np.float32, copy=False)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text query. Returns (dim,) L2-normalized float32."""
        if not text or not text.strip():
            raise ValueError("text query must be non-empty")
        self.load()
        import torch  # type: ignore[import-not-found]

        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().float().numpy()
        if arr.ndim == 2:
            arr = arr[0]
        if arr.shape[0] != self.spec.dim:
            raise RuntimeError(
                f"Unexpected text feature shape {arr.shape}, "
                f"expected ({self.spec.dim},)"
            )
        return arr.astype(np.float32, copy=False)


def create_embedder(spec: ModelSpec) -> MLXEmbedder | MobileCLIPEmbedder:
    """Factory: return the correct embedder for the given ModelSpec."""
    if spec.family == "mobileclip":
        return MobileCLIPEmbedder(spec)
    return MLXEmbedder(spec)


def _extract_image_embeds(outputs: Any) -> Any:
    """Pull image embeddings out of a variety of output container shapes."""
    for name in ("image_embeds", "vision_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(outputs, name):
            return getattr(outputs, name)
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
    raise RuntimeError("Could not locate image embeddings in model output")


def _extract_text_embeds(outputs: Any) -> Any:
    """Pull text embeddings out of a variety of output container shapes."""
    for name in ("text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(outputs, name):
            return getattr(outputs, name)
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
    raise RuntimeError("Could not locate text embeddings in model output")
