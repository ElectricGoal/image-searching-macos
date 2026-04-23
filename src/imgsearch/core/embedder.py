"""Embedding backends for MLX and transformers/torch."""

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

    def load(self) -> None: ...
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
    # torch tensors need to be handled before np.asarray(), otherwise some
    # transformers model outputs can degrade into object arrays.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    # mlx arrays expose __array__ via np.asarray()
    try:
        return np.asarray(value)
    except Exception:
        pass
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


class TorchEmbedder:
    """Production embedder backed by transformers + torch."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None

    def load(self) -> None:
        """Download (if needed) and load model + processor. Idempotent."""
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "transformers and torch are required on Linux. "
                "Install with: pip install 'torch>=2.6' 'transformers>=4.51'"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.spec.id)
        self._model = AutoModel.from_pretrained(self.spec.id)
        self._model.eval()
        self._model.to(self._device)

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        self.load()
        images = [load_rgb(p) for p in paths]
        return self.encode_pil(images)

    def encode_pil(self, images: Sequence[Any]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.spec.dim), dtype=np.float32)
        self.load()
        features = self._run_image_batch(images)
        return _l2_normalize(features)

    def _run_image_batch(self, images: Iterable[Any]) -> np.ndarray:
        assert self._model is not None and self._processor is not None and self._device is not None
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError("torch is required to run the selected backend") from exc

        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = _move_to_device(inputs, self._device)
        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                feats = self._model.get_image_features(**inputs)
                feats = _coerce_feature_output(feats, kind="image")
            else:
                outputs = self._model(**inputs)
                feats = _extract_image_embeds(outputs)
        arr = _to_numpy(feats).astype(np.float32, copy=False)
        if arr.ndim != 2 or arr.shape[1] != self.spec.dim:
            raise RuntimeError(
                f"Unexpected image feature shape {arr.shape}, "
                f"expected (*, {self.spec.dim})"
            )
        return arr

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("text query must be non-empty")
        self.load()
        assert self._model is not None and self._processor is not None and self._device is not None
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError("torch is required to run the selected backend") from exc

        inputs = self._processor(text=[text], padding=True, truncation=True, return_tensors="pt")
        inputs = _move_to_device(inputs, self._device)
        with torch.inference_mode():
            if hasattr(self._model, "get_text_features"):
                feats = self._model.get_text_features(**inputs)
                feats = _coerce_feature_output(feats, kind="text")
            else:
                outputs = self._model(**inputs)
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


def _extract_image_embeds(outputs: Any) -> Any:
    """Pull image embeddings out of a variety of output container shapes."""
    if isinstance(outputs, (tuple, list)) and outputs:
        return _extract_image_embeds(outputs[0])
    for name in ("image_embeds", "vision_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(outputs, name):
            return getattr(outputs, name)
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
    raise RuntimeError("Could not locate image embeddings in model output")


def _extract_text_embeds(outputs: Any) -> Any:
    """Pull text embeddings out of a variety of output container shapes."""
    if isinstance(outputs, (tuple, list)) and outputs:
        return _extract_text_embeds(outputs[0])
    for name in ("text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(outputs, name):
            return getattr(outputs, name)
        if isinstance(outputs, dict) and name in outputs:
            return outputs[name]
    raise RuntimeError("Could not locate text embeddings in model output")


def _coerce_feature_output(value: Any, kind: str) -> Any:
    """Normalize backend feature helpers that return full model outputs."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value
    if kind == "image":
        return _extract_image_embeds(value)
    if kind == "text":
        return _extract_text_embeds(value)
    raise ValueError(f"Unknown feature kind: {kind}")


def _move_to_device(inputs: Any, device: str) -> Any:
    """Recursively move tensors returned by the processor to the target device."""
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except Exception:
            pass
    if isinstance(inputs, dict):
        return {key: _move_to_device(value, device) for key, value in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        items = [_move_to_device(value, device) for value in inputs]
        return type(inputs)(items)
    return inputs


def create_embedder(spec: ModelSpec) -> EmbedderProtocol:
    """Instantiate the concrete embedder for the requested model backend."""
    if spec.backend == "mlx":
        return MLXEmbedder(spec)
    if spec.backend == "torch":
        return TorchEmbedder(spec)
    raise ValueError(f"Unsupported backend: {spec.backend}")
