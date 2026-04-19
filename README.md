# imgsearch

Semantic image search CLI for macOS, accelerated by [MLX](https://github.com/ml-explore/mlx).

Search a folder of images using **either text or another image** as the query. Embeddings are computed once with an MLX-native CLIP/SigLIP model, stored in a hidden `.imgsearch/` directory next to your photos, and searched with FAISS.

## Requirements

- macOS 13.5+
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- ~16 GB RAM recommended (8 GB works at reduced batch size)

## Install

### From PyPI (recommended for now)

```bash
pipx install imgsearch
```

### From Homebrew (once the tap is published)

```bash
brew tap toxu/imgsearch
brew install imgsearch
```

## Usage

### Index a folder

```bash
imgsearch index ~/Pictures
```

First run downloads the model (~400 MB for SigLIP2-base) into the HuggingFace cache. Re-running only re-embeds files whose contents changed.

Progress is shown with a live bar:

```
imgsearch indexing /Users/you/Pictures with SigLIP2 Base (patch16, 256)
Scanning files...
  found 1024 image files
  plan: +1024 new/changed, ~0 mtime-only, -0 removed, 0 unchanged
Loading model (first run downloads weights)...
  model ready in 2.1s
⠋ embedding ━━━━━━━━━━━━━━━━━━━━━━━━━━ 1024/1024  0:00:46
✓ indexed 1024 images in 46.2s (22.2 img/s)
```

### Search by text

```bash
imgsearch search ~/Pictures --text "a dog on a beach"
```

### Search by example image

```bash
imgsearch search ~/Pictures --image ~/Desktop/reference.jpg -k 5
```

### JSON output (for scripting)

```bash
imgsearch search ~/Pictures -t "sunset" --json | jq '.[0].path'
```

### Inspect / clean

```bash
imgsearch status ~/Pictures
imgsearch clean ~/Pictures        # removes the .imgsearch/ directory
```

## How it works

1. `scanner.py` walks the folder, collecting supported image files and their mtimes.
2. `index.plan()` compares mtime (fast path) then SHA-1 (slow path) against the existing metadata DB and decides what to re-embed.
3. `MLXEmbedder` runs a SigLIP2/CLIP model on Apple's MLX framework and returns L2-normalized float32 vectors.
4. `VectorStore` wraps a FAISS `IndexIDMap2(IndexFlatIP)` — exact brute-force cosine search, no training, crash-safe atomic writes.
5. `MetaStore` is a small SQLite database holding the mapping from FAISS id to file path, plus hashes and mtimes.

On 1,000 images on an M2 16GB laptop, indexing takes ~45s and search latency is sub-millisecond.

## Supported models

| Alias | HuggingFace id | Dim | Notes |
|---|---|---|---|
| `siglip2-base` *(default)* | `mlx-community/siglip2-base-patch16-256` | 768 | Best quality/speed trade-off |
| `siglip-base` | `mlx-community/siglip-base-patch16-224` | 768 | Older, slightly faster |
| `clip-vit-b32` | `mlx-community/clip-vit-base-patch32` | 512 | Fastest, OpenAI CLIP |

Change with `imgsearch index <folder> --model clip-vit-b32`. Once indexed, the manifest remembers the model id so subsequent searches use the same one.

## Troubleshooting

- **"Index was built with model X"**: you're searching with a different model than you indexed with. Either pass `--model X` to match or `imgsearch clean <folder>` and re-index.
- **Out of memory on 8 GB**: pass `--batch 4` to `index`.
- **HEIC files fail**: make sure `pillow-heif` installed cleanly. It ships wheels for macOS arm64.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency and environment management.

```bash
# One-time setup: create venv and install all deps (including dev group)
uv sync

# Run the CLI
uv run imgsearch --help

# Run tests
uv run pytest -q

# Add a new dependency
uv add <package>
uv add --dev <package>

# Upgrade lockfile
uv lock --upgrade
```

The project pins Python 3.11 via `.python-version`. `uv sync` will download a matching interpreter automatically if you don't have one. The lockfile (`uv.lock`) is committed so builds are reproducible.

## License

MIT
