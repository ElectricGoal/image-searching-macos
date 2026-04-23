# imgsearch

Cross-platform semantic image search CLI for macOS and Linux.

Index a folder of images once, then search it with either text or another image. Embeddings are stored in a hidden `.imgsearch/` directory next to your photos and searched with FAISS for fast similarity lookup.

## Platform support

- macOS: Apple Silicon only, using MLX
- Linux: `x86_64` and other pip-compatible targets using `transformers` + `torch`
- Python: 3.11 or 3.12

Backend selection is automatic:

- macOS arm64 uses `mlx` + `mlx-embeddings`
- Linux uses `transformers` + `torch`

## Install

### pip

```bash
pip install imgsearch
```

Notes:

- Linux does not require `apt-get` or `pacman`
- In development, `uv sync` is configured to use CPU-only `torch` on Linux
- If your system Python is 3.13, create a Python 3.11 or 3.12 environment first

## Usage

### Index a folder

```bash
imgsearch index ~/Pictures
```

### Search by text

```bash
imgsearch search ~/Pictures --text "a dog on a beach"
```

### Search by example image

```bash
imgsearch search ~/Pictures --image ~/Desktop/reference.jpg -k 5
```

### JSON output

```bash
imgsearch search ~/Pictures --text "sunset" --json
```

### Deduplicate similar and exact images

```bash
# Dry run (default)
imgsearch dedup ~/Pictures

# Exact duplicates only
imgsearch dedup ~/Pictures --exact-only

# Move duplicates to another folder
imgsearch dedup ~/Pictures --move-to ~/duplicates --no-dry-run

# Delete duplicates and keep the largest file
imgsearch dedup ~/Pictures --delete --keep largest --force
```

### Inspect or remove an index

```bash
imgsearch status ~/Pictures
imgsearch clean ~/Pictures
```

## How it works

1. `scanner.py` walks the target folder and finds supported image files.
2. `index.plan()` compares current files against the existing metadata and decides what to add, update, or remove.
3. The active embedding backend computes normalized vectors for text and images.
4. `VectorStore` stores vectors in FAISS using exact cosine similarity search.
5. `MetaStore` stores file paths, hashes, mtimes, and dimensions in SQLite.

The first index run downloads model weights into the Hugging Face cache. Re-indexing only recomputes files whose contents changed.

## Supported models

| Alias | Dim | Notes |
|---|---|---|
| `clip-vit-b32` *(default)* | 512 | Fastest, broadest compatibility |
| `clip-vit-l14` | 768 | Strong CLIP alternative |
| `siglip2-base-8bit` | 768 | Lower memory usage on macOS |
| `siglip-so400m-224` | 1152 | Better retrieval quality, faster than full 384 |
| `siglip-so400m` | 1152 | Best retrieval quality |

The underlying Hugging Face model id is backend-specific. Once a folder is indexed, the manifest records the exact model used so later searches stay compatible.

## Supported image formats

Common formats are supported, including:

- `jpg`, `jpeg`, `png`, `webp`
- `bmp`, `tif`, `tiff`
- `heic`, `heif`

## Troubleshooting

- If you see `Index was built with model X`, search with the same model or run `imgsearch clean <folder>` and re-index.
- If indexing uses too much memory, reduce batch size with `--batch 4`.
- If HEIC or HEIF files fail to load, verify `pillow-heif` installed successfully.
- If Python 3.13 is the only interpreter on the machine, create a 3.11 or 3.12 environment first.

## Development

This project uses [uv](https://github.com/astral-sh/uv) on both macOS and Linux.

```bash
# Create/update the environment
uv sync

# Run the CLI
uv run imgsearch --help

# Run tests
uv run pytest -q
```

Notes:

- `uv sync` respects the project Python constraint: `>=3.11,<3.13`
- On Linux, `uv sync` is configured to install CPU-only `torch`

## License

MIT
