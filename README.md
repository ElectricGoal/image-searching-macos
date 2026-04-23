# imgsearch

Semantic image search CLI for macOS and Linux with automatic embedding backend selection.

Search a folder of images using **either text or another image** as the query. Embeddings are computed once with a platform-appropriate CLIP/SigLIP backend, stored in a hidden `.imgsearch/` directory next to your photos, and searched with FAISS.

## Requirements

- Python 3.11 or 3.12
- macOS on Apple Silicon uses MLX
- Linux uses `transformers` + `torch` via `pip`
- ~16 GB RAM recommended (8 GB works at reduced batch size)

## Install

### From PyPI

```bash
pip install imgsearch
```

For Linux, no `apt-get` or `pacman` packages are required.

If your machine only has Python 3.13 installed, create a Python 3.11 or 3.12 environment first.

### macOS notes

```bash
pip install imgsearch
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

### Find and remove duplicate images

```bash
# Dry run — show duplicate groups, no files changed (default)
imgsearch dedup ~/Pictures

# Stricter threshold (default 0.98, range 0.5–1.0)
imgsearch dedup ~/Pictures --threshold 0.995

# Only find exact byte-for-byte duplicates (SHA-1), no embedding comparison
imgsearch dedup ~/Pictures --exact-only

# Move duplicates to a quarantine folder (safer than delete)
imgsearch dedup ~/Pictures --move-to ~/Desktop/duplicates --no-dry-run

# Delete duplicates, keep the largest file, skip confirmation
imgsearch dedup ~/Pictures --delete --keep largest --force
```

Two levels of detection — no model reload required:

- **Exact**: SHA-1 hash match (same file, different name/location) — instant, zero false positives
- **Similar**: cosine similarity on stored embeddings (same scene, different compression/crop) — no model reload needed

Keep strategies: `largest` (default), `newest`, `oldest`, `highest-res`.

Example output:

```
Found 3 duplicate group(s) — 5 redundant image(s)

Group 1 — exact (SHA-1)
  ✓ KEEP  /Users/you/Pictures/IMG_1234.jpg     3.2 MB
    DUP   /Users/you/Downloads/copy.jpg        3.2 MB

Group 2 — similar (cosine 0.994)
  ✓ KEEP  /Users/you/Pictures/sunset.heic      5.8 MB
    DUP   /Users/you/Desktop/sunset_edit.jpg   1.1 MB
    DUP   /Users/you/Desktop/sunset_small.png  412 KB
```

### Inspect / clean

```bash
imgsearch status ~/Pictures
imgsearch clean ~/Pictures        # removes the .imgsearch/ directory
```

## How it works

1. `scanner.py` walks the folder, collecting supported image files and their mtimes.
2. `index.plan()` compares mtime (fast path) then SHA-1 (slow path) against the existing metadata DB and decides what to re-embed.
3. The embedder backend is selected automatically:
   - macOS arm64: MLX via `mlx-embeddings`
   - Linux: `transformers` + `torch` with CUDA when available, otherwise CPU
4. `VectorStore` wraps a FAISS `IndexIDMap2(IndexFlatIP)` — exact brute-force cosine search, no training, crash-safe atomic writes.
5. `MetaStore` is a small SQLite database holding the mapping from FAISS id to file path, plus hashes and mtimes.

On 1,000 images on an M2 16GB laptop, indexing takes ~45s and search latency is sub-millisecond. Linux performance depends on whether `torch` runs on CPU or CUDA.

## Supported models

| Alias | HuggingFace id | Dim | Notes |
|---|---|---|---|
| `siglip-so400m` | backend-specific | 1152 | Best retrieval quality |
| `siglip-so400m-224` | backend-specific | 1152 | Same quality, faster indexing |
| `siglip2-base-8bit` | backend-specific | 768 | Lowest RAM on macOS, standard SigLIP2 on Linux |
| `clip-vit-b32` *(default)* | backend-specific | 512 | Fastest, widest compatibility |
| `clip-vit-l14` | backend-specific | 768 | Strong CLIP alternative |

Change with `imgsearch index <folder> --model clip-vit-b32`. Once indexed, the manifest remembers the model id so subsequent searches use the same one.

## Troubleshooting

- **"Index was built with model X"**: you're searching with a different model than you indexed with. Either pass `--model X` to match or `imgsearch clean <folder>` and re-index.
- **Out of memory on 8 GB**: pass `--batch 4` to `index`.
- **Python 3.13 installed locally**: create a Python 3.11 or 3.12 environment before installing `imgsearch`.
- **HEIC files fail**: make sure `pillow-heif` installed cleanly.

## Development

This project uses [uv](https://github.com/astral-sh/uv) for development workflow on both macOS and Linux.

```bash
# One-time setup: create venv and install all deps (including dev group)
uv sync

# Run the CLI
uv run imgsearch --help

# Run tests
uv run pytest -q
```

`uv sync` will use the project's Python constraint (`>=3.11,<3.13`). On this machine, that means using Python 3.11 or 3.12 rather than the system Python 3.13.
On Linux, `uv sync` is configured to install CPU-only `torch` from PyTorch's official CPU wheel index.

## License

MIT
