# imgsearch

Semantic image search CLI for macOS, accelerated by [MLX](https://github.com/ml-explore/mlx).

Search a folder of images using **either text or another image** as the query. Embeddings are computed once using [MobileCLIP-S4](https://huggingface.co/apple/MobileCLIP-S4) (default) or other CLIP/SigLIP models, stored in a hidden `.imgsearch/` directory next to your photos, and searched with FAISS.

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

After install, add **MobileCLIP-S4** (the default model):

```bash
pip install "mobileclip @ git+https://github.com/apple/ml-mobileclip.git"
```

> MobileCLIP is distributed as a git package with non-standard build metadata, so it is listed as an optional dependency rather than bundled in the main install. All other models (SigLIP, CLIP) work without this step.

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

First run downloads the model weights into the HuggingFace cache (~200 MB for MobileCLIP-S4). Re-running only re-embeds files whose contents changed.

Progress is shown with a live bar:

```
imgsearch indexing /Users/you/Pictures with MobileCLIP-S4 (Apple)
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
3. The embedder (MobileCLIP via PyTorch MPS, or SigLIP/CLIP via MLX) returns L2-normalized float32 vectors.
4. `VectorStore` wraps a FAISS `IndexIDMap2(IndexFlatIP)` — exact brute-force cosine search, no training, crash-safe atomic writes.
5. `MetaStore` is a small SQLite database holding the mapping from FAISS id to file path, plus hashes and mtimes.

On 1,000 images on an M2 16GB laptop, indexing takes ~45s and search latency is sub-millisecond.

## Supported models

| Alias | HuggingFace id | Dim | Backend | Notes |
|---|---|---|---|---|
| `mobileclip-s4` *(default)* | `apple/MobileCLIP-S4` | 768 | PyTorch MPS | Apple's fastest high-quality CLIP. Requires ml-mobileclip. |
| `siglip-so400m` | `mlx-community/siglip-so400m-patch14-384` | 1152 | MLX | Best retrieval quality |
| `siglip-so400m-224` | `mlx-community/siglip-so400m-patch14-224` | 1152 | MLX | Same quality, faster indexing |
| `siglip2-base-8bit` | `mlx-community/siglip2-base-patch16-224-8bit` | 768 | MLX | Quantized, lowest RAM usage |
| `clip-vit-b32` | `mlx-community/clip-vit-base-patch32` | 512 | MLX | Fastest, OpenAI CLIP |
| `clip-vit-l14` | `mlx-community/clip-vit-large-patch14` | 768 | MLX | Strong CLIP alternative |

Change with `imgsearch index <folder> --model siglip-so400m`. Once indexed, the manifest remembers the model id so subsequent searches use the same one.

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
