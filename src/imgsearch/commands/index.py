"""`imgsearch index <folder>` — build or update an index."""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from imgsearch.commands._common import (
    console,
    err_console,
    preflight,
    resolve_folder,
    resolve_model_arg,
)
from imgsearch.config import DEFAULT_BATCH_SIZE
from imgsearch.core.embedder import EmbedderProtocol, create_embedder
from imgsearch.core.index import Index
from imgsearch.core.preprocess import ImageLoadError, PreparedImage, load_and_prepare
from imgsearch.core.scanner import DiscoveredFile, scan

# Number of batches to prefetch while the active backend embeds the current one.
_PREFETCH_QUEUE_DEPTH = 2
# Threads for concurrent image decode+hash+resize. PIL releases the GIL during I/O.
_IO_WORKERS = 4


def run(
    folder: Path = typer.Argument(..., help="Folder to index."),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model alias (siglip-so400m, siglip2-base-8bit, clip-vit-b32, clip-vit-l14).",
    ),
    batch: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch", "-b", min=1, max=128, help="Image batch size."
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into subfolders."),
) -> None:
    preflight()
    folder = resolve_folder(folder)
    alias, spec = resolve_model_arg(model)

    console.print(
        f"[bold]imgsearch[/bold] indexing [cyan]{folder}[/cyan] with [green]{spec.display_name}[/green]"
    )

    embedder = create_embedder(spec)
    with Index(folder, spec, alias) as index:
        index.open(create=True)

        console.print("Scanning files...")
        discovered: list[DiscoveredFile] = list(scan(folder, recursive=recursive))
        console.print(f"  found [bold]{len(discovered)}[/bold] image files")

        plan = index.plan(discovered)
        console.print(
            f"  plan: [green]+{len(plan.to_embed)}[/green] new/changed, "
            f"[yellow]~{len(plan.to_touch_mtime)}[/yellow] mtime-only, "
            f"[red]-{len(plan.to_delete_ids)}[/red] removed, "
            f"[dim]{plan.unchanged} unchanged[/dim]"
        )

        index.apply_deletes(plan.to_delete_ids)
        index.apply_mtime_touches(plan.to_touch_mtime)

        if not plan.to_embed:
            index.commit()
            console.print("[bold green]✓ up to date[/bold green]")
            return

        console.print("Loading model (first run downloads weights)...")
        t0 = time.monotonic()
        embedder.load()
        console.print(f"  model ready in {time.monotonic() - t0:.1f}s")

        failures = 0
        processed = 0
        t_start = time.monotonic()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]embedding[/bold blue]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("embedding", total=len(plan.to_embed))
            chunks = _chunks(plan.to_embed, batch)
            for prepared_batch in _prefetch_batches(chunks, spec.image_size):
                ok_files, vectors, sha1s, dims, batch_failures = _embed_prepared(
                    embedder, prepared_batch
                )
                failures += batch_failures
                if ok_files:
                    index.add_batch(ok_files, vectors, sha1s, dims)
                processed += len(prepared_batch)
                progress.update(task, advance=len(prepared_batch))

        index.commit()

        elapsed = time.monotonic() - t_start
        rate = processed / elapsed if elapsed > 0 else 0.0
        console.print(
            f"[bold green]✓ indexed {len(plan.to_embed) - failures} images[/bold green] "
            f"in {elapsed:.1f}s ({rate:.1f} img/s)"
        )
        if failures:
            err_console.print(f"[yellow]warning:[/yellow] skipped {failures} files due to decode errors")


@dataclass
class _PreparedEntry:
    """One successfully loaded image within a prefetched batch."""

    file: DiscoveredFile
    prepared: PreparedImage


def _chunks(items: list[DiscoveredFile], size: int) -> list[list[DiscoveredFile]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _prepare_one(file: DiscoveredFile, image_size: int) -> _PreparedEntry | None:
    """Load + hash + resize a single file. Returns None on decode failure."""
    try:
        prepared = load_and_prepare(file.abs_path, image_size)
        return _PreparedEntry(file=file, prepared=prepared)
    except ImageLoadError:
        return None


def _prefetch_batches(
    chunks: list[list[DiscoveredFile]],
    image_size: int,
) -> list[list[_PreparedEntry | None]]:
    """Yield prepared batches. Each chunk is loaded concurrently via a thread pool
    while the previous batch is being embedded on the GPU.

    Uses a bounded queue so at most _PREFETCH_QUEUE_DEPTH batches sit in RAM
    simultaneously, preventing memory overcommit on large folders.
    """
    result_queue: queue.Queue[list[_PreparedEntry | None]] = queue.Queue(
        maxsize=_PREFETCH_QUEUE_DEPTH
    )
    _SENTINEL = object()

    def _producer() -> None:
        with ThreadPoolExecutor(max_workers=_IO_WORKERS) as pool:
            for chunk in chunks:
                prepared = list(pool.map(_prepare_one, chunk, [image_size] * len(chunk)))
                result_queue.put(prepared)
        result_queue.put(_SENTINEL)  # type: ignore[arg-type]

    import threading
    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        item = result_queue.get()
        if item is _SENTINEL:
            break
        yield item  # type: ignore[misc]

    t.join()


def _embed_prepared(
    embedder: EmbedderProtocol,
    entries: list[_PreparedEntry | None],
) -> tuple[
    list[DiscoveredFile],
    np.ndarray,
    list[str],
    list[tuple[int | None, int | None]],
    int,
]:
    """Embed a pre-loaded batch. Entries that failed to load are None (counted as failures)."""
    ok_files: list[DiscoveredFile] = []
    pil_images: list[Image.Image] = []
    sha1s: list[str] = []
    dims: list[tuple[int | None, int | None]] = []
    failures = sum(1 for e in entries if e is None)

    for entry in entries:
        if entry is None:
            continue
        ok_files.append(entry.file)
        pil_images.append(entry.prepared.image)
        sha1s.append(entry.prepared.sha1)
        dims.append((entry.prepared.orig_width, entry.prepared.orig_height))

    if not ok_files:
        return [], np.zeros((0, embedder.spec.dim), dtype=np.float32), [], [], failures

    vectors = embedder.encode_pil(pil_images)
    # Release PIL images immediately after GPU submission to free RAM.
    for img in pil_images:
        img.close()

    return ok_files, vectors, sha1s, dims, failures
