"""`imgsearch search <folder> --text/--image` — similarity query."""

from __future__ import annotations

import json as json_mod
from pathlib import Path

import typer
from rich.table import Table

from imgsearch.commands._common import (
    console,
    err_console,
    preflight,
    resolve_folder,
    resolve_model_arg,
)
from imgsearch.config import DEFAULT_TOP_K
from imgsearch.core.embedder import create_embedder
from imgsearch.core.index import Index


def run(
    folder: Path = typer.Argument(..., help="Folder to search (must be indexed)."),
    text: str = typer.Option(None, "--text", "-t", help="Text query."),
    image: Path = typer.Option(None, "--image", "-i", help="Image query (file path)."),
    k: int = typer.Option(DEFAULT_TOP_K, "-k", "--top-k", min=1, max=500, help="Number of hits."),
    threshold: float = typer.Option(
        0.0, "--threshold", help="Minimum cosine similarity to keep a hit (0-1)."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override model alias."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON (for scripting)."),
) -> None:
    if (text is None) == (image is None):
        err_console.print("[red]error:[/red] provide exactly one of --text or --image")
        raise typer.Exit(code=2)

    preflight()
    folder = resolve_folder(folder)
    alias, spec = resolve_model_arg(model)

    embedder = create_embedder(spec)
    with Index(folder, spec, alias) as index:
        index.open(create=False)

        if index.count == 0:
            err_console.print("[yellow]warning:[/yellow] index is empty")
            if as_json:
                typer.echo("[]")
            raise typer.Exit(code=1)

        embedder.load()
        if text is not None:
            query = embedder.encode_text(text)
            query_label = f'text="{text}"'
        else:
            img_path = image.expanduser().resolve()
            if not img_path.is_file():
                err_console.print(f"[red]error:[/red] query image not found: {img_path}")
                raise typer.Exit(code=2)
            query = embedder.encode_images([img_path])[0]
            query_label = f"image={img_path.name}"

        hits = index.search(query, k)
        hits = [h for h in hits if h.similarity >= threshold]

        if as_json:
            payload = [
                {
                    "path": str(folder / h.rel_path),
                    "rel_path": h.rel_path,
                    "similarity": round(h.similarity, 6),
                }
                for h in hits
            ]
            typer.echo(json_mod.dumps(payload, indent=2))
            return

        _print_table(folder, hits, query_label)


def _print_table(root: Path, hits, label: str) -> None:
    if not hits:
        console.print("[yellow]no matches above threshold[/yellow]")
        return
    table = Table(title=f"top {len(hits)} for {label}", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("score", justify="right", style="cyan")
    table.add_column("path", overflow="fold")
    for i, h in enumerate(hits, start=1):
        table.add_row(str(i), f"{h.similarity:.4f}", str(root / h.rel_path))
    console.print(table)
