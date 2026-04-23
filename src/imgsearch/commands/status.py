"""`imgsearch status <folder>` — show index stats."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.table import Table

from imgsearch.commands._common import (
    console,
    err_console,
    preflight,
    resolve_folder,
)
from imgsearch.config import resolve_model
from imgsearch.core.index import Index
from imgsearch.core.manifest import Manifest


def run(
    folder: Path = typer.Argument(..., help="Folder to inspect."),
) -> None:
    preflight(skip_platform=True)
    folder = resolve_folder(folder)

    manifest_path = folder / ".imgsearch" / "manifest.json"
    if not manifest_path.exists():
        err_console.print(f"[yellow]no index found at[/yellow] {folder}")
        raise typer.Exit(code=1)

    manifest = Manifest.load(manifest_path)

    # Re-open the index so count is trustworthy even if manifest is stale.
    try:
        spec = resolve_model(manifest.model_id)
    except ValueError:
        spec = None

    count = manifest.count
    if spec is not None:
        with Index(folder, spec, manifest.model_alias) as idx:
            idx.open(create=False)
            count = idx.count

    table = Table(title=f"imgsearch status — {folder}")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("model", manifest.model_id)
    table.add_row("alias", manifest.model_alias or "-")
    table.add_row("dim", str(manifest.dim))
    table.add_row("image_size", str(manifest.image_size))
    table.add_row("family", manifest.family)
    table.add_row("index_type", manifest.index_type)
    table.add_row("count", str(count))
    table.add_row("created", _fmt(manifest.created_at))
    table.add_row("updated", _fmt(manifest.updated_at))
    console.print(table)


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
