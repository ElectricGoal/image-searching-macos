"""Shared CLI helpers: platform guard, model resolution, Rich console."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from imgsearch.config import DEFAULT_MODEL_ALIAS, ModelSpec, check_platform, resolve_model

console = Console()
err_console = Console(stderr=True)


def preflight(skip_platform: bool = False) -> None:
    """Run startup checks, printing a friendly error and exiting on failure."""
    if skip_platform:
        return
    try:
        check_platform()
    except Exception as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def resolve_folder(folder: Path) -> Path:
    """Resolve folder argument; error out if missing."""
    folder = folder.expanduser().resolve()
    if not folder.exists():
        err_console.print(f"[red]error:[/red] folder not found: {folder}")
        raise typer.Exit(code=2)
    if not folder.is_dir():
        err_console.print(f"[red]error:[/red] not a directory: {folder}")
        raise typer.Exit(code=2)
    return folder


def resolve_model_arg(model: str | None) -> tuple[str, ModelSpec]:
    alias = model or DEFAULT_MODEL_ALIAS
    try:
        spec = resolve_model(alias)
    except ValueError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    return alias, spec
