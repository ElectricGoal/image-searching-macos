"""`imgsearch clean <folder>` — delete the .imgsearch directory."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from imgsearch.commands._common import console, err_console, resolve_folder
from imgsearch.config import INDEX_DIR_NAME


def run(
    folder: Path = typer.Argument(..., help="Folder whose index should be removed."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    folder = resolve_folder(folder)
    target = folder / INDEX_DIR_NAME
    if not target.exists():
        err_console.print(f"[yellow]nothing to clean at[/yellow] {target}")
        raise typer.Exit(code=0)
    if not force:
        confirm = typer.confirm(f"Delete {target}?", default=False)
        if not confirm:
            console.print("aborted")
            raise typer.Exit(code=1)
    shutil.rmtree(target)
    console.print(f"[green]removed[/green] {target}")
