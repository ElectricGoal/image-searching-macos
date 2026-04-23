"""Typer CLI entry point."""

from __future__ import annotations

import typer

from imgsearch import __version__
from imgsearch.commands import clean as clean_cmd
from imgsearch.commands import dedup as dedup_cmd
from imgsearch.commands import index as index_cmd
from imgsearch.commands import search as search_cmd
from imgsearch.commands import status as status_cmd

app = typer.Typer(
    name="imgsearch",
    help="Semantic image search with automatic backend selection.",
    no_args_is_help=True,
    add_completion=True,
)

app.command("index", help="Build or update the index for a folder.")(index_cmd.run)
app.command("search", help="Search a folder by text or image.")(search_cmd.run)
app.command("dedup", help="Find and remove duplicate images.")(dedup_cmd.run)
app.command("status", help="Show index info for a folder.")(status_cmd.run)
app.command("clean", help="Remove the index directory from a folder.")(clean_cmd.run)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"imgsearch {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """imgsearch — semantic image search for macOS and Linux."""


if __name__ == "__main__":  # pragma: no cover
    app()
