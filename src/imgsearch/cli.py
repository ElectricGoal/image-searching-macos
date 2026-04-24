"""Typer CLI entry point."""

from __future__ import annotations

# faiss-cpu and PyTorch both bundle libomp on macOS arm64. Loading both in the
# same process aborts with "OMP: Error #15 — multiple copies of the OpenMP
# runtime". KMP_DUPLICATE_LIB_OK lets libomp tolerate the duplicate load;
# OMP_NUM_THREADS=1 then prevents the two runtimes from fighting over the same
# thread pool (without this the process still SIGSEGVs on the first faiss
# search after torch has been used). Embedding is GPU-bound on MPS and FAISS
# does exact brute-force search on folders of tens of thousands of images in
# microseconds, so losing CPU parallelism here is not a meaningful regression.
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import typer  # noqa: E402

from imgsearch import __version__
from imgsearch.commands import clean as clean_cmd
from imgsearch.commands import dedup as dedup_cmd
from imgsearch.commands import index as index_cmd
from imgsearch.commands import search as search_cmd
from imgsearch.commands import status as status_cmd

app = typer.Typer(
    name="imgsearch",
    help="Semantic image search for macOS — MLX-accelerated.",
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
    """imgsearch — semantic image search for macOS."""


if __name__ == "__main__":  # pragma: no cover
    app()
