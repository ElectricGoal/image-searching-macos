"""`imgsearch dedup <folder>` — find and remove duplicate images."""

from __future__ import annotations

import shutil
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
from imgsearch.core.duplicate_finder import build_groups, find_exact_groups, find_near_groups
from imgsearch.core.index import Index
from imgsearch.core.manifest import Manifest


def run(
    folder: Path = typer.Argument(..., help="Folder to deduplicate (must be indexed)."),
    threshold: float = typer.Option(
        0.98, "--threshold", "-t", min=0.5, max=1.0,
        help="Cosine similarity threshold for near-duplicate detection.",
    ),
    exact_only: bool = typer.Option(
        False, "--exact-only", help="Only find SHA-1 exact duplicates."
    ),
    keep: str = typer.Option(
        "largest", "--keep",
        help="Which copy to keep: largest, newest, oldest, highest-res.",
    ),
    move_to: Path = typer.Option(
        None, "--move-to", help="Move duplicates here instead of deleting."
    ),
    delete: bool = typer.Option(
        False, "--delete", help="Delete duplicates (prompts for confirmation)."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation prompt."
    ),
    min_group: int = typer.Option(
        2, "--min-group", "-k", min=2, help="Minimum group size to report."
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--no-dry-run",
        help="Print groups only, no file changes (default on).",
    ),
) -> None:
    if keep not in {"largest", "newest", "oldest", "highest-res"}:
        err_console.print(
            f"[red]error:[/red] --keep must be one of: largest, newest, oldest, highest-res"
        )
        raise typer.Exit(code=2)

    preflight(skip_platform=True)
    folder = resolve_folder(folder)

    manifest_path = folder / ".imgsearch" / "manifest.json"
    if not manifest_path.exists():
        err_console.print(f"[red]error:[/red] no index at {folder} — run `imgsearch index` first")
        raise typer.Exit(code=2)

    manifest = Manifest.load(manifest_path)
    try:
        spec = resolve_model(manifest.model_id)
    except ValueError:
        err_console.print("[red]error:[/red] could not resolve model from manifest")
        raise typer.Exit(code=2)

    with Index(folder, spec, manifest.model_alias) as idx:
        idx.open(create=False)
        meta = idx._meta
        vec = idx._vectors

        console.print(f"Scanning index ({idx.count} images)...")

        exact_raw = find_exact_groups(meta)
        console.print(f"  exact duplicates: [bold]{sum(len(g)-1 for g in exact_raw)}[/bold] redundant files in {len(exact_raw)} group(s)")

        near_raw = []
        if not exact_only:
            try:
                near_raw = find_near_groups(vec, meta, threshold)
                console.print(
                    f"  near-duplicates (≥{threshold:.0%}): "
                    f"[bold]{sum(len(g)-1 for g, _ in near_raw)}[/bold] redundant files in {len(near_raw)} group(s)"
                )
            except ValueError as exc:
                err_console.print(f"[yellow]warning:[/yellow] {exc}")

        groups = [
            g for g in build_groups(folder, exact_raw, near_raw, keep)
            if len(g.all_paths) >= min_group
        ]

        if not groups:
            console.print("[green]✓ no duplicates found[/green]")
            return

        total_redundant = sum(len(g.duplicates) for g in groups)
        console.print(
            f"\n[bold]Found {len(groups)} duplicate group(s)[/bold] — "
            f"[red]{total_redundant} redundant image(s)[/red]\n"
        )

        _print_groups(folder, groups)

        if dry_run and not delete and move_to is None:
            console.print(
                "\n[dim]Dry run — no files changed. "
                "Use --delete or --move-to to act.[/dim]"
            )
            return

        # Confirm destructive action
        action_label = f"move to {move_to}" if move_to else "delete"
        if not force:
            confirmed = typer.confirm(
                f"\n{action_label.capitalize()} {total_redundant} file(s)?",
                default=False,
            )
            if not confirmed:
                console.print("aborted")
                raise typer.Exit(code=1)

        removed_ids: list[int] = []
        for group in groups:
            for rel in group.duplicates:
                abs_path = folder / rel
                if not abs_path.exists():
                    continue
                row = meta.get_by_path(rel)
                if row:
                    removed_ids.append(row.faiss_id)
                if move_to is not None:
                    _move_file(abs_path, Path(move_to))
                else:
                    abs_path.unlink()

        if removed_ids:
            idx.apply_deletes(removed_ids)
            idx.commit()

        verb = "moved" if move_to else "deleted"
        console.print(f"[green]✓ {verb} {len(removed_ids)} file(s), index updated[/green]")


def _print_groups(root: Path, groups) -> None:
    for i, group in enumerate(groups, start=1):
        label = (
            "exact (SHA-1)" if group.kind == "exact"
            else f"similar (cosine {group.similarity:.3f})"
        )
        table = Table(title=f"Group {i} — {label}", show_header=False, show_lines=False, box=None)
        table.add_column("tag", style="bold", width=8)
        table.add_column("path", overflow="fold")
        table.add_column("info", style="dim")

        keeper_abs = root / group.keeper
        table.add_row("[green]✓ KEEP[/green]", str(keeper_abs), _file_info(keeper_abs))
        for dup in group.duplicates:
            dup_abs = root / dup
            table.add_row("[red]  DUP[/red]", str(dup_abs), _file_info(dup_abs))
        console.print(table)


def _file_info(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size >= 1_048_576:
            return f"{size / 1_048_576:.1f} MB"
        return f"{size / 1024:.0f} KB"
    except OSError:
        return "missing"


def _move_file(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        stem, suffix = src.stem, src.suffix
        n = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{n}{suffix}"
            n += 1
    shutil.move(str(src), dest)
