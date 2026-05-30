#!/usr/bin/env python3
"""
Policy Agent — Admin CLI (Qdrant backend)

Usage:
  python admin.py stats                         Overall index statistics
  python admin.py list [--limit N]              List all indexed sources
  python admin.py search QUERY [--top N]        Test semantic search
  python admin.py delete --source FILENAME      Remove a source document
  python admin.py delete --all                  Wipe the entire collection
  python admin.py reindex --dir ./policies      Re-ingest a directory
  python admin.py export --out sources.json     Export source list to JSON
"""

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


async def _init_store():
    from vector_store.store import VectorStore
    vs = VectorStore()
    await vs.initialize()
    return vs


# ── stats ──────────────────────────────────────────────────────────────────

async def cmd_stats(_args):
    vs = await _init_store()
    s  = vs.stats()
    console.print(Panel(
        f"[bold]Collection[/bold]  : {s['collection']}\n"
        f"[bold]Qdrant host[/bold] : {s['qdrant_host']}\n"
        f"[bold]Total chunks[/bold]: [cyan]{s['total_chunks']}[/cyan]",
        title="Vector Store Stats",
        border_style="cyan",
    ))


# ── list ───────────────────────────────────────────────────────────────────

async def cmd_list(args):
    vs = await _init_store()
    limit = getattr(args, "limit", 200)

    total = vs._client.count(vs._collection).count
    if total == 0:
        console.print("[yellow]Index is empty — run: python main.py ingest --dir ./policies[/]")
        return

    # Scroll through points to collect payloads
    records, offset = [], None
    fetched = 0
    while fetched < limit:
        batch_size = min(100, limit - fetched)
        batch, offset = vs._client.scroll(
            collection_name=vs._collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        fetched += len(batch)
        if offset is None:
            break

    # Deduplicate by source
    seen: dict = {}
    for rec in records:
        p   = rec.payload or {}
        src = p.get("source", "unknown")
        if src not in seen:
            seen[src] = {"type": p.get("type",""), "path": p.get("source_path",""), "chunks": 0}
        seen[src]["chunks"] += 1

    tbl = Table(title=f"Indexed Sources ({len(seen)} unique)", border_style="dim")
    tbl.add_column("Source",   style="cyan",  no_wrap=True)
    tbl.add_column("Type",     style="dim",   width=10)
    tbl.add_column("Chunks",   style="green", width=8, justify="right")
    tbl.add_column("Path / URL", style="dim")

    for src, info in sorted(seen.items()):
        tbl.add_row(
            src, info["type"], str(info["chunks"]),
            info["path"][:70] + ("…" if len(info["path"]) > 70 else ""),
        )

    console.print(tbl)
    console.print(f"[dim]Total chunks in index: {total}[/]")


# ── search ─────────────────────────────────────────────────────────────────

async def cmd_search(args):
    vs    = await _init_store()
    query = " ".join(args.query)
    top   = getattr(args, "top", 5)

    console.print(f"\n[bold]Searching:[/bold] [cyan]{query!r}[/cyan]\n")
    chunks = vs.search(query, top_k=top)

    if not chunks:
        console.print("[yellow]No results found.[/]")
        return

    for i, c in enumerate(chunks, 1):
        page  = f"  p.{c['page']}" if c.get("page") else ""
        sc    = c["relevance"]
        color = "green" if sc >= 0.5 else "yellow" if sc >= 0.3 else "red"
        console.print(
            f"[{color}][{i}] {c['source']}{page}  relevance={sc:.3f}[/{color}]"
        )
        preview = c["text"][:300]
        console.print(f"[dim]{preview}{'…' if len(c['text']) > 300 else ''}[/dim]\n")


# ── delete ─────────────────────────────────────────────────────────────────

async def cmd_delete(args):
    vs = await _init_store()

    if args.all:
        if not Confirm.ask("[red]Delete ALL chunks from the index?[/red]"):
            console.print("Aborted.")
            return
        vs._client.delete_collection(vs._collection)
        # Re-create empty collection
        from qdrant_client.models import Distance, VectorParams
        vs._client.create_collection(
            collection_name=vs._collection,
            vectors_config=VectorParams(size=vs._vector_size, distance=Distance.COSINE),
        )
        console.print("[green]Collection wiped and recreated empty.[/]")
        return

    if args.source:
        console.print(f"Deleting source: [cyan]{args.source}[/cyan]")
        vs.delete_source(args.source)
        console.print("[green]Done.[/]")
        return

    console.print("[yellow]Specify --source NAME or --all[/]")


# ── reindex ────────────────────────────────────────────────────────────────

async def cmd_reindex(args):
    from ingestion.pipeline import IngestionPipeline
    vs   = await _init_store()
    pipe = IngestionPipeline(vs)

    if args.dir:
        console.print(f"[cyan]Re-indexing directory:[/] {args.dir}")
        n = pipe.ingest_directory(args.dir)
        console.print(f"[green]Done.[/] {n} chunks upserted.")
    elif args.file:
        for f in args.file:
            console.print(f"[cyan]Re-indexing file:[/] {f}")
            n = pipe.ingest_file(f)
            console.print(f"  +{n} chunks")


# ── export ─────────────────────────────────────────────────────────────────

async def cmd_export(args):
    vs = await _init_store()

    if vs._client is None:
        console.print("[yellow]Not connected.[/]")
        return

    records, offset = [], None
    while True:
        batch, offset = vs._client.scroll(
            collection_name=vs._collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break

    seen: dict = {}
    for rec in records:
        p   = rec.payload or {}
        src = p.get("source", "unknown")
        if src not in seen:
            seen[src] = {
                "source":      src,
                "type":        p.get("type",""),
                "source_path": p.get("source_path",""),
                "chunks":      0,
            }
        seen[src]["chunks"] += 1

    out_path = getattr(args, "out", "sources.json")
    with open(out_path, "w") as f:
        json.dump(list(seen.values()), f, indent=2)

    console.print(f"[green]Exported {len(seen)} sources → {out_path}[/]")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    import logging
    from rich.logging import RichHandler
    logging.basicConfig(level=logging.WARNING, handlers=[RichHandler(show_path=False)])

    parser = argparse.ArgumentParser(prog="admin", description="Policy Agent — Admin CLI")
    sub    = parser.add_subparsers(dest="command")

    sub.add_parser("stats", help="Overall index statistics")

    ls = sub.add_parser("list", help="List indexed sources")
    ls.add_argument("--limit", type=int, default=200)

    sr = sub.add_parser("search", help="Test semantic search")
    sr.add_argument("query", nargs="+")
    sr.add_argument("--top",  type=int, default=5)

    dl = sub.add_parser("delete", help="Delete from index")
    dl.add_argument("--source", help="Source name")
    dl.add_argument("--all", action="store_true")

    ri = sub.add_parser("reindex", help="Re-ingest sources")
    ri.add_argument("--dir",  help="Directory")
    ri.add_argument("--file", nargs="+")

    ex = sub.add_parser("export", help="Export source list to JSON")
    ex.add_argument("--out", default="sources.json")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "stats":   cmd_stats,
        "list":    cmd_list,
        "search":  cmd_search,
        "delete":  cmd_delete,
        "reindex": cmd_reindex,
        "export":  cmd_export,
    }
    asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    main()
