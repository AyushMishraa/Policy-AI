#!/usr/bin/env python3
"""
Policy Agent — entry point

Usage:
  python main.py server          Start the async TCP server
  python main.py client          Start the terminal client
  python main.py ingest [opts]   Ingest policy documents into the vector store
  python main.py shell           One-shot query (no server needed, for testing)
"""

import argparse
import asyncio
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
    )
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── server ─────────────────────────────────────────────────────────────────

async def _run_server(args):
    from config import config
    from vector_store.store import VectorStore
    from agent.graph import PolicyAgentRunner
    from server import PolicyServer

    console.print("[bold cyan]Initialising Policy Agent server…[/]")

    vs = VectorStore()
    await vs.initialize()

    stats = vs.stats()
    console.print(
        f"[green]Vector store ready[/]  "
        f"[dim]{stats['total_chunks']} chunks in '{stats['collection']}'[/]"
    )

    if stats["total_chunks"] == 0:
        console.print(
            "[yellow]Warning: no policy documents indexed yet.[/]\n"
            "[dim]Run:  python main.py ingest --dir ./policies[/]"
        )

    runner = PolicyAgentRunner(vector_store=vs)
    server = PolicyServer()

    console.print(
        f"[bold green]Server ready[/]  "
        f"[dim]{config.SERVER_HOST}:{config.SERVER_PORT}  "
        f"workers={config.MAX_WORKERS}[/]\n"
    )

    try:
        await server.start(runner)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down…[/]")
        await server.stop()


# ── client ─────────────────────────────────────────────────────────────────

async def _run_client(args):
    from client import PolicyClient
    client = PolicyClient()
    await client.run()


# ── ingest ─────────────────────────────────────────────────────────────────

async def _run_ingest(args):
    from vector_store.store import VectorStore
    from ingestion.pipeline import IngestionPipeline

    vs = VectorStore()
    await vs.initialize()
    pipe = IngestionPipeline(vs)

    total = 0

    if args.dir:
        console.print(f"[cyan]Ingesting directory:[/] {args.dir}")
        total += pipe.ingest_directory(args.dir, recursive=not args.no_recursive)

    if args.file:
        for f in args.file:
            console.print(f"[cyan]Ingesting file:[/] {f}")
            total += pipe.ingest_file(f)

    if args.url:
        for u in args.url:
            console.print(f"[cyan]Ingesting URL:[/] {u}")
            total += pipe.ingest_url(u)

    if args.db_url:
        console.print(f"[cyan]Ingesting database:[/] {args.db_url}")
        total += pipe.ingest_database(
            connection_url=args.db_url,
            query=args.db_query,
            text_column=args.db_text_col,
            source_column=args.db_src_col,
        )

    if total == 0:
        console.print(
            "[yellow]Nothing ingested.[/] "
            "Use --dir, --file, --url, or --db-url"
        )
    else:
        stats = vs.stats()
        console.print(
            f"\n[bold green]Done![/]  +{total} chunks ingested  "
            f"(total in store: {stats['total_chunks']})"
        )


# ── shell (quick local test, no server) ────────────────────────────────────

async def _run_shell(args):
    from vector_store.store import VectorStore
    from agent.graph import PolicyAgentRunner
    from core.models import QueryRequest

    vs = VectorStore()
    await vs.initialize()
    runner = PolicyAgentRunner(vector_store=vs)

    console.print("[bold cyan]Policy Agent Shell[/] [dim](no server — local mode)[/]\n")
    console.print("[dim]Type a question or Ctrl-C to quit.[/]\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        req = QueryRequest(session_id="shell", query_text=query)
        console.print()
        async for chunk in runner.run(req):
            if chunk.error:
                console.print(f"[red]Error: {chunk.error}[/]")
            elif chunk.is_done:
                if chunk.sources:
                    console.print("\n[dim]Sources:[/]")
                    for s in chunk.sources:
                        console.print(f"  [cyan]{s['source']}[/]  p.{s.get('page','')}")
                console.print(f"\n[dim]({chunk.processing_time}s)[/]\n")
            else:
                print(chunk.text, end="", flush=True)


# ── CLI wiring ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="policy-agent",
        description="Async CLI Policy Agent powered by LangGraph + Claude",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    sub = parser.add_subparsers(dest="command")

    # server
    sub.add_parser("server", help="Start the async TCP server")

    # client
    sub.add_parser("client", help="Start the terminal client")

    # shell
    sub.add_parser("shell", help="One-shot local shell (no server needed)")

    # ingest
    ing = sub.add_parser("ingest", help="Ingest policy documents")
    ing.add_argument("--dir",  help="Directory to ingest (recursive by default)")
    ing.add_argument("--file", nargs="+", help="One or more files to ingest")
    ing.add_argument("--url",  nargs="+", help="One or more URLs to scrape and ingest")
    ing.add_argument(
        "--no-recursive", action="store_true",
        help="Do not recurse into sub-directories",
    )
    ing.add_argument("--db-url",      help="SQLAlchemy connection URL for database source")
    ing.add_argument("--db-query",    default="SELECT * FROM policies",
                     help="SQL query to fetch rows")
    ing.add_argument("--db-text-col", default="content",
                     help="Column containing policy text")
    ing.add_argument("--db-src-col",  default="title",
                     help="Column to use as the source label")

    args = parser.parse_args()
    _setup_logging(args.log_level)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "server": _run_server,
        "client": _run_client,
        "ingest": _run_ingest,
        "shell":  _run_shell,
    }

    try:
        asyncio.run(dispatch[args.command](args))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")


if __name__ == "__main__":
    main()
