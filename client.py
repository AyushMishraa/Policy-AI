"""
Terminal client for the Policy Agent.

Features:
  - Rich formatted output (colours, spinners, source tables)
  - Text mode  : type questions directly
  - Voice mode : press Enter to record, Whisper transcribes, TTS reads reply
  - /voice     : toggle voice input on/off mid-session
  - /tts       : toggle TTS output on/off mid-session
  - /stats     : show server queue + session stats
  - /help      : list commands
  - /quit      : exit
"""

import asyncio
import json
import os
import sys
import threading
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from config import config
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

console = Console()

BANNER = """
[bold cyan]╔══════════════════════════════════════════╗[/]
[bold cyan]║      Policy Agent  —  Terminal Client     ║[/]
[bold cyan]╚══════════════════════════════════════════╝[/]
[dim]Commands: /voice  /tts  /stats  /help  /quit[/]
"""

HELP_TEXT = """
[bold]Available commands:[/]
  [cyan]/voice[/]   Toggle voice INPUT  (mic → Whisper STT)
  [cyan]/tts[/]     Toggle voice OUTPUT (TTS reads the answer)
  [cyan]/stats[/]   Show server queue & session statistics
  [cyan]/help[/]    Show this help message
  [cyan]/quit[/]    Disconnect and exit

[bold]Query modes:[/]
  Just type your question and press Enter for text mode.
  In voice mode, press Enter to start recording; pause to stop.

[bold]Priority (optional prefix):[/]
  [yellow]!h [/]  High priority query   e.g.  !h What is the leave policy?
  [yellow]!l [/]  Low  priority query   e.g.  !l Summarise all IT policies
"""


class PolicyClient:
    def __init__(self):
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._voice_input = False
        self._tts_output  = False
        self._current_req: Optional[str] = None
        self._response_buf = ""
        self._stt  = SpeechToText(model_name=config.WHISPER_MODEL)
        self._tts  = TextToSpeech(rate=config.TTS_VOICE_RATE, enabled=config.TTS_ENABLED)

    # ── Connect ────────────────────────────────────────────────────────────

    async def connect(self):
        try:
            self._reader, self._writer = await asyncio.open_connection(
                config.SERVER_HOST, config.SERVER_PORT
            )
        except ConnectionRefusedError:
            console.print(
                f"[red]Cannot connect to server at "
                f"{config.SERVER_HOST}:{config.SERVER_PORT}[/]\n"
                "[dim]Is the server running?  python main.py server[/]"
            )
            sys.exit(1)

        console.print(BANNER)

        # Load voice models in background threads to avoid blocking
        if config.TTS_ENABLED:
            threading.Thread(target=self._tts.load, daemon=True).start()
        threading.Thread(target=self._stt.load, daemon=True).start()

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self):
        await self.connect()

        recv_task = asyncio.create_task(self._receive_loop())

        try:
            await self._input_loop()
        finally:
            recv_task.cancel()
            if self._writer:
                self._writer.close()

    # ── Input loop (user → server) ─────────────────────────────────────────

    async def _input_loop(self):
        loop = asyncio.get_event_loop()

        while True:
            # Build prompt label
            mode_label = (
                "[cyan]🎙 voice[/cyan]" if self._voice_input
                else "[green]✎ text[/green]"
            )
            tts_label = " [magenta]🔊[/magenta]" if self._tts_output else ""
            prompt = f"{mode_label}{tts_label} [bold]›[/bold] "
            console.print(prompt, end="")

            # Read input in executor so the event loop isn't blocked
            raw = await loop.run_in_executor(None, sys.stdin.readline)
            text = raw.strip()

            if not text:
                if self._voice_input:
                    text = await self._record_voice()
                    if not text:
                        console.print("[yellow]Nothing heard — try again.[/]")
                        continue
                else:
                    continue

            # ── Commands ──────────────────────────────────────────────────
            if text.lower() == "/quit":
                console.print("[dim]Goodbye.[/]")
                break

            if text.lower() == "/voice":
                self._voice_input = not self._voice_input
                state = "ON" if self._voice_input else "OFF"
                console.print(f"[cyan]Voice input {state}[/]")
                if self._voice_input and not self._stt.is_available():
                    console.print(
                        "[yellow]Whisper model not yet loaded — "
                        "wait a moment then try again.[/]"
                    )
                continue

            if text.lower() == "/tts":
                self._tts_output = not self._tts_output
                state = "ON" if self._tts_output else "OFF"
                console.print(f"[magenta]TTS output {state}[/]")
                continue

            if text.lower() == "/stats":
                await self._send({"type": "stats"})
                await asyncio.sleep(0.5)
                continue

            if text.lower() == "/help":
                console.print(HELP_TEXT)
                continue

            # ── Priority prefix ───────────────────────────────────────────
            priority = 5   # NORMAL
            if text.startswith("!h "):
                priority = 1
                text = text[3:]
                console.print("[yellow]↑ High priority[/]")
            elif text.startswith("!l "):
                priority = 10
                text = text[3:]

            # ── Send query ────────────────────────────────────────────────
            self._response_buf = ""
            await self._send({
                "type":       "query",
                "text":       text,
                "input_mode": "voice" if self._voice_input else "text",
                "priority":   priority,
            })

            # Wait for this request to finish before accepting the next input
            await self._wait_for_done()

    # ── Receive loop (server → display) ───────────────────────────────────

    async def _receive_loop(self):
        while True:
            try:
                raw = await self._reader.readline()
            except (asyncio.IncompleteReadError, ConnectionResetError):
                console.print("\n[red]Server disconnected.[/]")
                break

            if not raw:
                break

            try:
                msg = json.loads(raw.decode().strip())
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")

            if mtype == "status":
                req_id = msg.get("request_id", "")
                label = f"[dim]{req_id}[/dim] " if req_id else ""
                console.print(f"\n{label}[dim]{msg['message']}[/dim]")
                if req_id:
                    self._current_req = req_id

            elif mtype == "chunk":
                text = msg.get("text", "")
                self._response_buf += text
                console.print(text, end="", highlight=False)

            elif mtype == "done":
                req_id = msg.get("request_id", "")
                sources = msg.get("sources", [])
                ptime   = msg.get("processing_time", 0)

                # Finish the printed line
                console.print()

                if sources:
                    self._render_sources(sources)

                console.print(
                    f"[dim]─── {ptime}s  req:{req_id} ───[/dim]\n"
                )

                # TTS — speak the answer (strip source section)
                if self._tts_output and self._response_buf:
                    answer_for_tts = self._response_buf.split("Sources:")[0].strip()
                    asyncio.create_task(self._tts.speak(answer_for_tts))

                self._response_buf = ""
                self._current_req  = None
                self._done_event.set()

            elif mtype == "error":
                console.print(f"\n[red]Error: {msg.get('message')}[/]\n")
                self._done_event.set()

            elif mtype == "stats":
                self._render_stats(msg.get("data", {}))

            elif mtype == "pong":
                pass  # heartbeat, ignore

    # ── Voice helpers ──────────────────────────────────────────────────────

    async def _record_voice(self) -> Optional[str]:
        loop = asyncio.get_event_loop()
        with console.status("[cyan]Recording… speak now[/]", spinner="dots"):
            text = await loop.run_in_executor(None, self._stt.listen)
        if text:
            console.print(f"[dim]Heard:[/dim] [italic]{text}[/italic]")
        return text

    # ── Wait helper ────────────────────────────────────────────────────────

    def __init_done_event(self):
        self._done_event = asyncio.Event()

    async def _wait_for_done(self):
        self._done_event = asyncio.Event()
        try:
            await asyncio.wait_for(self._done_event.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            console.print("[yellow]Request timed out.[/]")

    # ── Send ───────────────────────────────────────────────────────────────

    async def _send(self, payload: dict):
        self._writer.write((json.dumps(payload) + "\n").encode())
        await self._writer.drain()

    # ── Renderers ──────────────────────────────────────────────────────────

    @staticmethod
    def _render_sources(sources: list):
        if not sources:
            return
        tbl = Table(
            title="Sources",
            title_style="bold dim",
            show_header=True,
            header_style="dim",
            border_style="dim",
            padding=(0, 1),
        )
        tbl.add_column("#",      style="dim",  width=3)
        tbl.add_column("Document", style="cyan")
        tbl.add_column("Page",   style="dim",  width=6)
        for i, src in enumerate(sources, 1):
            tbl.add_row(str(i), src.get("source", "—"), str(src.get("page", "—")))
        console.print(tbl)

    @staticmethod
    def _render_stats(data: dict):
        q = data.get("queue", {})
        s = data.get("sessions", {})
        console.print(Panel(
            f"[bold]Queue[/bold]\n"
            f"  Depth       : {q.get('queue_size', '?')}\n"
            f"  Workers     : {q.get('workers', '?')}\n"
            f"  Processed   : {q.get('requests_processed', '?')}\n"
            f"  Avg time    : {q.get('avg_processing_time', '?')}s\n\n"
            f"[bold]Sessions[/bold]\n"
            f"  Connected   : {s.get('total_sessions', '?')}",
            title="Server Stats",
            border_style="cyan",
        ))
