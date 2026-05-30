"""
Text-to-Speech module using pyttsx3 (fully offline, cross-platform).
Runs speech synthesis in a separate thread so it doesn't block the async loop.
"""

import asyncio
import logging
import threading
from typing import Optional

logger = logging.getLogger("tts")


class TextToSpeech:
    def __init__(self, rate: int = 175, enabled: bool = True):
        self._rate    = rate
        self._enabled = enabled
        self._engine: Optional[object] = None
        self._lock    = threading.Lock()

    def load(self):
        if not self._enabled:
            return
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            # Pick the first available voice (index 0 is usually the system default)
            voices = self._engine.getProperty("voices")
            if voices:
                self._engine.setProperty("voice", voices[0].id)
            logger.info("pyttsx3 TTS loaded ✓")
        except Exception as exc:
            logger.warning(f"TTS unavailable: {exc}")

    def is_available(self) -> bool:
        return self._engine is not None and self._enabled

    async def speak(self, text: str):
        """
        Async wrapper — runs blocking pyttsx3 in a thread executor
        so the event loop stays free.
        """
        if not self.is_available():
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._speak_sync, text)

    def _speak_sync(self, text: str):
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                logger.warning(f"TTS speak error: {exc}")

    def stop(self):
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
