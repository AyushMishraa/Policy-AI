"""
Speech-to-Text module using faster-whisper (local, no API key needed).

Records from the default microphone until silence is detected,
then transcribes and returns the text.
"""

import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger("stt")

# Silence detection thresholds
SILENCE_THRESHOLD   = 0.01    # RMS amplitude below this = silence
SILENCE_DURATION    = 1.5     # seconds of silence to stop recording
MAX_RECORD_SECONDS  = 30      # hard cap


class SpeechToText:
    def __init__(self, model_name: str = "base"):
        self._model_name = model_name
        self._model      = None

    def load(self):
        """Load the Whisper model (call once at startup — takes a few seconds)."""
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_name,
                device="cpu",
                compute_type="int8",
            )
            logger.info(f"Whisper model '{self._model_name}' loaded ✓")
        except ImportError:
            logger.warning(
                "faster-whisper not installed. "
                "Voice input disabled. Run: pip install faster-whisper"
            )

    def is_available(self) -> bool:
        return self._model is not None

    def listen(self) -> Optional[str]:
        """
        Record audio from the microphone until silence, then transcribe.
        Returns transcribed text or None on failure.
        """
        if not self.is_available():
            logger.error("Whisper model not loaded — cannot transcribe")
            return None

        try:
            import numpy as np
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            logger.error("sounddevice / soundfile not installed")
            return None

        sample_rate = 16000
        channels    = 1

        logger.info("🎙  Listening… (speak now, pause to stop)")

        frames    = []
        silence_s = 0.0
        chunk_s   = 0.1  # 100 ms chunks
        chunk_len = int(sample_rate * chunk_s)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        ) as stream:
            total_s = 0.0
            while total_s < MAX_RECORD_SECONDS:
                data, _ = stream.read(chunk_len)
                frames.append(data.copy())
                total_s += chunk_s

                rms = float(np.sqrt(np.mean(data ** 2)))
                if rms < SILENCE_THRESHOLD:
                    silence_s += chunk_s
                    if silence_s >= SILENCE_DURATION and total_s > 1.0:
                        break
                else:
                    silence_s = 0.0

        audio = np.concatenate(frames, axis=0).flatten()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, sample_rate)

        try:
            segments, info = self._model.transcribe(tmp_path, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
            logger.info(f"Transcribed: {text!r}")
            return text if text else None
        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            return None
        finally:
            os.unlink(tmp_path)
