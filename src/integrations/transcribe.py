import base64
import io
import logging

from openai import OpenAI

from src.config import WHISPER_API_KEY

logger = logging.getLogger(__name__)

# OpenAI Whisper direto — só para transcrição de áudio
# O texto transcrito é enviado ao Grok para processamento
_client = OpenAI(api_key=WHISPER_API_KEY)


def transcribe_audio(audio_base64: str) -> str:
    """Transcreve áudio base64 para texto usando OpenAI Whisper."""
    audio_bytes = base64.b64decode(audio_base64)
    buf = io.BytesIO(audio_bytes)
    buf.name = "audio.ogg"

    logger.info("Transcrevendo audio (%d bytes) via OpenAI Whisper...", len(audio_bytes))
    transcript = _client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        language="pt",
    )
    logger.info("Transcricao concluida: %.100s", transcript.text)
    return transcript.text
